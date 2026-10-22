// ================================================================
// FlashRT — CUTLASS SM80 BF16 Up-GEMM + gated-GELU epilogue
//
// hidden = gelu(gate) ⊙ (up_activation @ W_up^T)
//
// Reads the gate buffer (from the prior gate GEMM) via VisitorAuxLoad
// inside the epilogue and folds the gated-GELU into the up GEMM,
// eliminating the separate gate_geglu elementwise kernel. BF16 sibling
// of cutlass_sm80_int8_silu_gated.cu (same EVT structure, float accum).
//
// Numerics: the functor rounds the fp32 accumulator to bf16 BEFORE the
// gated activation, mirroring the separate path (cuBLASLt stores the up
// GEMM to bf16, then gate_geglu reads it), so the fused output targets
// bit-identity with the unfused sequence.
// ================================================================
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"

#include "cute/tensor.hpp"

namespace flash_rt {
namespace gemm {
namespace cutlass_bf16_geglu {

using namespace cute;

using ElementA = cutlass::bfloat16_t;
using LayoutA  = cutlass::layout::RowMajor;
using ElementB = cutlass::bfloat16_t;
// Pipeline weights are stored TRANSPOSED: (K, N) row-major memory (the
// safetensors (N,K) is .t()'d at load time, see pi05_rtx.py convert step).
// Same layout bf16_nn's B_desc reads — makes this kernel a drop-in for the
// pipeline's weight buffers. (gemm_runner.h's "B: (N,K)" comment is stale.)
using LayoutB  = cutlass::layout::RowMajor;
using ElementOutput = cutlass::bfloat16_t;
using LayoutC  = cutlass::layout::RowMajor;
using ElementAccumulator = float;
using ElementCompute     = float;

constexpr int AlignmentA = 8;
constexpr int AlignmentB = 8;
constexpr int AlignmentC = 8;

using ArchTag        = cutlass::arch::Sm80;
using OperatorClass  = cutlass::arch::OpClassTensorOp;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
constexpr int NumStages        = 4;
constexpr int EVTEpilogueStages = 1;

using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ThreadblockShape, WarpShape, ElementOutput, AlignmentC, EVTEpilogueStages>;

using AccFetch = cutlass::epilogue::threadblock::VisitorAccFetch;

// ── Gate buffer (M, N) BF16, element-wise via VisitorAuxLoad ──
using GateLoad = cutlass::epilogue::threadblock::VisitorAuxLoad<
    OutputTileThreadMap, cutlass::bfloat16_t, Stride<int64_t, _1, int64_t>>;

// out = up * gated-activation(gate). The activation here is the tanh/
// sigmoid-approx GELU used by gate_geglu (same constant), NOT SiLU. The
// fp32 accumulator is rounded to bf16 first to mirror the separate path
// (cuBLASLt stores up to bf16; gate_geglu reads that bf16).
template <class T>
struct GatedGeluFunctor {
    __device__ T operator()(T up_val, T gate_val) const {
        return impl(up_val, gate_val,
                    typename cutlass::platform::is_floating_point<T>::type{});
    }

private:
    template <class S>
    __device__ S impl(S up, S gate,
                      cutlass::platform::true_type) const {
        float g = float(gate);
        float u = float(cutlass::bfloat16_t(float(up)));
        float a = g / (1.0f + expf(-1.5957691216057308f * g *
                                   (1.0f + 0.044715f * g * g)));
        return S(a * u);
    }

    template <class Arr>
    __device__ Arr impl(Arr const& up, Arr const& gate,
                        cutlass::platform::false_type) const {
        Arr result;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Arr::kElements; ++i) {
            float g = float(gate[i]);
            float u = float(cutlass::bfloat16_t(float(up[i])));
            float a = g / (1.0f + expf(-1.5957691216057308f * g *
                                       (1.0f + 0.044715f * g * g)));
            result[i] = typename Arr::Element(a * u);
        }
        return result;
    }
};

using MulGatedGelu = cutlass::epilogue::threadblock::VisitorCompute<
    GatedGeluFunctor, float, float, cutlass::FloatRoundStyle::round_to_nearest>;

using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, cutlass::bfloat16_t,
    cutlass::FloatRoundStyle::round_to_nearest,
    Stride<int64_t, _1, int64_t>>;

//   AccFetch ──┐
//   GateLoad ──┴─ MulGatedGelu ── StoreD
using EVT_Gated = cutlass::epilogue::threadblock::Sm80EVT<
    MulGatedGelu, AccFetch, GateLoad>;
using EVT_Final = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT_Gated>;

using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
    ElementOutput, LayoutC, AlignmentC,
    ElementAccumulator,
    ElementCompute,
    OperatorClass, ArchTag,
    ThreadblockShape, WarpShape, InstructionShape,
    EVT_Final,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    cutlass::arch::OpMultiplyAdd,
    EVTEpilogueStages
>::GemmKernel;

using GemmDevice = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

static int run(
        void const* act_bf16,  // (M, K) BF16 RowMajor — RMSNormed activation
        void const* up_w_bf16, // (K, N) BF16 RowMajor memory (transposed at load)
        void const* gate_buf,  // (M, N) BF16 — gate GEMM output
        void*       D,         // (M, N) BF16 — gelu(gate) ⊙ up
        int M, int N, int K,
        cudaStream_t stream) {

    cutlass::gemm::GemmCoord problem(M, N, K);

    typename EVT_Final::Arguments evt_args{
        // EVT_Gated::Arguments {AccFetch_args, GateLoad_args, functor_args}
        {
            {},    // AccFetch — no arguments
            {
                const_cast<cutlass::bfloat16_t*>(
                    reinterpret_cast<cutlass::bfloat16_t const*>(gate_buf)),
                cutlass::bfloat16_t{},
                {static_cast<int64_t>(N), _1{},
                 static_cast<int64_t>(M) * static_cast<int64_t>(N)}
            },
            {}     // GatedGeluFunctor — stateless
        },
        // StoreD::Arguments {ptr, stride}
        {
            reinterpret_cast<cutlass::bfloat16_t*>(D),
            {static_cast<int64_t>(N), _1{},
             static_cast<int64_t>(M) * static_cast<int64_t>(N)}
        }
    };

    typename GemmDevice::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem, 1,
        evt_args,
        reinterpret_cast<ElementA const*>(act_bf16),
        reinterpret_cast<ElementB const*>(up_w_bf16),
        nullptr, nullptr,
        static_cast<int64_t>(M) * K,
        static_cast<int64_t>(N) * K,
        0, 0, K, N, N, N);

    GemmDevice gemm;
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr,
            "[cutlass_bf16_geglu] can_implement failed M=%d N=%d K=%d: %d\n",
            M, N, K, static_cast<int>(st));
        return static_cast<int>(st) | 0x10000;
    }

    static void* ws_ptr = nullptr;
    static size_t ws_cap = 0;
    size_t ws_sz = GemmDevice::get_workspace_size(args);
    if (ws_sz > ws_cap) {
        if (ws_ptr) cudaFree(ws_ptr);
        if (cudaMalloc(&ws_ptr, ws_sz) != cudaSuccess) {
            ws_ptr = nullptr; ws_cap = 0; return -1;
        }
        ws_cap = ws_sz;
    }

    st = gemm.initialize(args, ws_ptr, stream);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr,
            "[cutlass_bf16_geglu] init failed M=%d N=%d K=%d: %d\n",
            M, N, K, static_cast<int>(st));
        return static_cast<int>(st) | 0x20000;
    }

    st = gemm.run(stream);
    return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
}

}  // namespace cutlass_bf16_geglu
}  // namespace gemm
}  // namespace flash_rt


// ── C extern for Python binding ──
extern "C" int cutlass_bf16_geglu_gemm(
        void const* act_bf16,
        void const* up_w_bf16,
        void const* gate_buf,
        void* D,
        int M, int N, int K,
        cudaStream_t stream) {
    return flash_rt::gemm::cutlass_bf16_geglu::run(
        act_bf16, up_w_bf16, gate_buf, D, M, N, K, stream);
}
