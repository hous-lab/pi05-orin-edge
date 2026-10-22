//
// mixed-K16 Phase C: fused add+add+GeGLU epilogue for the gate/up mixed path.
//
// Replaces the v1.7 tail chain
//   add_bf16_out(gate_buf, gate_side, gate_buf)      // 3x17MB DRAM
//   add_bf16_out(up_out, up_side, up_out)            // 3x17MB DRAM
//   gate_geglu_vec2(gate_buf, up_out, hidden)        // 3x17MB DRAM
// with ONE pass: h = GELU(g_i8 + g_side) * (u_i8 + u_side).
// DRAM traffic 155MB -> 86MB per layer (seq=526): ~8ms/17 layers.
//
// Numerics: fp32 sums (no intermediate bf16 rounding of g/u) then the SAME
// tanh-approx GELU constants as gate_geglu_vec2 (1.5957691216057308 /
// 0.044715), so the delta vs the chain is only the removed intermediate
// roundings (1-ulp class, direction favorable). BF16-only: the mixed path is
// all-BF16.
#include "activation.cuh"
#include "common.cuh"

__global__ void gateup_side_geglu_bf16_kernel(
        const __nv_bfloat16* __restrict__ gate_i8p,
        const __nv_bfloat16* __restrict__ gate_side,
        const __nv_bfloat16* __restrict__ up_i8p,
        const __nv_bfloat16* __restrict__ up_side,
        __nv_bfloat16* __restrict__ out,
        int n) {
    using T2 = typename packed2<__nv_bfloat16>::type;
    int n2 = n >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        T2 gv = reinterpret_cast<const T2*>(gate_i8p)[idx];
        T2 gs = reinterpret_cast<const T2*>(gate_side)[idx];
        T2 uv = reinterpret_cast<const T2*>(up_i8p)[idx];
        T2 us = reinterpret_cast<const T2*>(up_side)[idx];
        float g0 = to_f32(gv.x) + to_f32(gs.x);
        float g1 = to_f32(gv.y) + to_f32(gs.y);
        float u0 = to_f32(uv.x) + to_f32(us.x);
        float u1 = to_f32(uv.y) + to_f32(us.y);
        float a0 = g0 / (1.0f + expf(-1.5957691216057308f * g0 * (1.0f + 0.044715f * g0 * g0)));
        float a1 = g1 / (1.0f + expf(-1.5957691216057308f * g1 * (1.0f + 0.044715f * g1 * g1)));
        float b0 = a0 * u0;
        float b1 = a1 * u1;
        reinterpret_cast<T2*>(out)[idx] = make_packed2<__nv_bfloat16>(
            from_f32<__nv_bfloat16>(b0), from_f32<__nv_bfloat16>(b1));
    }
}

void gateup_side_geglu_bf16(
        const __nv_bfloat16* gate_i8p, const __nv_bfloat16* gate_side,
        const __nv_bfloat16* up_i8p, const __nv_bfloat16* up_side,
        __nv_bfloat16* out, int n, cudaStream_t stream) {
    int n2 = n >> 1;
    gateup_side_geglu_bf16_kernel<<<(n2 + 255) / 256, 256, 0, stream>>>(
        gate_i8p, gate_side, up_i8p, up_side, out, n);
}

// ═══════════════════════════════════════════════════════════════════════
// Soup-fusion (encoder-down, 2026-10-10): gateup_side_geglu + rowwise INT8
// quantize in ONE pass, for the encoder FFN-down input under the mixed
// gate/up path. Replaces
//   gateup_side_geglu_bf16(...)            // writes bf16 hidden to DRAM
//   quantize_int8_rowwise_fast(hidden,...) // re-reads hidden, emits int8
// The h values are computed identically (fp32 sums, same GELU constants)
// and rounded to bf16 (__float2bfloat16, same as from_f32) BEFORE the amax
// scan, so the quantize input is bit-identical to the bf16 hidden the
// unfused pair would have round-tripped through DRAM.
//
// Quantize formula copied verbatim from quantize_int8_rowwise_fast_kernel
// (quantize.cu): order-free fmaxf amax, scale = fmaxf(amax/127, 1e-10),
// __float2int_rn, clamp to [-127, 127] — bit-exact codes AND scales.
//
// Layout: one block per row (cols = ENC_H = 16384 → CHUNKS=2, 1024 threads,
// same launch shape as quantize_int8_rowwise_fast<2>). Each thread keeps its
// h values in registers between the amax reduction and the int8 store, so
// the row is produced and consumed with zero extra DRAM round-trips.
template <int CHUNKS>
__global__ void gateup_side_geglu_int8_kernel(
        const __nv_bfloat16* __restrict__ gate_i8p,
        const __nv_bfloat16* __restrict__ gate_side,
        const __nv_bfloat16* __restrict__ up_i8p,
        const __nv_bfloat16* __restrict__ up_side,
        int8_t* __restrict__ out,
        float* __restrict__ scales,
        int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const int nvec = cols / 8;

    const uint4* g4  = reinterpret_cast<const uint4*>(gate_i8p  + (size_t)row * cols);
    const uint4* gs4 = reinterpret_cast<const uint4*>(gate_side + (size_t)row * cols);
    const uint4* u4  = reinterpret_cast<const uint4*>(up_i8p  + (size_t)row * cols);
    const uint4* us4 = reinterpret_cast<const uint4*>(up_side + (size_t)row * cols);
    uint4 hv[CHUNKS];

    float tmax = 0.0f;
    #pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int vi = tid + c * nthr;
        if (vi < nvec) {
            const uint4 gv  = g4[vi];
            const uint4 gsv = gs4[vi];
            const uint4 uv  = u4[vi];
            const uint4 usv = us4[vi];
            const __nv_bfloat162* g2  = reinterpret_cast<const __nv_bfloat162*>(&gv);
            const __nv_bfloat162* gs2 = reinterpret_cast<const __nv_bfloat162*>(&gsv);
            const __nv_bfloat162* u2  = reinterpret_cast<const __nv_bfloat162*>(&uv);
            const __nv_bfloat162* us2 = reinterpret_cast<const __nv_bfloat162*>(&usv);
            __nv_bfloat162* h2 = reinterpret_cast<__nv_bfloat162*>(&hv[c]);
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                float g0 = to_f32(g2[p].x) + to_f32(gs2[p].x);
                float g1 = to_f32(g2[p].y) + to_f32(gs2[p].y);
                float u0 = to_f32(u2[p].x) + to_f32(us2[p].x);
                float u1 = to_f32(u2[p].y) + to_f32(us2[p].y);
                float a0 = g0 / (1.0f + expf(-1.5957691216057308f * g0 * (1.0f + 0.044715f * g0 * g0)));
                float a1 = g1 / (1.0f + expf(-1.5957691216057308f * g1 * (1.0f + 0.044715f * g1 * g1)));
                float b0 = a0 * u0;
                float b1 = a1 * u1;
                h2[p] = make_packed2<__nv_bfloat16>(
                    from_f32<__nv_bfloat16>(b0), from_f32<__nv_bfloat16>(b1));
                tmax = fmaxf(tmax, fabsf(__bfloat162float(h2[p].x)));
                tmax = fmaxf(tmax, fabsf(__bfloat162float(h2[p].y)));
            }
        }
    }

    for (int off = 16; off > 0; off >>= 1) {
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, off));
    }

    __shared__ float warp_max[32];
    const int wid = tid >> 5;
    const int lid = tid & 31;
    if (lid == 0) {
        warp_max[wid] = tmax;
    }
    __syncthreads();

    if (wid == 0) {
        const int nwarp = (nthr + 31) >> 5;
        tmax = (lid < nwarp) ? warp_max[lid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, off));
        }
    }

    __shared__ float scale_s;
    if (tid == 0) {
        const float s = fmaxf(tmax / 127.0f, 1e-10f);
        scales[row] = s;
        scale_s = s;
    }
    __syncthreads();

    const float inv_s = 1.0f / scale_s;
    uint2* o2 = reinterpret_cast<uint2*>(out + (size_t)row * cols);
    #pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int vi = tid + c * nthr;
        if (vi < nvec) {
            const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&hv[c]);
            int q[8];
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                q[2 * p]     = __float2int_rn(__bfloat162float(h2[p].x) * inv_s);
                q[2 * p + 1] = __float2int_rn(__bfloat162float(h2[p].y) * inv_s);
            }
            #pragma unroll
            for (int p = 0; p < 8; ++p) {
                q[p] = (q[p] < -127) ? -127 : ((q[p] > 127) ? 127 : q[p]);
            }
            uint2 packed;
            packed.x = (uint32_t)(uint8_t)q[0] | ((uint32_t)(uint8_t)q[1] << 8) |
                       ((uint32_t)(uint8_t)q[2] << 16) | ((uint32_t)(uint8_t)q[3] << 24);
            packed.y = (uint32_t)(uint8_t)q[4] | ((uint32_t)(uint8_t)q[5] << 8) |
                       ((uint32_t)(uint8_t)q[6] << 16) | ((uint32_t)(uint8_t)q[7] << 24);
            o2[vi] = packed;
        }
    }
}

void gateup_side_geglu_int8(
        const __nv_bfloat16* gate_i8p, const __nv_bfloat16* gate_side,
        const __nv_bfloat16* up_i8p, const __nv_bfloat16* up_side,
        int8_t* out, float* scales, int rows, int cols,
        cudaStream_t stream) {
    if (cols % 128 != 0 || cols > 8 * 1024 * 8) {
        // Unsupported shape: fall back to the unfused pair (geglu → bf16
        // scratch is not available here; caller must keep the two-kernel
        // path). This wrapper therefore requires cols % 128 == 0.
        return;
    }
    const int nvec = cols / 8;
    int chunks = 1;
    while (nvec / chunks > 1024) chunks *= 2;
    int threads = ((nvec / chunks) + 31) & ~31;
    switch (chunks) {
        case 1: gateup_side_geglu_int8_kernel<1><<<rows, threads, 0, stream>>>(
                    gate_i8p, gate_side, up_i8p, up_side,
                    out, scales, rows, cols); break;
        case 2: gateup_side_geglu_int8_kernel<2><<<rows, threads, 0, stream>>>(
                    gate_i8p, gate_side, up_i8p, up_side,
                    out, scales, rows, cols); break;
        case 4: gateup_side_geglu_int8_kernel<4><<<rows, threads, 0, stream>>>(
                    gate_i8p, gate_side, up_i8p, up_side,
                    out, scales, rows, cols); break;
        default: break;  // unreachable given the cols guard above
    }
}
