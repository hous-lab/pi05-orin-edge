// ================================================================
// FlashRT — Activation kernel declarations
// GeLU, SiLU, Gate*Act*Mul (BF16/FP16 and fused FP8 variants)
// Supports: __half (FP16), __nv_bfloat16 (BF16)
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ── BF16 (original signatures, backward compatible) ──

// NOTE: these were previously misnamed `gate_silu_mul*`; they compute the
// tanh/sigmoid-approx GELU (GeGLU), not SiLU. Renamed to `gate_geglu*`.
void gate_geglu(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                   __nv_bfloat16* out, int n, cudaStream_t stream = 0);

// Vectorized GeGLU (packed2): same math as gate_geglu, bit-identical.
void gate_geglu_vec2(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                     __nv_bfloat16* out, int n, cudaStream_t stream = 0);

// mixed-K16 Phase C: fused add+add+GeGLU epilogue (bf16-only).
// h = GELU_tanh(gate_i8p + gate_side) * (up_i8p + up_side), elementwise over n
// elements. fp32 sums (no intermediate bf16 rounding). Replaces
// add_bf16_out x2 + gate_geglu[_vec2] in the gate/up mixed path tail.
void gateup_side_geglu_bf16(
        const __nv_bfloat16* gate_i8p, const __nv_bfloat16* gate_side,
        const __nv_bfloat16* up_i8p, const __nv_bfloat16* up_side,
        __nv_bfloat16* out, int n, cudaStream_t stream = 0);

// Soup-fusion: same GeGLU math as gateup_side_geglu_bf16, but also emits the
// rowwise-INT8 quantized hidden + per-row scales in the same pass (formula
// identical to quantize_int8_rowwise_fast — bit-exact codes/scales). Requires
// cols % 128 == 0 and cols <= 64K; callers must keep the unfused fallback.
void gateup_side_geglu_int8(
        const __nv_bfloat16* gate_i8p, const __nv_bfloat16* gate_side,
        const __nv_bfloat16* up_i8p, const __nv_bfloat16* up_side,
        int8_t* out, float* scales, int rows, int cols,
        cudaStream_t stream = 0);
void gelu_inplace(__nv_bfloat16* x, int n, cudaStream_t stream = 0);
void relu2_inplace_bf16(__nv_bfloat16* x, int n, cudaStream_t stream = 0);

// G7.11 — fused (bias add + GELU(tanh)) in-place on bf16 tensor.
// x: (M, N) bf16; bias: (N,) bf16 broadcast over rows. Replaces
// add_bias_bf16 + gelu_inplace pair (2 launches -> 1).
void bias_gelu_inplace_bf16(__nv_bfloat16* x, const __nv_bfloat16* bias,
                              int M, int N, cudaStream_t stream = 0);

// Strict variant matching add_bias_bf16 + gelu_inplace numerics: the
// bias-add result is rounded back to BF16 before applying GELU.
void bias_gelu_inplace_bf16_strict(__nv_bfloat16* x,
                                   const __nv_bfloat16* bias,
                                   int M, int N, cudaStream_t stream = 0);

void gate_geglu_merged(const __nv_bfloat16* merged, __nv_bfloat16* out,
                           int seq, int half_dim, cudaStream_t stream = 0);

void gate_geglu_merged_fp8(const __nv_bfloat16* merged, __nv_fp8_e4m3* out,
                               int seq, int half_dim,
                               const float* d_scale, cudaStream_t stream = 0);

// ── FP16 variants ──

void gate_geglu_fp16(const __half* gate, const __half* up,
                        __half* out, int n, cudaStream_t stream = 0);

void gelu_inplace_fp16(__half* x, int n, cudaStream_t stream = 0);

void gate_geglu_merged_fp16(const __half* merged, __half* out,
                                int seq, int half_dim, cudaStream_t stream = 0);

// Element-wise multiply: out[i] = a[i] * b[i] for i in [0, n).
// FP16 inputs and output, FP32 multiply.  Used by R3.1 split-G7 path
// to combine GELU(gate) with up after two separate GEMMs.
void mul_fp16(const __half* a, const __half* b, __half* out,
              int n, cudaStream_t stream = 0);

void gate_geglu_merged_fp8_fp16(const __half* merged, __nv_fp8_e4m3* out,
                                    int seq, int half_dim,
                                    const float* d_scale, cudaStream_t stream = 0);

// Split SiLU: separate gate and up buffers → FP8 output
// Matches pi05 silu_mul_split_fp8_k (split gate+up GEMMs for L2 optimization)
void silu_mul_split_fp8_fp16(const __half* gate, const __half* up,
                              __nv_fp8_e4m3* out, int n,
                              const float* d_scale, cudaStream_t stream = 0);

// GeGLU with fused per-tensor amax: writes fp16 output and folds its
// abs-max into a caller-zeroed device accumulator (for fused dynamic
// FP8 quantize). d_amax must be memset to 0 by the caller first.
void gate_geglu_amax_fp16(const __half* gate, const __half* up, __half* out,
                          float* d_amax, int n, cudaStream_t stream = 0);
