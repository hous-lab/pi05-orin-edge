//
// mixed-K16 (Q2-ext Plan A): masked fused residual-add + RMSNorm -> INT8.
//
// Same fused contract as residual_add_rms_norm_int8_rowwise (norm.cu), with
// the outlier-channel split validated by the Q2-ext GEMM-level sim: the
// per-row INT8 scale is computed over NON-outlier channels only (removes the
// rowwise-scale hijack - one 40x channel otherwise coarsens the whole row's
// quant step ~40x). Outlier positions of the INT8 output are forced to 0 so
// the full-K INT8 GEMM sees exactly the non-outlier contribution (the mix
// weights zero the outlier K-columns on the weight side, so the act values
// there are don't-care - belt and braces). The gathered BF16 outlier values
// are emitted as an (rows, n_out) operand for the BF16 side GEMM.
//
// mask_lut: (cols,) uint8, 1 = outlier (excluded from amax, forced 0 in int8)
// out_idx:  (n_out,) int32 channel indices gathered into xn_out (rows, n_out)
#include "norm.cuh"
#include "common.cuh"

__global__ void residual_add_rms_norm_int8_rowwise_masked_kernel(
        __nv_bfloat16* __restrict__ residual,
        const __nv_bfloat16* __restrict__ x,
        const __nv_bfloat16* __restrict__ weight,
        int8_t* __restrict__ out,
        float* __restrict__ scales,
        __nv_bfloat16* __restrict__ xn_out,
        const int* __restrict__ out_idx,
        int n_out,
        const uint8_t* __restrict__ mask_lut,
        int rows, int cols, float eps) {
    extern __shared__ float smem[];
    float* partial = smem + cols;

    int row = blockIdx.x;
    if (row >= rows) return;

    __nv_bfloat16* res_row = residual + (int64_t)row * cols;
    const __nv_bfloat16* x_row = x + (int64_t)row * cols;
    int8_t* out_row = out + (int64_t)row * cols;

    // Pass 1: residual += x, stage into smem, accumulate sum of squares
    float sum_sq = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float ri = __bfloat162float(res_row[i]) + __bfloat162float(x_row[i]);
        res_row[i] = __float2bfloat16(ri);
        smem[i] = ri;
        sum_sq += ri * ri;
    }
    float rms = rsqrtf(block_reduce_sum(sum_sq, partial) / cols + eps);

    // Pass 2: normalize, amax over NON-outlier channels only
    float max_abs = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = smem[i] * rms * __bfloat162float(weight[i]);
        smem[i] = v;
        if (!mask_lut[i]) max_abs = fmaxf(max_abs, fabsf(v));
    }
    float scale = fmaxf(block_reduce_max(max_abs, partial) / 127.f, 1e-12f);
    if (threadIdx.x == 0) scales[row] = scale;
    float inv_s = 1.f / scale;

    // Gather outlier channels' BF16 values for the BF16 side GEMM. n_out
    // (16/32) is far below blockDim.x, so only the first n_out threads act.
    for (int j = threadIdx.x; j < n_out; j += blockDim.x) {
        xn_out[row * n_out + j] = __float2bfloat16(smem[out_idx[j]]);
    }

    // Pass 3: write INT8 (outlier positions forced to 0).
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = smem[i] * inv_s;
        out_row[i] = mask_lut[i]
            ? (int8_t)0
            : (int8_t)__float2int_rn(fmaxf(-127.f, fminf(127.f, v)));
    }
}

void residual_add_rms_norm_int8_rowwise_masked(
        __nv_bfloat16* residual, const __nv_bfloat16* x,
        const __nv_bfloat16* weight,
        int8_t* out, float* scales,
        __nv_bfloat16* xn_out, const int* out_idx, int n_out,
        const uint8_t* mask_lut,
        int seq_len, int dim, float eps,
        cudaStream_t stream) {
    int smem = (dim + 32) * sizeof(float);
    residual_add_rms_norm_int8_rowwise_masked_kernel<<<seq_len, 256, smem, stream>>>(
        residual, x, weight, out, scales, xn_out, out_idx, n_out, mask_lut,
        seq_len, dim, eps);
}
