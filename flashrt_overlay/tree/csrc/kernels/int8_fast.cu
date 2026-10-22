// ================================================================
// FlashRT — vectorized single-pass INT8 norm/quant fast paths (pi05 Orin).
//
// Part of the B-4 quant-fusion work: the stock two-pass
// quantize_int8_rowwise re-reads the activation from DRAM on the FFN-down
// site (662 x 16384 bf16 = 27 MB >> ~4 MB L2), and the stock norm→int8
// kernels run at 60-95 GB/s with scalar loads. These variants keep each
// thread's elements in registers between reductions and store, so every row
// is read from DRAM exactly once, with uint4/uint2 vector accesses.
//
// Numerics contract vs the stock kernels:
//  - amax/scale math is order-free (fmaxf) — identical scales.
//  - sum_sq accumulation order differs (consecutive per thread vs strided),
//    so rms (and downstream int8 codes) can differ in the last ulp —
//    validate with cos, not bit-exactness.
// ================================================================

#include "norm.cuh"
#include "common.cuh"

// Pack 8 int8 lane values into one 8-byte word for vectorized stores.
__device__ __forceinline__ uint2 pack_int8x8(const int (&q)[8]) {
    uint2 r;
    r.x = (uint32_t)(uint8_t)q[0] | ((uint32_t)(uint8_t)q[1] << 8) |
          ((uint32_t)(uint8_t)q[2] << 16) | ((uint32_t)(uint8_t)q[3] << 24);
    r.y = (uint32_t)(uint8_t)q[4] | ((uint32_t)(uint8_t)q[5] << 8) |
          ((uint32_t)(uint8_t)q[6] << 16) | ((uint32_t)(uint8_t)q[7] << 24);
    return r;
}

// Dispatch helper shared by the fast kernels: cols must be uint4-aligned and
// the per-thread chunk count stays <= 8 (cols up to 64K); else fall back.
__device__ __host__ inline bool int8_fast_ok(int cols) {
    return (cols % 128 == 0) && (cols <= 8 * 1024 * 8);
}

// ── rms_norm_int8_rowwise_fast ──
template <int CHUNKS>
__global__ void rms_norm_int8_rowwise_fast_kernel(
        const __nv_bfloat16* __restrict__ x,
        const __nv_bfloat16* __restrict__ weight,
        int8_t* __restrict__ out,
        float* __restrict__ scales,
        int rows, int cols, float eps) {
    extern __shared__ float smem[];   // block-reduce partials
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const int nvec = cols / 8;

    const uint4* x4 = reinterpret_cast<const uint4*>(x + (int64_t)row * cols);
    const uint4* w4 = reinterpret_cast<const uint4*>(weight);
    uint4 xv[CHUNKS], wv[CHUNKS];

    float local = 0.f;
    #pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int vi = tid + c * nthr;
        if (vi < nvec) {
            xv[c] = x4[vi];
            wv[c] = w4[vi];
            const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv[c]);
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float a = __bfloat162float(x2[p].x);
                const float b = __bfloat162float(x2[p].y);
                local += a * a + b * b;
            }
        }
    }
    const float rms = rsqrtf(block_reduce_sum(local, smem) / cols + eps);

    float amax = 0.f;
    #pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int vi = tid + c * nthr;
        if (vi < nvec) {
            const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv[c]);
            const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv[c]);
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float v0 = __bfloat162float(x2[p].x) * rms * __bfloat162float(w2[p].x);
                const float v1 = __bfloat162float(x2[p].y) * rms * __bfloat162float(w2[p].y);
                amax = fmaxf(amax, fabsf(v0));
                amax = fmaxf(amax, fabsf(v1));
            }
        }
    }
    const float scale = fmaxf(block_reduce_max(amax, smem) / 127.f, 1e-12f);
    if (tid == 0) scales[row] = scale;
    const float inv_s = 1.f / scale;

    uint2* o2 = reinterpret_cast<uint2*>(out + (int64_t)row * cols);
    #pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int vi = tid + c * nthr;
        if (vi < nvec) {
            const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv[c]);
            const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv[c]);
            int q[8];
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float v0 = __bfloat162float(x2[p].x) * rms * __bfloat162float(w2[p].x);
                const float max0 = v0 * inv_s;
                const float v1 = __bfloat162float(x2[p].y) * rms * __bfloat162float(w2[p].y);
                const float max1 = v1 * inv_s;
                q[2 * p]     = __float2int_rn(max0);
                q[2 * p + 1] = __float2int_rn(max1);
            }
            o2[vi] = pack_int8x8(q);
        }
    }
}

// ── residual_add_rms_norm_int8_rowwise_fast ──
template <int CHUNKS>
__global__ void residual_add_rms_norm_int8_rowwise_fast_kernel(
        __nv_bfloat16* __restrict__ residual,
        const __nv_bfloat16* __restrict__ x,
        const __nv_bfloat16* __restrict__ weight,
        int8_t* __restrict__ out,
        float* __restrict__ scales,
        int rows, int cols, float eps) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const int nvec = cols / 8;
    __nv_bfloat16* rraw = residual + (int64_t)row * cols;
    const uint4* r4 = reinterpret_cast<const uint4*>(rraw);
    const uint4* x4 = reinterpret_cast<const uint4*>(x + (int64_t)row * cols);
    const uint4* w4 = reinterpret_cast<const uint4*>(weight);
    uint4 rv[CHUNKS], xv[CHUNKS], wv[CHUNKS];

    float local = 0.f;
    __nv_bfloat162* rs2 = reinterpret_cast<__nv_bfloat162*>(rraw);
    float fa[8 * CHUNKS], fb[8 * CHUNKS];   // unrounded sums, fp32 (as smem
                                            // staging does in the stock kernel)
    #pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int vi = tid + c * nthr;
        if (vi < nvec) {
            rv[c] = r4[vi];
            xv[c] = x4[vi];
            wv[c] = w4[vi];
        }
    }
    #pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int vi = tid + c * nthr;
        if (vi < nvec) {
            const __nv_bfloat162* r2 = (const __nv_bfloat162*)&rv[c];
            const __nv_bfloat162* x2 = (const __nv_bfloat162*)&xv[c];
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float a = __bfloat162float(r2[p].x) + __bfloat162float(x2[p].x);
                const float b = __bfloat162float(r2[p].y) + __bfloat162float(x2[p].y);
                local += a * a + b * b;
                __nv_bfloat162 rn2;
                rn2.x = __float2bfloat16(a);
                rn2.y = __float2bfloat16(b);
                rs2[vi * 4 + p] = rn2;   // residual stream update (bf16, in place)
                fa[8 * c + 2 * p]     = a;   // norm consumes the UNROUNDED
                fa[8 * c + 2 * p + 1] = b;   // fp32 sums (matches the stock
            }                                // fused kernel's smem staging)
        }
    }
    const float rms = rsqrtf(block_reduce_sum(local, smem) / cols + eps);

    float amax = 0.f;
    #pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int vi = tid + c * nthr;
        if (vi < nvec) {
            const __nv_bfloat162* w2 = (const __nv_bfloat162*)&wv[c];
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float v0 = fa[8 * c + 2 * p] * rms * __bfloat162float(w2[p].x);
                const float v1 = fa[8 * c + 2 * p + 1] * rms * __bfloat162float(w2[p].y);
                amax = fmaxf(amax, fabsf(v0));
                amax = fmaxf(amax, fabsf(v1));
            }
        }
    }
    const float scale = fmaxf(block_reduce_max(amax, smem) / 127.f, 1e-12f);
    if (tid == 0) scales[row] = scale;
    const float inv_s = 1.f / scale;

    uint2* o2 = reinterpret_cast<uint2*>(out + (int64_t)row * cols);
    #pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int vi = tid + c * nthr;
        if (vi < nvec) {
            const __nv_bfloat162* w2 = (const __nv_bfloat162*)&wv[c];
            int q[8];
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float v0 = fa[8 * c + 2 * p] * rms * __bfloat162float(w2[p].x);
                const float v1 = fa[8 * c + 2 * p + 1] * rms * __bfloat162float(w2[p].y);
                q[2 * p]     = __float2int_rn(v0 * inv_s);
                q[2 * p + 1] = __float2int_rn(v1 * inv_s);
            }
            o2[vi] = pack_int8x8(q);
        }
    }
}

// ── single-pass dispatch wrappers ──
void rms_norm_int8_rowwise_fast(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                int8_t* out, float* scales,
                                int seq_len, int dim, float eps,
                                cudaStream_t stream) {
    if (!int8_fast_ok(dim)) {
        rms_norm_int8_rowwise(x, weight, out, scales, seq_len, dim, eps, stream);
        return;
    }
    const int nvec = dim / 8;
    int chunks = 1;
    while (nvec / chunks > 1024) chunks *= 2;
    // Full warps only (same tail-warp constraint as quantize_int8_rowwise_fast).
    const int threads = ((nvec / chunks) + 31) & ~31;
    const int smem = 32 * sizeof(float);
    switch (chunks) {
        case 1: rms_norm_int8_rowwise_fast_kernel<1><<<seq_len, threads, smem, stream>>>(
                    x, weight, out, scales, seq_len, dim, eps); break;
        case 2: rms_norm_int8_rowwise_fast_kernel<2><<<seq_len, threads, smem, stream>>>(
                    x, weight, out, scales, seq_len, dim, eps); break;
        case 4: rms_norm_int8_rowwise_fast_kernel<4><<<seq_len, threads, smem, stream>>>(
                    x, weight, out, scales, seq_len, dim, eps); break;
        case 8: rms_norm_int8_rowwise_fast_kernel<8><<<seq_len, threads, smem, stream>>>(
                    x, weight, out, scales, seq_len, dim, eps); break;
        default: rms_norm_int8_rowwise(x, weight, out, scales, seq_len, dim, eps, stream); break;
    }
}

void residual_add_rms_norm_int8_rowwise_fast(
        __nv_bfloat16* residual, const __nv_bfloat16* x,
        const __nv_bfloat16* weight, int8_t* out, float* scales,
        int seq_len, int dim, float eps, cudaStream_t stream) {
    if (!int8_fast_ok(dim)) {
        residual_add_rms_norm_int8_rowwise(
            residual, x, weight, out, scales, seq_len, dim, eps, stream);
        return;
    }
    const int nvec = dim / 8;
    int chunks = 1;
    while (nvec / chunks > 1024) chunks *= 2;
    const int threads = ((nvec / chunks) + 31) & ~31;
    const int smem = 32 * sizeof(float);
    switch (chunks) {
        case 1: residual_add_rms_norm_int8_rowwise_fast_kernel<1><<<seq_len, threads, smem, stream>>>(
                    residual, x, weight, out, scales, seq_len, dim, eps); break;
        case 2: residual_add_rms_norm_int8_rowwise_fast_kernel<2><<<seq_len, threads, smem, stream>>>(
                    residual, x, weight, out, scales, seq_len, dim, eps); break;
        case 4: residual_add_rms_norm_int8_rowwise_fast_kernel<4><<<seq_len, threads, smem, stream>>>(
                    residual, x, weight, out, scales, seq_len, dim, eps); break;
        case 8: residual_add_rms_norm_int8_rowwise_fast_kernel<8><<<seq_len, threads, smem, stream>>>(
                    residual, x, weight, out, scales, seq_len, dim, eps); break;
        default: residual_add_rms_norm_int8_rowwise(
                     residual, x, weight, out, scales, seq_len, dim, eps, stream); break;
    }
}
