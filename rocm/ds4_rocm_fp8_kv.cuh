// DS4 ROCm FP8 KV quantization and raw-cache store kernels.
//
// This file is included from ds4_cuda.cu (same translation unit) so these
// kernels can reuse the backend's existing device helpers without HIP device
// linking or behavior changes.

__global__ static void fp8_kv_quantize_kernel(float *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    const uint32_t row = blockIdx.x;
    const uint32_t grp = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t off = grp * 64u;
    if (row >= n_tok || off >= n_nope) return;
    float *xr = x + (uint64_t)row * head_dim;
    __shared__ float scratch[64];
    float v = 0.0f;
    if (tid < 64u && off + tid < n_nope) v = xr[off + tid];
    scratch[tid] = (tid < 64u && off + tid < n_nope) ? fabsf(v) : 0.0f;
    __syncthreads();
    for (uint32_t stride = 32; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        __syncthreads();
    }
    const float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
    if (tid < 64u && off + tid < n_nope) {
        const float q = dsv4_e4m3fn_dequant_dev(fminf(448.0f, fmaxf(-448.0f, v / scale))) * scale;
        xr[off + tid] = q;
    }
}

/* =========================================================================
 * ROCm packed FP8 compressed-KV cache (opt-in, --kv-cache-fp8).
 * =========================================================================
 *
 * Packs the same E4M3 amax/2^e-scale/round-to-even values that
 * fp8_kv_quantize_kernel() above computes into a compact byte layout: one
 * E4M3 sign+magnitude byte per non-RoPE element, one scale-exponent byte per
 * 64-wide block, and the RoPE tail kept at F16.  This mirrors (and must stay
 * bit-for-bit consistent with) dsv4_fp8_kv_pack_row_cpu()/
 * dsv4_fp8_kv_unpack_row_cpu() in ds4.c, which is the CPU reference used for
 * checkpoint round trips and the standalone unit test.
 */

__device__ static uint8_t dsv4_e4m3fn_encode_dev(float x) {
    const uint8_t sign_bit = x < 0.0f ? 0x80u : 0x00u;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value_dev(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        float bd = fabsf(ax - dsv4_e4m3fn_value_dev(best));
        float nd = fabsf(ax - dsv4_e4m3fn_value_dev(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) best++;
    }
    return (uint8_t)(sign_bit | (unsigned)best);
}

__device__ static float dsv4_e4m3fn_decode_code_dev(uint8_t code) {
    const float sign = (code & 0x80u) ? -1.0f : 1.0f;
    return sign * dsv4_e4m3fn_value_dev(code & 0x7f);
}

/* Packs one 64-wide non-RoPE block per (row, block) launch cell: computes
 * the block amax/scale exactly like fp8_kv_quantize_kernel() and writes the
 * E4M3 codes plus the scale-exponent byte into the packed row. */
__global__ static void fp8_kv_pack_nonrope_kernel(
        uint8_t *packed, uint64_t row_bytes, uint32_t codes_off, uint32_t scale_off,
        const float *rows_f32, uint32_t head_dim, uint32_t n_nope, uint32_t n_rows) {
    const uint32_t row = blockIdx.x;
    const uint32_t grp = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t off = grp * 64u;
    if (row >= n_rows || off >= n_nope) return;
    const float *xr = rows_f32 + (uint64_t)row * head_dim;
    uint8_t *out = packed + (uint64_t)row * row_bytes;
    __shared__ float scratch[64];
    float v = 0.0f;
    if (tid < 64u && off + tid < n_nope) v = xr[off + tid];
    scratch[tid] = (tid < 64u && off + tid < n_nope) ? fabsf(v) : 0.0f;
    __syncthreads();
    for (uint32_t stride = 32; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        __syncthreads();
    }
    const int e = (int)ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f));
    const float scale = exp2f((float)e);
    if (tid == 0) out[scale_off + grp] = (uint8_t)(int8_t)e;
    if (tid < 64u && off + tid < n_nope) {
        out[codes_off + off + tid] = dsv4_e4m3fn_encode_dev(fminf(448.0f, fmaxf(-448.0f, v / scale)));
    }
}

__global__ static void fp8_kv_pack_rot_kernel(
        uint8_t *packed, uint64_t row_bytes, uint32_t rot_off,
        const float *rows_f32, uint32_t head_dim, uint32_t n_nope, uint32_t n_rot, uint32_t n_rows) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_rows * n_rot;
    if (gid >= n) return;
    const uint32_t row = gid / n_rot;
    const uint32_t i = gid - (uint64_t)row * n_rot;
    const float v = rows_f32[(uint64_t)row * head_dim + n_nope + i];
    uint16_t *rot = (uint16_t *)(void *)(packed + (uint64_t)row * row_bytes + rot_off);
    rot[i] = f32_to_f16_bits_hip_round(v);
}

__global__ static void fp8_kv_unpack_nonrope_kernel(
        float *rows_f32, uint32_t head_dim,
        const uint8_t *packed, uint64_t row_bytes, uint32_t codes_off, uint32_t scale_off,
        uint32_t n_nope, uint32_t n_rows) {
    const uint32_t row = blockIdx.x;
    const uint32_t grp = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t off = grp * 64u;
    if (row >= n_rows || off >= n_nope) return;
    const uint8_t *in = packed + (uint64_t)row * row_bytes;
    if (tid < 64u && off + tid < n_nope) {
        const float scale = exp2f((float)(int8_t)in[scale_off + grp]);
        rows_f32[(uint64_t)row * head_dim + off + tid] =
                dsv4_e4m3fn_decode_code_dev(in[codes_off + off + tid]) * scale;
    }
}

__global__ static void fp8_kv_unpack_rot_kernel(
        float *rows_f32, uint32_t head_dim, uint32_t n_nope,
        const uint8_t *packed, uint64_t row_bytes, uint32_t rot_off,
        uint32_t n_rot, uint32_t n_rows) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_rows * n_rot;
    if (gid >= n) return;
    const uint32_t row = gid / n_rot;
    const uint32_t i = gid - (uint64_t)row * n_rot;
    const uint16_t *rot = (const uint16_t *)(const void *)(packed + (uint64_t)row * row_bytes + rot_off);
    rows_f32[(uint64_t)row * head_dim + n_nope + i] = f16_bits_to_f32(rot[i]);
}

/* =========================================================================
 * ROCm packed Q8 compressed-KV cache (opt-in, --kv-cache-q8).
 * =========================================================================
 *
 * Conventional symmetric int8 storage for the non-RoPE prefix: one signed Q8
 * byte per value, one F32 amax/127 scale per 64-wide block, and the RoPE tail
 * kept at F16.  This mirrors dsv4_q8_kv_pack_row_cpu()/unpack_row_cpu().
 */

__device__ static int8_t q8_kv_quantize_code_dev(float v, float scale) {
    if (scale <= 0.0f) return 0;
    float qf = v / scale;
    qf = fminf(127.0f, fmaxf(-127.0f, qf));
    int q = qf >= 0.0f ? (int)floorf(qf + 0.5f) : (int)ceilf(qf - 0.5f);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    return (int8_t)q;
}

__global__ static void q8_kv_pack_nonrope_kernel(
        uint8_t *packed, uint64_t row_bytes, uint32_t codes_off, uint32_t scale_off,
        const float *rows_f32, uint32_t head_dim, uint32_t n_nope, uint32_t n_rows) {
    const uint32_t row = blockIdx.x;
    const uint32_t grp = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t off = grp * 64u;
    if (row >= n_rows || off >= n_nope) return;
    const float *xr = rows_f32 + (uint64_t)row * head_dim;
    uint8_t *out = packed + (uint64_t)row * row_bytes;
    __shared__ float scratch[64];
    float v = 0.0f;
    if (tid < 64u && off + tid < n_nope) v = xr[off + tid];
    scratch[tid] = (tid < 64u && off + tid < n_nope) ? fabsf(v) : 0.0f;
    __syncthreads();
    for (uint32_t stride = 32; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        __syncthreads();
    }
    const float scale = scratch[0] > 0.0f ? scratch[0] / 127.0f : 1.0e-8f;
    if (tid == 0) ((float *)(void *)(out + scale_off))[grp] = scale;
    if (tid < 64u && off + tid < n_nope) {
        ((int8_t *)(void *)(out + codes_off))[off + tid] = q8_kv_quantize_code_dev(v, scale);
    }
}

__global__ static void q8_kv_unpack_nonrope_kernel(
        float *rows_f32, uint32_t head_dim,
        const uint8_t *packed, uint64_t row_bytes, uint32_t codes_off, uint32_t scale_off,
        uint32_t n_nope, uint32_t n_rows) {
    const uint32_t row = blockIdx.x;
    const uint32_t grp = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t off = grp * 64u;
    if (row >= n_rows || off >= n_nope) return;
    const uint8_t *in = packed + (uint64_t)row * row_bytes;
    if (tid < 64u && off + tid < n_nope) {
        const float scale = ((const float *)(const void *)(in + scale_off))[grp];
        const int8_t code = ((const int8_t *)(const void *)(in + codes_off))[off + tid];
        rows_f32[(uint64_t)row * head_dim + off + tid] = (float)code * scale;
    }
}

__global__ static void store_raw_kv_batch_kernel(float *raw, const float *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t t = gid / head_dim;
    uint32_t row = (pos0 + t) % raw_cap;
    const uint16_t hb = f32_to_f16_bits_hip_round(kv[(uint64_t)t * head_dim + d]);
    raw[(uint64_t)row * head_dim + d] = f16_bits_to_f32(hb);
}
