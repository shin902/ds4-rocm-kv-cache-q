extern "C" int ds4_gpu_dsv4_fp8_kv_quantize_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    if (n_rot > head_dim || !cuda_tensor_has_elems2(x, n_tok, head_dim, sizeof(float))) return 0;
    if (n_tok == 0u || head_dim == 0u) return 1;
    const uint32_t n_nope = head_dim - n_rot;
    if (n_nope == 0) return 1;
    const uint32_t groups = (n_nope + 63u) / 64u;
    fp8_kv_quantize_kernel<<<dim3(n_tok, groups), 64>>>((float *)x->ptr, n_tok, head_dim, n_rot);
    return cuda_ok(cudaGetLastError(), "fp8_kv_quantize launch");
}

/* Layout helpers shared by pack/unpack below.  Must stay in lockstep with
 * dsv4_fp8_kv_pack_n_blocks()/dsv4_fp8_kv_pack_scale_off()/
 * dsv4_fp8_kv_pack_rot_off()/dsv4_fp8_kv_packed_row_bytes_cpu() in ds4.c. */
static uint32_t ds4_rocm_fp8_kv_n_blocks(uint32_t n_nope) {
    return (n_nope + 63u) / 64u;
}

static uint32_t ds4_rocm_fp8_kv_rot_off(uint32_t n_nope, uint32_t n_blocks) {
    const uint32_t off = n_nope + n_blocks;
    return off + (off & 1u);
}

extern "C" uint64_t ds4_gpu_kv_fp8_packed_row_bytes(uint32_t head_dim, uint32_t n_rot) {
    if (n_rot > head_dim) return 0;
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t n_blocks = ds4_rocm_fp8_kv_n_blocks(n_nope);
    const uint32_t rot_off = ds4_rocm_fp8_kv_rot_off(n_nope, n_blocks);
    return (uint64_t)rot_off + (uint64_t)n_rot * sizeof(uint16_t);
}

extern "C" int ds4_gpu_kv_fp8_pack_tensor(
        ds4_gpu_tensor       *packed_cache,
        uint64_t                dst_row_offset_bytes,
        const ds4_gpu_tensor *rows_f32,
        uint32_t                n_rows,
        uint32_t                head_dim,
        uint32_t                n_rot) {
    if (n_rot > head_dim || !packed_cache || !rows_f32) return 0;
    if (n_rows == 0u || head_dim == 0u) return 1;
    const uint32_t n_nope = head_dim - n_rot;
    const uint64_t row_bytes = ds4_gpu_kv_fp8_packed_row_bytes(head_dim, n_rot);
    if (row_bytes == 0 ||
        !cuda_tensor_has_elems2(rows_f32, n_rows, head_dim, sizeof(float)) ||
        dst_row_offset_bytes > packed_cache->bytes ||
        (uint64_t)n_rows * row_bytes > packed_cache->bytes - dst_row_offset_bytes) {
        return 0;
    }
    uint8_t *dst = (uint8_t *)packed_cache->ptr + dst_row_offset_bytes;
    if (n_nope != 0) {
        const uint32_t n_blocks = ds4_rocm_fp8_kv_n_blocks(n_nope);
        fp8_kv_pack_nonrope_kernel<<<dim3(n_rows, n_blocks), 64>>>(
                dst, row_bytes, 0u, n_nope,
                (const float *)rows_f32->ptr, head_dim, n_nope, n_rows);
        if (!cuda_ok(cudaGetLastError(), "fp8_kv_pack_nonrope launch")) return 0;
    }
    if (n_rot != 0) {
        const uint32_t n_blocks = ds4_rocm_fp8_kv_n_blocks(n_nope);
        const uint32_t rot_off = ds4_rocm_fp8_kv_rot_off(n_nope, n_blocks);
        const uint64_t n = (uint64_t)n_rows * n_rot;
        fp8_kv_pack_rot_kernel<<<(n + 255) / 256, 256>>>(
                dst, row_bytes, rot_off,
                (const float *)rows_f32->ptr, head_dim, n_nope, n_rot, n_rows);
        if (!cuda_ok(cudaGetLastError(), "fp8_kv_pack_rot launch")) return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_kv_fp8_unpack_tensor(
        ds4_gpu_tensor       *out_f32,
        const ds4_gpu_tensor *packed_cache,
        uint64_t                src_row_offset_bytes,
        uint32_t                n_rows,
        uint32_t                head_dim,
        uint32_t                n_rot) {
    if (n_rot > head_dim || !out_f32 || !packed_cache) return 0;
    if (n_rows == 0u || head_dim == 0u) return 1;
    const uint32_t n_nope = head_dim - n_rot;
    const uint64_t row_bytes = ds4_gpu_kv_fp8_packed_row_bytes(head_dim, n_rot);
    if (row_bytes == 0 ||
        !cuda_tensor_has_elems2(out_f32, n_rows, head_dim, sizeof(float)) ||
        src_row_offset_bytes > packed_cache->bytes ||
        (uint64_t)n_rows * row_bytes > packed_cache->bytes - src_row_offset_bytes) {
        return 0;
    }
    const uint8_t *src = (const uint8_t *)packed_cache->ptr + src_row_offset_bytes;
    if (n_nope != 0) {
        const uint32_t n_blocks = ds4_rocm_fp8_kv_n_blocks(n_nope);
        fp8_kv_unpack_nonrope_kernel<<<dim3(n_rows, n_blocks), 64>>>(
                (float *)out_f32->ptr, head_dim,
                src, row_bytes, 0u, n_nope, n_nope, n_rows);
        if (!cuda_ok(cudaGetLastError(), "fp8_kv_unpack_nonrope launch")) return 0;
    }
    if (n_rot != 0) {
        const uint32_t n_blocks = ds4_rocm_fp8_kv_n_blocks(n_nope);
        const uint32_t rot_off = ds4_rocm_fp8_kv_rot_off(n_nope, n_blocks);
        const uint64_t n = (uint64_t)n_rows * n_rot;
        fp8_kv_unpack_rot_kernel<<<(n + 255) / 256, 256>>>(
                (float *)out_f32->ptr, head_dim, n_nope,
                src, row_bytes, rot_off, n_rot, n_rows);
        if (!cuda_ok(cudaGetLastError(), "fp8_kv_unpack_rot launch")) return 0;
    }
    return 1;
}
