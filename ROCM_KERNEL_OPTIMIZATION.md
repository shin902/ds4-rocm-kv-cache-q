# ROCm GPU kernel optimization notes

This note summarizes the current Strix Halo / ROCm performance picture for DS4, especially the difference between KV-cache packing and model-weight Q8 kernels.

## Terminology

There are two unrelated "caches" in the logs:

### KV cache

Transformer runtime state:

- Stores previous-token key/value rows.
- Avoids recomputing past tokens during decode.
- `--kv-cache-fp8` and `--kv-cache-q8` reduce the resident size of the **compressed KV cache**.
- This helps fit larger contexts / larger prefill chunks, but it does not reduce the model weights read per decoded token.

### Q8 FP16 weight cache

ROCm weight acceleration path:

- Some Q8 model tensors can be dequantized once into FP16 buffers.
- Later matmuls can use hipBLAS/hipBLASLt-style FP16 paths.
- If this cache cannot be allocated, DS4 falls back to direct Q8 kernels:

```text
ds4: ROCm q8 fp16 cache budget exhausted; using q8 kernels
```

This is not the KV cache. It means the model-weight fast path has no extra memory budget.

## Current observed state

Example log:

```text
ds4: ROCm preparing model tensor mappings: 80.24 GiB
ds4: ROCm q8 fp16 cache budget exhausted; using q8 kernels (request=4.00 MiB cached=0.00 GiB free=0.99 GiB reserve=4.70 GiB total=94.00 GiB)
ds4: context buffers 23.90 MiB (ctx=163, backend=rocm, prefill_chunk=163, raw_kv_rows=256, compressed_kv_rows=42)
ds4: prefill: 22.38 t/s, generation: 16.35 t/s
```

Interpretation:

- The model fits in memory.
- The KV cache exists and functions normally.
- The Q8 FP16 weight cache has **0 GiB** allocated.
- Even the first 4 MiB cache request is rejected because free memory is below the safety reserve:

```text
cache budget = free - reserve = 0.99 GiB - 4.70 GiB < 0
```

So inference uses the direct Q8 weight kernels.

## Decode bandwidth estimate

For DeepSeek V4 Flash shape:

- Layers: 43
- Embedding: 4096
- Active experts: 6
- FF expert dim: 2048
- Main Q8 tensors include attention projections, shared experts, and output head.
- Routed experts are active-subset only, not all experts every token.

Approximate active model-weight traffic per decoded token:

```text
IQ2XXS active routed experts: ~5.6 GB/token
Q2_K active routed experts:  ~6.0 GB/token
```

At observed decode speed:

```text
16.35 tok/s * 5.6..6.0 GB/token = ~91..99 GB/s
```

Against a nominal 256 GB/s memory bandwidth:

```text
~36..39% of peak
```

At 20 tok/s:

```text
20 tok/s * 5.6..6.0 GB/token = ~112..120 GB/s
=> ~44..47% of 256 GB/s
```

This is a lower-bound estimate based mainly on model-weight reads. Real traffic also includes scales, activations, intermediate writes, non-coalesced access, and dispatch overhead.

## Why decode may not improve much

Decode is mostly one token at a time:

- Low arithmetic reuse per weight read.
- Large amount of model weight traffic per token.
- Direct Q8 kernels must load quantized bytes and scales, then dequantize in-kernel.
- If memory bandwidth is the bottleneck, kernel tuning can improve efficiency but may not change the fundamental TPS ceiling dramatically.

In contrast, prefill can process multiple tokens at once. Larger prefill chunks improve reuse and make GEMM-like kernels more efficient, but they require more memory.

## Current Q8 direct path

The direct path is already active when the log says:

```text
using q8 kernels
```

Relevant files:

- `rocm/ds4_rocm_matmul.cuh`
- `rocm/ds4_rocm_attention_launch.cuh`
- `rocm/ds4_rocm_norm_rope.cuh`
- `rocm/ds4_rocm_runtime.cuh`

The FP16 weight-cache selection is around:

- `cuda_q8_f16_ptr(...)`
- `cuda_q8_f16_transpose_ptr(...)`
- `cuda_q8_f16_cache_has_budget(...)`

## Optimization directions

### 1. Make direct Q8 kernels more bandwidth-efficient

Possible kernel-level work:

- Better tiling for small-batch decode.
- Coalesced/vectorized loads for Q8 blocks and scales.
- Reduce redundant scale loads.
- Fuse dequantization with dot accumulation more tightly.
- Use LDS/shared memory where reuse exists.
- Specialize hot shapes instead of one generic path.
- Reduce launch count for tiny decode operations.

Expected impact:

- Potentially useful, but bounded if decode is already bandwidth-limited.

### 2. Improve prefill kernels

Prefill has more room for GPU occupancy and reuse:

- Token batch dimension > 1.
- Weight reuse across tokens.
- More GEMM-like workload.

Possible work:

- Q8 direct batched matmul tiling.
- Better chunk-size-specific kernels.
- Fuse adjacent projection/norm/activation steps where safe.

Expected impact:

- More promising than decode if memory allows chunk sizes above very small values.

### 3. Free memory to enable some Q8 FP16 weight cache

This is not kernel tuning, but may have a large effect:

- Reduce context size.
- Use `--kv-cache-q8` / `--kv-cache-fp8` to reduce KV pressure.
- Use SSD streaming to lower resident model pressure, then allocate selected hot FP16 caches.
- Lower the reserve only if OOM risk is acceptable.

Expected impact:

- If even a subset of hot Q8 tensors can be cached as FP16, selected matmuls may switch to faster library paths.

### 4. Reduce decoded-token weight traffic

If decode is memory-bandwidth bound, the strongest levers reduce bytes/token:

- More aggressive weight quantization for hot tensors.
- Smaller model / fewer active experts.
- Speculative decoding / MTP to amortize decode overhead.
- Multi-request batching, if serving multiple requests.

## Practical conclusion

`--kv-cache-q8` is primarily a memory-capacity feature. It can help fit larger contexts or larger prefill chunks, but it should not be expected to greatly improve single-stream decode TPS.

For decode, current performance is likely constrained by direct Q8 model-weight reads and available memory bandwidth. Kernel work can improve utilization, but large gains may require reducing bytes/token or enabling some FP16 weight cache.
