# Strix Halo / ROCm 向け GPU カーネル最適化メモ

対象はこの構成に絞る。

```text
GPU/UMA: Strix Halo / AMD Radeon 8060S Graphics
メモリ: 約94 GiB usable
モデル: DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
目的: このモデルを長いコンテキストで動かす
```

## 現状ログの意味

例:

```text
ds4: ROCm preparing model tensor mappings: 80.24 GiB
ds4: ROCm q8 fp16 cache budget exhausted; using q8 kernels (request=4.00 MiB cached=0.00 GiB free=0.99 GiB reserve=4.70 GiB total=94.00 GiB)
ds4: context buffers 859.93 MiB (ctx=60000, backend=rocm, prefill_chunk=256, raw_kv_rows=512, compressed_kv_rows=15002)
```

この状態では、モデル本体はメモリに乗っている。KV cache も機能している。

問題は、Q8 重みを FP16 に展開して置く高速化用バッファが 0 GiB であること。

```text
cached=0.00 GiB
```

ただし、この構成で長い context を優先するなら、これはある程度受け入れるしかない。モデル本体が約80 GiBあり、94 GiB UMA上で長い KV/context buffer も持つため、FP16 展開済み weight cache に回す余裕はほぼない。

つまり方針は:

```text
FP16 weight cache で速くする
```

ではなく、

```text
Q8/IQ2/Q2 の直読み kernel のまま、長い context を成立させる
```

になる。

## 今回の `--kv-cache-q8` の位置づけ

`--kv-cache-q8` は decode の重み読みを減らす機能ではない。

目的は:

```text
compressed KV cache の常駐量を減らす
→ ctx を長くする
→ prefill_chunk を少しでも確保する
→ OOM を避ける
```

この目的には合っている。

単独で decode TPS が大きく上がることは期待しない。

## decode の帯域見積もり

このモデルは MoE なので、毎 token で 80 GiB 全部を読むわけではない。active expert の分だけ読む。

Flash 形状と active experts=6 から概算すると、decode 1 token あたりの主な重み読みはおおよそ:

```text
5.6〜6.0 GB / token
```

観測値:

```text
generation: 16.35 t/s
```

なら:

```text
16.35 * 5.6〜6.0 GB = 約91〜99 GB/s
```

Strix Halo の理論帯域を 256 GB/s とすると:

```text
約36〜39%
```

20 t/s なら:

```text
20 * 5.6〜6.0 GB = 約112〜120 GB/s
理論比 約44〜47%
```

この数字は、重み本体の読みだけを見た下限。実際には scale、activation、temporary、非連続アクセス、kernel launch、dequant 計算もある。

したがって decode はすでにメモリ帯域寄りで、GPU kernel を少し触っても大幅には伸びない可能性が高い。

## 優先順位

このモデルと長 context を前提にすると、優先順位は以下。

### 1. 長 context を落とさない

最優先。

```text
--ctx 60000 以上を成立させる
OOM killer を避ける
server の KV checkpoint も成立させる
```

そのため、KV cache の Q8/FP8 packing は有効。

### 2. prefill をできるだけ落とさない

長い prompt / checkpoint rebuild / server 利用では prefill が重要。

ただしメモリが厳しいため、prefill_chunk は大きくできない。

```text
ctx=60000 で prefill_chunk=256 程度
```

のような条件で、Q8/IQ2/Q2 直読み kernel がどれだけ効率よく動くかが重要。

### 3. decode は大幅改善を期待しすぎない

decode は 1 token ずつで、重み再利用が少ない。

このモデルをこのメモリ量で動かす限り、decode は:

```text
Q8/IQ2/Q2 重みを毎 token 読む
```

構造から逃げにくい。

## GPU カーネル最適化の対象

この方針で触るべきファイルは主に以下。

```text
rocm/ds4_rocm_matmul.cuh
rocm/ds4_rocm_attention_launch.cuh
rocm/ds4_rocm_norm_rope.cuh
rocm/ds4_rocm_runtime.cuh
```

特に見る対象:

```text
Q8_0 direct matmul
Q8_0 batched matmul
attention output A/B
attn_q_a / attn_q_b
shared expert Q8
output Q8
routed expert IQ2XXS / Q2_K path
```

## 変更方針

### 方針A: decode ではなく prefill 小chunk向けに寄せる

この構成では decode の大幅改善は難しい可能性が高い。

一方で prefill は token batch があるため、chunk=256 程度でも decode よりは再利用余地がある。

狙うなら:

```text
prefill_chunk=128〜512
```

の範囲で効く kernel。

大きな batch 向け GEMM だけを速くしても、この環境では使えない可能性がある。

### 方針B: FP16 weight cache 前提の高速化は捨てる

このモデル + 長 context では、FP16 weight cache 用の空きがほぼない。

したがって:

```text
cuda_q8_f16_ptr()
cuda_q8_f16_transpose_ptr()
```

に乗る前提の最適化は優先度が低い。

必要なのは:

```text
cache が 0 GiB でも速い direct path
```

### 方針C: kernel launch 数を減らす

chunk が小さいと、launch overhead や stage 分割の影響が大きい。

可能なら:

```text
norm + matmul
matmul + activation
Q8 dequant + dot
attention output の小さい projection 群
```

などをまとめる余地を見る。

ただし可読性と検証コストは上がる。

### 方針D: Q8 だけでなく IQ2XXS/Q2_K も見る

モデル名から、Q8 だけを速くしても全体改善は限定される可能性がある。

```text
IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8
```

なので、active expert 側の IQ2XXS / Q2_K 読みも decode/prefill の支配要因になる。

Q8 kernel だけを最適化する場合は、まず Q8 部分が実際にどれだけ時間を食っているか測る必要がある。

## やるなら最初に入れるべき計測

kernel を触る前に、ROCm path に以下の粒度の時間ログを入れる。

```text
layerごと
stageごと
Q8 matmulごと
expert matmulごと
attention outputごと
prefill/decode別
```

欲しい出力例:

```text
layer 12 decode attn_q_a_q8: 0.12 ms
layer 12 decode attn_q_b_q8: 0.31 ms
layer 12 decode attn_output_a_q8: 0.08 ms
layer 12 decode attn_output_b_q8: 0.22 ms
layer 12 decode routed_iq2: 1.40 ms
layer 12 decode shared_q8: 0.35 ms
```

これがないと、Q8 を触るべきか、IQ2/Q2 expert を触るべきか、attention output を触るべきか判断できない。

## 具体的な実装候補

### 1. 小chunk prefill 用 Q8 batched matmul

対象:

```text
rocm/ds4_rocm_matmul.cuh
```

狙い:

```text
n_tok = 128〜512
in_dim/out_dim は DS4 固定形状
Q8_0 weight direct read
```

やること:

```text
shape 固定の specialized kernel
weight block と scale の coalesced load
activation tile の再利用
出力 tile の連続 write
```

期待:

```text
prefill 改善の可能性あり
decode には大きく効かない
```

### 2. attention output A/B の direct Q8 path 改善

対象:

```text
rocm/ds4_rocm_attention_launch.cuh
```

このモデルでは `AProjQ8` / `OutQ8` が重い。

FP16 transpose cache が作れない前提で、Q8 direct のまま output projection を速くする。

期待:

```text
prefill と decode の両方に少し効く可能性
ただし decode は帯域上限で伸びにくい
```

### 3. routed expert IQ2XXS/Q2_K path の計測と最適化

対象:

```text
rocm/ds4_rocm_matmul.cuh
expert系kernel
```

このモデルでは active expert 6個を毎 token 読む。

decode ではここが支配的な可能性がある。

期待:

```text
decode に効く可能性はQ8単体より高いかもしれない
ただし実装難度は高い
```

### 4. Q8 FP16 cache reserve を削るのは最後

ログ上:

```text
free=0.99 GiB
reserve=4.70 GiB
```

reserve を削れば一部 cache は作れるかもしれない。

しかし長 context 目的では OOM risk が高い。

今回の主目的が:

```text
このモデルを長 context で安定動作させる
```

なので、reserve 削減は最後の手段。

## 現時点の結論

このモデル・Strix Halo・長 context 前提では、最適化方針は以下。

```text
1. KV cache Q8/FP8 packing は維持する
2. FP16 weight cache に頼らない
3. decode の大幅改善は期待しすぎない
4. prefill 小chunk向け direct Q8/IQ2/Q2 kernel を見る
5. まず stage別/kernel別の計測を入れる
```

`--kv-cache-q8` は速度チューニングではなく、長 context を成立させるためのメモリチューニングとして扱う。

GPU kernel 最適化を続けるなら、最初の作業は実装変更ではなく、ROCm decode/prefill の stage 別 profiling ログを入れること。
