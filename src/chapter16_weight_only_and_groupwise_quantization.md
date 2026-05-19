# Chapter 16: Weight-Only and Group-Wise Quantization

In this chapter, we quantize model weights while keeping activations in higher precision.

## The Memory Problem Returns

Inference is memory-bandwidth-bound: the bottleneck is loading parameters from memory, not computing with them. SmoothQuant addresses activation quantization â€” making activations survive int8 by redistributing outliers. But for the largest language models, a different approach has become dominant: quantize only the weights and leave activations in higher precision entirely.
(Weight-only means only weights are quantized; activations remain float during compute.)

The reasoning is economic. A 70-billion-parameter model in float16 occupies 140 GB. At a memory bandwidth of 900 GB/s, loading the full model for one token takes ~156 milliseconds. At int4, the model is 35 GB â€” loading takes ~39 milliseconds, a 4Ã— improvement.

Activations, by contrast, are small during decode. For a single inference with batch size 1, the activation tensor at each layer is shaped [1, hidden_dim] â€” typically a few kilobytes. Even in float16, activations contribute negligibly to the memory traffic during decode. Quantizing them saves almost nothing in bandwidth. In prefill or large-batch scenarios, activation traffic and compute can dominate â€” but for the decode bottleneck, the entire bandwidth argument from Chapter 1 applies to weights, not activations.

---

## Compute-Bound vs. Memory-Bound: When Weight-Only Applies

Not all workloads benefit from weight-only quantization. The benefit depends on whether inference is *memory-bound* or *compute-bound*.

**Memory-bound workloads** (LLM token generation, batch size 1): The GPU spends most of its time loading weights from HBM. The compute units are idle, waiting for data. Weight-only quantization cuts the data to load by 2â€“4Ã—, directly improving tokens-per-second. This is the dominant LLM serving scenario.

**Compute-bound workloads** (CNNs, large-batch LLM prefill, image models): The compute units are fully utilized. The bottleneck is arithmetic, not data loading. Weight-only quantization still reduces memory footprint (useful for fitting the model on fewer GPUs), but it does *not* improve throughput â€” because the dequantized float16 matmul runs at the same speed as a native float16 matmul. In compute-bound regimes, weight-only typically improves footprint more than throughput, and dequant overhead can even reduce performance if it becomes the limiter.

| Workload | Bottleneck | Weight-Only Benefit | Full int8 Benefit |
|---|---|---|---|
| LLM decode (batch=1) | Memory bandwidth | High (2â€“4Ã— throughput) | Limited (activations are tiny) |
| LLM prefill (batch=32) | Compute | Low (only memory savings) | Often high (up to 2Ã— compute throughput, hardware-dependent) |
| CNN inference | Compute | Low | High |
| Batch embedding | Compute | Low | High |

This distinction explains why weight-only quantization dominates in the LLM serving world but is rarely used for CNNs. A ResNet-50 inference at batch 32 is compute-bound â€” full int8 quantization doubles throughput; weight-only quantization changes nothing.

---

## Weight-Only Quantization

In weight-only quantization, weights are stored in a low-precision format â€” typically int4 or int8. At inference time, they are dequantized to float16 on-the-fly, just before the matrix multiplication. The matmul itself runs in float16. Activations are never quantized.

The data flow for a single linear layer:

1. Load int4 weights from memory (low bandwidth cost)
2. Dequantize to float16 on-the-fly (often fused into the GEMM kernel)
3. Multiply float16 activations Ã— float16 weights (standard matmul)
4. Output is float16 (no requantization)

The only quantization error is in the weight representation. And because weights are static â€” they do not change between inputs â€” the error is fixed and predictable. There is no per-input variation, no calibration drift, no dynamic range problem. The weight values are what they are, and the quantization error is baked in at model load time.

Weight-only quantization typically uses symmetric int4 (zero-point = 0) with per-group scales. This simplifies metadata (scales only, no zero-points) and avoids the asymmetric-scaling overhead.

The matmul itself is exact (within float16 precision). There is no requantization boundary (Chapter 7) at the output, because the output is float16 â€” no domain conversion is needed. The entire requantization error budget from Chapters 6â€“8 is irrelevant for weight-only quantization.

---

## The Resolution Problem at 4-Bit

Int4 has 16 levels. Representing a weight distribution that spans, say, [-0.5, 0.5] with 16 levels gives a step size of:

$$S = \frac{1.0}{15} \approx 0.067$$

Two weight values that differ by less than 0.067 become the same integer. For a model where weight precision matters â€” and in large language models, it matters enormously â€” this is coarse.

*Canonical category: Resolution Collapse â€” the 4-bit grid is too coarse to preserve weight distinctions that the model relies on.*

Per-tensor quantization (one scale for the entire weight matrix) makes this worse. If one region of the weight matrix contains values in [-0.01, 0.01] and another contains values in [-0.5, 0.5], the per-tensor scale is set by the wider range. The small-value region gets perhaps one grid point. The weight information in that region is effectively destroyed.

Per-channel quantization (one scale per output row) helps â€” each row gets its own range. But within a row, variance can still be high. The solution is to go finer.

---

## Group-Wise Quantization

Group-wise quantization assigns a separate scale (and optionally zero-point) to each *group* of consecutive weights along a dimension.
(Group-wise means splitting weights into small groups and giving each group its own scale.)
Instead of one scale per tensor or one scale per channel, there is one scale per group of \\(g\\) weights.

Consider a weight matrix of shape [4096, 4096] â€” 16.7 million weights.

**Per-tensor (group size = all):** 1 scale. The entire matrix shares one range. If any region has extreme values, all regions pay the resolution cost.

**Per-channel (group size = 4096, one per row):** 4,096 scales. Each row gets its own range. Variance between rows is handled, but variance within a row is not.

**Group-wise (group size = 128):** Each row is divided into groups of 128 weights. Each group gets its own scale. Total scales: \\(4096 \times (4096 / 128) = 131{,}072\\).

**Detailed example.** Take row 1 of a [4096 Ã— 4096] weight matrix, divided into 32 groups of 128 weights each:

- Group 1 (columns 0â€“127): weights in [-0.12, 0.15], range = 0.27. Scale \\(= 0.27 / 15 = 0.018\\). A weight of 0.05 maps to code \\(\text{round}((0.05 - (-0.12)) / 0.018) = \text{round}(9.44) = 9\\), dequantized to \\(-0.12 + 9 \times 0.018 = 0.042\\). Error: 0.008.
- Group 17 (columns 2176â€“2303): weights in [-0.48, 0.47], range = 0.95. Scale \\(= 0.95 / 15 = 0.063\\). The same weight 0.05 maps to code \\(\text{round}((0.05 - (-0.48)) / 0.063) = \text{round}(8.41) = 8\\), dequantized to \\(-0.48 + 8 \times 0.063 = 0.024\\). Error: 0.026 â€” 3.3Ã— worse.

Group 1â€™s narrow range gives it 3.3Ã— better precision than Group 17. Under per-tensor quantization, both groups would share the matrix-wide range (say [-0.52, 0.50], scale \\(= 1.02/15 = 0.068\\)), giving Group 1 only \\(0.27/0.068 \approx 4\\) usable levels instead of 15. Group-wise quantization preserves each groupâ€™s local precision.

For a group of 128 weights that all fall in [-0.02, 0.02]:

$$S = \frac{0.04}{15} \approx 0.00267$$

Compared to per-tensor with the full [-0.5, 0.5] range where \\(S = 0.067\\), this group's resolution is 25Ã— finer. The weights in this group are represented with precision matched to their actual range.

*Canonical category: group-wise quantization directly addresses Resolution Collapse and Distribution Mismatch / Budget Waste by matching each group's scale to its local range, rather than wasting codes on a globally-set range.*

### The Overhead

Each scale is stored alongside the weights. With 131,072 scales at 2 bytes each (float16), the metadata overhead is:

$$131{,}072 \times 2 = 262 \text{ KB}$$

The weight matrix itself at int4 occupies:

$$16.7 \times 10^6 \times 0.5 \text{ bytes} = 8.35 \text{ MB}$$

The metadata is ~3% of the weight data â€” a modest overhead for a significant accuracy improvement (scale-only metadata; zero-points or extra packing metadata increase this slightly).

### Group Size as a Knob

Smaller groups mean more scales, better accuracy, and more metadata overhead. The trade-off:

| Group Size | Scales per Row | Accuracy | Metadata Overhead |
|---|---|---|---|
| 4096 (per-channel) | 1 | Lowest | Negligible |
| 128 | 32 | Good | ~3% |
| 32 | 128 | Better | ~12% |
| 1 (per-element) | 4096 | Best (trivially) | 100%+ (defeats the purpose) |

Group sizes of 128 and 32 are the most common in practice â€” they provide substantial accuracy improvement over per-channel with manageable overhead. Group size 128 is the dominant choice for int4 LLM quantization today.

---

## The Boundary Rule Does Not Apply Here

A reader who has internalized Chapters 6â€“8 â€” where quantization boundaries, requantization error, and fusion dominate the discussion â€” might expect these concerns to carry over. They do not.

To understand why, recall the full integer pipeline from those chapters: int8 Ã— int8 â†’ int32 accumulator â†’ requantize to int8 â†’ next layer. Every arrow in that chain is a source of rounding error. In weight-only quantization, activations stay in float16 throughout â€” there is no int32 accumulator, no int8 output, and therefore no requantization step at all.

In weight-only quantization, the boundary framework from Chapters 6â€“8 is *bypassed entirely*:

- **No integer accumulator stage.** The matmul runs in float16. Accumulation happens in floating-point (fp16, or fp32 internally on some GPUs). There is no int32 accumulator and therefore no int32â†’int8 requantization step.
- **No requantization.** The output is float16. No conversion from int32â†’int8 is needed. The requantization error budget from Chapter 7 does not apply.
- **No fusion for error reduction.** Fusion in weight-only models reduces kernel launch overhead and memory traffic, but it does not eliminate requantization error â€” because there is no requantization error to eliminate.

The only error source is weight quantization itself: the difference between the float16 weight and its int4 representation. This error is static (input-independent) and there is no activation calibration step. Weight-only quantization avoids the repeated requantization noise introduced by int32â†’int8 boundaries â€” the Cumulative Rounding Noise pattern from Chapter 13 does not arise in its boundary-accumulation form.

*Canonical category: weight-only quantization bypasses Cumulative Rounding Noise (boundary form) and Calibration Mismatch (no activation calibration). The remaining risk is Resolution Collapse in the weights themselves, controlled by group size.*

---

## The KV-Cache: The Other Memory Bottleneck

Weight-only quantization solves the weight loading bottleneck. But for long-context LLM inference, a second memory bottleneck emerges: the *key-value cache*.

During autoregressive generation, each transformer layer stores the key and value projections of all previously generated tokens. For a 70B model with 80 layers, 8 KV heads, and head dimension 128, the KV-cache per token occupies:

$$80 \times 2 \times 8 \times 128 \times 2 \text{ bytes (float16)} = 327{,}680 \text{ bytes} \approx 320 \text{ KB per token}$$

At 8,192 tokens of context: \\(320 \text{ KB} \times 8{,}192 = 2.56 \text{ GB}\\).
At 128,000 tokens: \\(320 \text{ KB} \times 128{,}000 = 40 \text{ GB}\\).

The KV-cache grows linearly with sequence length and must be loaded from memory at every generation step. For long sequences, it rivals or exceeds the model weights in memory bandwidth cost.

KV-cache quantization reduces this cost. Storing keys and values in int8 instead of float16 cuts KV-cache memory by 50%. FP8 (Chapter 19) achieves similar savings with less precision loss due to its non-uniform grid.

But KV-cache quantization is fundamentally different from weight quantization:

- **Weights are static.** They are quantized once at load time. KV-cache entries are *generated dynamically* â€” each new token adds new entries that must be quantized on-the-fly during inference.
- **Weights can be calibrated offline.** KV-cache distributions depend on the actual input sequence and cannot be predicted in advance. Scales must be computed per-head, per-layer, and updated as the sequence grows.
- **Weight quantization error is fixed.** KV-cache quantization error varies with the content being generated â€” different sequences produce different KV distributions and different quantization error patterns.

*KV-cache quantization reintroduces Calibration Mismatch risk (scales derived online from limited context) and Tail Clipping risk (saturation if scales are not updated as the sequence grows). This connects KV-cache quantization back to the dynamic quantization framework from Chapter 12.*

KV-cache quantization is explored in depth in Chapter 18.

---

## Conceptual Consolidation

Weight-only quantization targets the binding constraint for LLM inference: memory bandwidth for loading weights. Activations stay in float16 because they are small and quantizing them saves negligible bandwidth. Group-wise quantization provides fine-grained precision control by assigning separate scales to groups of weights, with group size as the knob that trades accuracy for metadata overhead.

The question for weight-only quantization is not "what precision?" but "what group size?" The answer depends on the model's sensitivity and the acceptable metadata overhead.

