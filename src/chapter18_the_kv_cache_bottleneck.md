# Chapter 18: The KV-Cache Bottleneck

So far, we have seen: quantization = mapping values to lower precision with controlled error.
So far, we quantized static data (weights). KV-cache is different: it is generated during inference.
Naive quantization fails here because cache memory grows with sequence length and degrades long-context quality.
This section shows how we fix that with KV-aware quantization choices.

In this chapter, we quantize KV-cache tensors generated at runtime.

## The Second Memory Wall

Chapter 1 established that inference is memory-bandwidth-bound — the cost of loading model weights dominates latency. Chapter 16 addressed this by compressing weights from 140 GB to 35 GB. That was the first memory wall.

For long-context language model inference, a second memory wall appears — one that weight quantization does not touch. And quantizing it requires the same core machinery from this book — scales, observers, error trade-offs — but applied under a new constraint: the data being quantized is *dynamic*, generated during inference, and grows with every token.

During autoregressive generation, each transformer layer caches the key and value projections of every previously generated token. This *KV-cache* must be read from memory at every generation step, because the attention mechanism attends over all prior tokens to produce the next one. As the sequence grows, the KV-cache grows — and at some point, loading the KV-cache costs more than loading the model weights.

---

## The Arithmetic of KV-Cache Growth

Before the formula, a brief architecture note: a transformer has \\(L\\) *layers* (large models typically have 80–100), each with an *attention mechanism* that has \\(H\\) *KV-heads* (fewer in models using grouped-query attention), each operating on vectors of dimension \\(d\\) (typically 128). At each generation step, every head in every layer stores one key vector and one value vector for the current token. These accumulate as the sequence grows.

For a transformer with \\(L\\) layers, \\(H\\) KV-heads, and head dimension \\(d\\), the KV-cache stores two tensors (key and value) per layer per head. In float16, the KV-cache size for a sequence of \\(T\\) tokens is:

To see exactly how fast this memory grows, we quantify it:

$$\text{KV size} = L \times 2 \times H \times d \times T \times 2 \text{ bytes}$$

> **📊 INSERT DIAGRAM: KV-Cache Growth vs. Model Weights**
>
> A line chart with sequence length (T) on the x-axis (0 to 128K tokens) and memory in GB on the y-axis:
>
> ```
> Memory
> (GB)
>  40 │                                                ╱╱ KV-cache (float16)
>     │                                           ╱╱╱╱
>  35 │─────────────────────────── Model weights (int4) = 35 GB (constant)
>     │                                      ╱╱╱
>  30 │                                 ╱╱╱╱
>     │                            ╱╱╱╱
>  20 │                       ╱╱╱╱         KV-cache (int8) ← half the slope
>     │                  ╱╱╱╱
>  10 │             ╱╱╱╱           KV-cache (int4) ← quarter the slope
>     │        ╱╱╱╱
>   0 │────────────────────────────────────────────────
>     0     4K     16K     32K     64K     128K  tokens
> ```
>
> Annotate the crossover point where KV-cache (float16) exceeds model weight memory.
> Label: "At ~110K tokens, the KV-cache alone exceeds the 35 GB model weights. This is why KV-cache quantization matters for long-context models."
> Show int8 and int4 KV-cache lines to demonstrate how quantization pushes the crossover point further out.

For a concrete model — 80 layers, 8 KV-heads (with grouped-query attention), head dimension 128:

| Sequence Length | KV-Cache Size (float16) | Model Weights (int4) |
|---|---|---|
| 1,024 tokens | 320 MiB | 35 GB |
| 8,192 tokens | 2.5 GiB | 35 GB |
| 32,768 tokens | 10 GiB | 35 GB |
| 131,072 tokens | 40 GiB | 35 GB |

At 131K tokens, the KV-cache exceeds the model itself. At each decode step, the runtime must read a substantial fraction of the KV-cache and stream a substantial fraction of the weights, so KV traffic eventually rivals or exceeds weight traffic for long contexts. Weight-only quantization halved the first cost but left the second untouched.

For models that support 1M+ token contexts — a direction the industry is moving rapidly — the KV-cache at float16 would require ~300+ GiB of memory. This exceeds the capacity of any single GPU.

---

## Why KV-Cache Quantization Is Different

Weight quantization (Chapters 16–17) operates on static values: the weights are fixed after training, their distributions are known, and quantization parameters can be chosen with unlimited time and calibration data.

KV-cache quantization operates on *dynamic values* that are generated during inference:

**1. Values are created token by token.** Each new token adds a new key and value vector to the cache. These values must be quantized immediately — there is no opportunity for offline calibration.

**2. Distributions shift with content.** A KV-cache for a code generation task has different value distributions than one for creative writing. The same model produces different KV distributions for different inputs. Scales computed from one sequence may be wrong for another.

**3. Scales cannot be precomputed.** Weight scales are baked into the model at load time. KV-cache scales must be computed on-the-fly — per head, per layer, and potentially updated as the sequence grows and the distribution shifts.

**4. The cache grows monotonically.** Weights are a fixed-size tensor. The KV-cache grows with every token. A scale computed from the first 100 tokens may be wrong after 10,000 tokens if the value distribution widens.

---

## Approaches to KV-Cache Quantization

Naive cache quantization fails because distribution drift over tokens makes one fixed scale unreliable.
This section shows practical strategies and their trade-offs.

### Per-Token Quantization

The simplest approach: quantize each new KV entry independently using per-token statistics. Implementations often compute a lightweight statistic (e.g., absmax) per token per head rather than full min/max over the entire vector, to keep overhead low. Each token's key and value vectors get their own scale.

Overhead: one scale per token per head per layer. For the 80-layer, 8-head model at 32K tokens: \\(80 \times 2 \times 8 \times 32{,}768 = 41{,}943{,}040\\) scales. At 2 bytes each, this is ~80 MiB — modest compared to the KV-cache itself.

The problem: per-token quantization captures the range of each individual token but does not capture inter-token relationships. Two tokens with similar scales but different distributions get similar quantization treatment, even if one carries more information.

**Worked example: per-token vs. per-head scaling.** Layer with 128-dim KV heads.

- Token 1: values in [-0.5, 0.8]. Per-token scale \\(= 1.3 / 255 \approx 0.0051\\). All 128 values get the full 256 int8 codes.
- Token 100: values in [-2.1, 3.2]. Per-token scale \\(= 5.3 / 255 \approx 0.0208\\). Same 256 codes, but each step is 4× wider.

Under per-head shared scale (covering both tokens): range must span [-2.1, 3.2], so scale \\(= 5.3 / 255 = 0.0208\\). Token 1’s values in [-0.5, 0.8] now get only \\(1.3 / 0.0208 \approx 63\\) usable levels instead of 256 — a 4× loss in resolution. Token 1’s key vector, which encodes subtle distinctions the attention mechanism relies on, is crushed to 63 levels. Per-token quantization avoids this by giving each token its own optimal scale.

*Canonical category: mitigates Calibration Mismatch (per-input scaling). May still suffer Distribution Mismatch within the vector if the distribution is highly peaked.*

### Per-Head Quantization

A coarser approach: compute one scale per head per layer, shared across all tokens. This reduces metadata dramatically but forces all tokens in a head to share one range. If early tokens have small activations and late tokens have large activations (common in long-context generation), the shared scale is set by the maximum — causing representation error for the majority, exactly the outlier explosion from Chapter 13.

*Canonical category: Resolution Collapse for the bulk of tokens when a few tokens force a wide scale. Distribution Mismatch / Budget Waste — most codes are allocated to an empty range.*

### Sliding-Window Scale Updates

A compromise: compute the scale from the most recent \\(W\\) tokens and apply it to the entire cache. As new tokens arrive, the scale is recomputed and the existing cache is (optionally) requantized to the updated scale.

This introduces a new problem: requantizing old cache entries under a new scale introduces error at every update. If the distribution is stable, updates are rare and error is low. If the distribution drifts, frequent updates trade requantization error for range accuracy.

*Canonical categories: frequent scale updates introduce Cumulative Rounding Noise (requantization events on historical entries). Too-infrequent updates risk Tail Clipping (distribution widens beyond scale) or Distribution Mismatch / Budget Waste (scale widened preemptively, wasting codes).*

### Asymmetric Key vs. Value Treatment

Recent research shows that keys and values have different quantization sensitivities:

- **Keys** participate in attention score computation (\\(QK^T\\)). Errors in keys shift attention weights, potentially causing the model to attend to the wrong tokens. Key quantization is high-sensitivity.
- **Values** are weighted-summed by the attention weights. Errors in values are smoothed by the averaging — an error in one value is diluted by all the other values it's averaged with. Value quantization is lower-sensitivity.

This asymmetry suggests different precision for keys and values: int8 keys with int4 values, or FP8 keys with int8 values. The memory savings from aggressive value quantization can be substantial — values constitute half the KV-cache — while keys maintain higher precision where it matters.

**Worked example: asymmetric key/value quantization.** 70B model, 80 layers, 8 KV-heads, dim 128, 32K tokens.

- Uniform int8 (keys and values): KV-cache \\(= 80 \times 2 \times 8 \times 128 \times 32{,}768 \times 1 = 5.12\\) GB. Memory savings vs float16: 50%.
- Uniform int4: KV-cache \\(= 2.56\\) GB. Savings: 75%. But key quantization error at int4 is severe — attention scores shift by \\(\pm 0.3\\) (from Chapter 5’s noise model at int4 step sizes), causing wrong-token attention. Perplexity increase: ~2–5 points.
- Asymmetric (int8 keys, int4 values): KV-cache \\(= 80 \times 8 \times 128 \times 32{,}768 \times (1 + 0.5) = 3.84\\) GB. Savings: 62.5%. Key precision maintained. Value errors are smoothed by the attention-weighted averaging: a value quantization error of 0.03 is diluted across all tokens being averaged (typically 10–50 tokens with significant attention weight), reducing effective error to \\(0.03 / \sqrt{20} \approx 0.007\\). Perplexity increase: ~0.1–0.5 points — dramatically better than uniform int4.

*Canonical category: reduces effective Resolution Collapse where it matters most (keys, which steer attention) and accepts more in values (which are averaged).*

### When Each Approach Wins

- **Per-token / per-block:** best quality, most metadata and per-token compute. Use when accuracy at long context is critical.
- **Per-head shared scale:** cheapest metadata, highest risk of Resolution Collapse and Budget Waste. Use when sequence lengths are short enough that distribution drift is minimal.
- **Sliding-window updates:** middle ground; introduces Cumulative Rounding Noise if requantizing history. Use when moderate context lengths show measurable drift.
- **Asymmetric key/value precision:** pairs well with any of the above. Use when profiling shows key errors dominate attention degradation.

---

## The Accuracy Impact

KV-cache quantization affects model behavior differently from weight quantization:

**Weight quantization error is static.** The same weights are used for every input. The error is constant and predictable. If a quantized model passes validation, it will behave similarly on production data (absent calibration drift).

**KV-cache quantization error is input-dependent.** Different sequences produce different KV distributions, different quantization errors, and different accuracy impacts. A model that generates coherent text at 4K tokens with int8 KV-cache might produce degraded output at 32K tokens — because the longer sequence exposes distribution shifts that the scale parameters cannot accommodate.

The failure mode is subtle: the model does not crash or produce garbage. It gradually loses coherence — forgetting earlier context, attending to wrong tokens, or producing slightly less precise reasoning. These failures are difficult to detect with standard benchmarks, which typically test short contexts.

---

## Practical State of the Art

The most common deployed approach today:

1. **Keys in int8, values in int8**, with per-head scales updated every \\(N\\) tokens (typically \\(N = 256\\) or \\(N = 512\\)).
2. **FP8 (E4M3)** for both keys and values, which provides a non-uniform grid (Chapter 19) better suited to the peaked distributions of attention projections.
3. **Paged attention** (used in vLLM and TensorRT-LLM) manages KV-cache memory in fixed-size blocks, where quantization operates per-block. This aligns memory management with quantization granularity.

The field is evolving rapidly. New techniques — multi-scale KV quantization, importance-aware eviction (dropping low-attention KV entries instead of quantizing them), and learned quantization functions — are active research directions.

**Diagnostic metrics for KV-cache quantization:** track saturation rate (Tail Clipping), range growth over tokens (Calibration Mismatch / Drift), and attention quality degradation over long contexts (comparing attention weight distributions between quantized and floating-point inference at various sequence lengths).

---

## Conceptual Consolidation

The KV-cache is the second memory wall for LLM inference. It grows linearly with sequence length and, for long contexts, exceeds model weights in size. KV-cache quantization reduces this cost but faces challenges that weight quantization does not: dynamic value distributions, per-token scale requirements, and input-dependent error patterns.

Weight quantization asks "how do I compress a fixed set of parameters?" KV-cache quantization asks "how do I compress a growing, shifting, input-dependent data structure without losing the information the model needs to maintain coherence over long sequences?" The question is harder, and the solutions are less mature.
