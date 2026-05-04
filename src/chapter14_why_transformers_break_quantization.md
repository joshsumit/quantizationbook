# Chapter 14: Why Transformers Break Quantization

## A Different Failure Regime

Everything in Chapters 1–13 works well for convolutional neural networks and simple feedforward architectures. Models like ResNet-50 quantize to int8 with PTQ and lose less than 1% accuracy. The weight distributions are compact, the activation ranges are bounded, and the standard quantization machinery handles them cleanly.

Transformers are different. Large language models and vision transformers produce activation distributions that violate the assumptions baked into standard quantization. The failure is not "harder quantization" — it is a structurally different problem that the standard tools were not designed for.

---

## Activation Outliers

In transformer models, a small number of activation channels consistently produce values that are orders of magnitude larger than typical activations. This is not noise. It is a structural property of the computation.

In a typical large language model, a given linear layer might have 512 output channels. Of these, 510 channels produce activations with maximum absolute values below 2.0. The remaining 2 channels produce values reaching 60 or 80 — 30 to 40× larger than the typical channels.

These outlier channels are consistent: the same channels produce extreme values across diverse inputs. They emerge in sufficiently large transformers (often at the multi-billion parameter scale) and grow more extreme as model size increases. They are a property of the transformer architecture and its training dynamics, not artifacts of specific datasets or training runs.

Why do they appear? Transformer models use *LayerNorm* (layer normalization) before each linear projection. LayerNorm rescales each layer’s output to have unit variance — but this rescaling is per-token, not per-channel. As a result, a few channels that consistently carry important structural information (e.g., “is this a verb?” or “is this the end of a sentence?”) can grow extreme magnitudes during training without being suppressed, because the per-token normalization doesn’t distinguish between channels. This is not a bug — it is how transformers learn to maintain stable representations.

Outliers are especially common in activations immediately after LayerNorm-projected linear layers.

*Canonical category: Distribution Mismatch / Budget Waste.*

---

## Why Outliers Break Per-Tensor Quantization

Per-tensor quantization assigns a single scale to the entire activation tensor. The scale must accommodate the maximum value in the tensor — including the outlier channels.

Consider the layer described above. Under per-tensor int8 quantization:

$$S = \frac{2 \times 60.0}{255} = \frac{120.0}{255} \approx 0.47$$

The step size is 0.47. For the 510 normal channels with values in [-2.0, 2.0], the number of usable grid points is:

$$\frac{4.0}{0.47} \approx 8.5$$

Eight to nine levels for channels where the model's behavior depends on distinguishing activations at a resolution of 0.001 or better in floating-point. The floating-point model had millions of distinct values in this range. The quantized model has nine.

For comparison, if the outlier channels did not exist:

$$S = \frac{2 \times 2.0}{255} = \frac{4.0}{255} \approx 0.0157$$

The normal channels would get approximately $4.0 / 0.0157 \approx 255$ levels — the full resolution of int8. The outlier channels, which represent less than 0.4% of the channels, are costing the other 99.6% a 30× reduction in resolution.

This is the outlier explosion pattern from Chapter 13, but at a scale and consistency that makes it the dominant failure mode rather than an occasional problem.

*Canonical category: Resolution Collapse (in normal channels) caused by Distribution Mismatch / Budget Waste.*

---

## Per-Channel Does Not Fully Solve It

Per-channel quantization assigns a separate scale to each output channel. This eliminates the problem of outlier channels contaminating normal channels — each channel gets its own range.

But transformer outliers are not only per-channel. They also vary across the token dimension. For a given channel, the outlier magnitude may be 80 for one token and 3 for another. Per-channel quantization uses a single scale per channel across all tokens. If that scale is set by the maximum (80), tokens with activations near 3 get poor resolution. If it is set for the typical range, the outlier tokens clip.

The problem is two-dimensional — channel-local and token-variant: outliers are concentrated in specific channels, but their magnitude varies across tokens. Per-channel quantization addresses the first dimension but not the second.

*Canonical category: Tail Clipping vs Budget Waste trade-off across the token dimension.*

---

## Attention Score Distributions

Transformers compute attention scores that pass through a softmax function. The output of softmax is highly concentrated near 0 with a small number of dominant values near 1 — a spiky distribution rather than a smooth one.

A uniform quantization grid distributes grid points evenly across the range. For softmax outputs:

- The region near 0 (values 0.00 to 0.05) contains a large fraction of values — the "not attending" scores. These need fine resolution to distinguish between "not attending at all" and "attending slightly."
- The region near 1 (values 0.95 to 1.00) contains a few dominant attention scores. These also need fine resolution.
- The region from 0.1 to 0.9 contains very few values but receives the majority of the grid points.

**Worked example: softmax output quantization.** An 8-head attention layer produces softmax outputs for a sequence. A typical distribution for one head, one query token attending to 8 key tokens:

$$[0.001, 0.002, 0.005, 0.012, 0.03, 0.15, 0.35, 0.45]$$

These sum to 1.0. Under int8 quantization over [0, 1] with $S = 1.0/255 \approx 0.00392$:

| Region | Value count | Int8 codes allocated | Utilization |
|---|---|---|---|
| [0, 0.05) | 5 values | ~13 codes | 38% |
| [0.05, 0.5) | 3 values | ~115 codes | 2.6% |
| [0.5, 1.0] | 0 values | ~127 codes | 0% |

115 codes are wasted on the [0.05, 0.5) region where only 3 values exist, and 127 codes cover the [0.5, 1.0] range that has no values at all. The 5 near-zero values — the “not attending” scores — get 13 codes, which means 0.001 and 0.002 both map to code 0 (indistinguishable). Whether a token is “not attending at all” vs “attending very slightly” is lost.

The uniform grid allocates its budget uniformly, but the data distribution is spiky — concentrated near the extremes of [0, 1]. Most of the grid is wasted on an empty region, and the two regions where values actually cluster get inadequate resolution.

This is representation error (Chapter 5) at a structural level: the uniform grid is fundamentally mismatched to the softmax output distribution.

*Canonical category: Distribution Mismatch / Budget Waste (grid wasted in [0.1, 0.9]); Resolution Collapse near 0 and near 1 where precision matters most.*

---

## The Scale of the Problem

These are not edge cases. In sufficiently large transformers (multi-billion parameters and above):

- Activation outliers are present in every linear layer following a LayerNorm
- Outlier magnitudes grow with model scale
- The max/normal ratio can grow dramatically — exceeding 100:1 in the largest models
- Softmax distributions are spiky in every attention head

**Concrete example at 100:1 ratio.** In a 70B model, a linear layer with 512 channels: channels 1–510 peak at 1.8–2.1, channels 511–512 peak at 180 and 210. Per-tensor scale: $S = (210 - (-210)) / 255 = 420 / 255 \approx 1.65$. For normal channels with range [-2.1, 2.1]: usable levels $= 4.2 / 1.65 \approx 2.5$ — effectively 2–3 distinct representable values per channel. The model cannot distinguish an activation of 0.5 from 1.5 in those channels. With 510 out of 512 channels at 2–3 levels, the layer’s output is essentially random for normal-magnitude channels. This is not “slightly degraded” — it is complete representational collapse.

Standard int8 PTQ on these models can cause severe quality collapse (task- and metric-dependent), often far beyond the small degradations seen in CNNs. The model's outputs become incoherent. QAT (Chapter 11) can help, but full-model fine-tuning at tens of billions of parameters is operationally expensive; many deployments therefore prefer distribution-shaping or selective precision strategies.

The standard quantization machinery from Chapters 1–13 — designed for smooth, bounded, channel-balanced distributions — encounters distributions that are none of these things. Different approaches are required.

---

## Conceptual Consolidation

Transformers break standard quantization because they produce activation distributions with structural outliers (specific channels, 10–100× typical magnitude) and spiky attention scores (concentrated at 0 and 1). These properties violate the assumption of bounded, approximately uniform distributions that per-tensor and per-channel quantization rely on.

The core assumption that breaks: standard affine uniform quantization assumes a single scale can cover the tensor while preserving useful resolution for the bulk of values. Transformers violate this because the bulk and the extremes are structurally separated — by channel and by token.

The question is no longer "what scale should we use?" It is: how do we transform the distributions to be quantizable before applying the standard machinery? We need distribution-shaping or granularity changes (channel × token) before the standard quantization tools from Chapters 1–13 can work.
