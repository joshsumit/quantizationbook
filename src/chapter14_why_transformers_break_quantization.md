# Chapter 14: Why Transformers Break Quantization

In this chapter, we quantize transformer activations and show where that fails.

## A Different Failure Regime

Everything in Chapters 1–13 works well for convolutional neural networks and simple feedforward architectures. Models like ResNet-50 quantize to INT8 using Post-Training Quantization (PTQ) while losing less than 1% accuracy. Their weight distributions are compact, activation ranges are bounded, and standard quantization handles them cleanly.

Large language models and vision transformers produce activation distributions that violate the core assumptions of standard quantization. Transformers pose a structurally different problem, driven by extreme activation outliers, that standard tools were not designed to solve.

---

## Activation Outliers

> **Systems Note:** Throughout this section, we use the term **channels** to refer to the feature dimensions (or hidden units/neurons) of the transformer. While "channel" traditionally originates from convolutional neural networks, hardware profiling tools, kernels, and quantization literature use it because these dimensions map directly to the contiguous memory columns of the activation tensor during matrix multiplication.

In transformer models, a small number of activation channels consistently produce values that are orders of magnitude larger than typical activations. This is a structural property of the computation, not noise.

In a typical large language model, a given linear layer might have 512 output channels. Of these, 510 channels produce activations with maximum absolute values below 2.0. The remaining 2 channels produce values reaching 60 or 80—30 to 40× larger than the typical channels. These outlier channels are consistent: the same channels produce extreme values across diverse inputs. They emerge in sufficiently large transformers (often at the multi-billion parameter scale) and grow more extreme as model size increases. They are a property of the transformer architecture and its training dynamics, not artifacts of specific datasets or training runs.

Outliers are especially common in activations immediately after LayerNorm-projected linear layers.

### Origin of Activation Outliers

Transformer architectures use Layer Normalization (*LayerNorm*) before each linear projection. LayerNorm rescales the activations to ensure unit variance, but this operation is executed per-token across the hidden dimensions rather than per-channel. 

To visualize this at the hardware and tensor level, consider an activation tensor entering a LayerNorm block with a 2D shape of `[Sequence Length (Tokens), Hidden Dimension (Channels)]`:

```text
               CHANNELS (e.g., 512)
             C1   C2   C3  ...  C512
   Token 1 [ 1.2, 0.1, 75.0, ..., 0.4 ]  --> Normalized together as one row
T  Token 2 [ 0.8, 0.3, 82.0, ..., 0.2 ]  --> Normalized together as one row
O  Token 3 [ 1.1, 0.2, 79.0, ..., 0.5 ]  --> Normalized together as one row
K  ...
N  Token N [ 0.9, 0.1, 80.0, ..., 0.3 ]  --> Normalized together as one row
                        |
                        v
             Look down Channel 3 (C3):
             Every single value is massive.
```

Because LayerNorm operates horizontally across the channel vector for each individual token, it never normalizes vertically across the sequence for a single channel. This means if a specific channel column consistently produces massive values for every token in a sequence, LayerNorm has no mechanism to scale down that specific column relative to its neighbors.

Consequently, specific channels that encode persistent structural information—such as syntax markers or sequence delimiters—can accumulate high magnitudes during training. The per-token normalization loop preserves these channel-wise variations, allowing a small subset of features to scale up without triggering a normalization penalty. This behavior is a structural mechanism of the transformer architecture to maintain representational capacity across layers, not a training anomaly.

*Canonical category: Distribution Mismatch / Budget Waste.*

---

### Precision Collapse under Per-Tensor Quantization

Per-tensor quantization assigns a single scale factor \\($S$\\) to an entire activation tensor. This scale factor must accommodate the absolute maximum value present in the tensor, forcing the quantization grid to span the extreme outliers.

Consider a layer with 512 channels under symmetric per-tensor `int8` quantization. In this scenario, 2 channels act as extreme outliers that peak at 60.0, leaving the remaining 510 channels operating within a normal baseline range. 

Because symmetric quantization requires a balanced grid around zero, the dynamic range must span from -60.0 to +60.0. The total window width is calculated by multiplying the peak magnitude by 2 \\($2 \times 60.0$\\) 

The scale factor is calculated as:

$$S = \frac{2 \times 60.0}{255} = \frac{120.0}{255} \approx 0.47$$

With a quantization step size of 0.47, the 510 typical channels—which we will assume for this example lie strictly within a baseline \\([-2.0, 2.0]\\) range—are compressed into a heavily constrained set of usable grid points:

$$\text{Usable Levels with Outliers} = \frac{4.0}{0.47} \approx 8.5$$

The floating-point model relies on millions of distinct values within this standard range to capture fine-grained behavioral signals at a resolution of 0.001 or finer. Post-quantization, these channels are truncated into just eight or nine discrete integer levels, resulting in massive quantization noise and representation collapse.

To isolate the impact of these outliers, consider the quantization resolution if they did not exist. The scale factor for a maximum value of 2.0 would be:

$$S = \frac{2 \times 2.0}{255} = \frac{4.0}{255} \approx 0.0157$$

Without outliers, the standard channels would utilize the entire dynamic range of the `int8` data type. We find the available resolution by dividing the baseline range width by the clean step size:

$$\text{Usable Levels without outliers} = \frac{4.0}{0.0157} \approx 255$$

The 2 outlier channels represent less than 0.4% of the layer's width \\(\frac{\text{Total Outliers}}{\text{Total Channels}} = \frac{2}{512} \approx 0.39\%\\), yet they cost the remaining 99.6% of the features a 30× reduction in resolution \\(\frac{\text{Levels without Outliers}}{\text{Levels with Outliers}} = \frac{255}{8.5} = 30\\).

This structural phenomenon makes the activation outlier problem the primary failure mode for naive uniform quantization in multi-billion parameter transformers.

*Canonical category: Resolution Collapse (in normal channels) caused by Distribution Mismatch / Budget Waste.*

---

## Per-Channel Quantization: The Residual Failure Mode

Per-channel quantization assigns a unique scale factor to each output channel. By decoupling the quantization grids, an extreme value in a single outlier channel can no longer contaminate the resolution of neighboring, normal channels. Each channel scales strictly to its own dynamic range.

However, transformer outliers are not statically bound to a single dimension; they vary dynamically across the token (sequence) dimension. While a specific channel may consistently harbor outliers, the magnitude of those outliers fluctuates aggressively from token to token. 

> **For example:** In a given outlier channel, Token A (e.g., a high-signal word or punctuation mark) might fire with an activation magnitude of 80.0, while Token B (e.g., a standard filler word) might only reach a magnitude of 3.0.

Because standard per-channel quantization applies a single static scale factor across the entire sequence length, it forces a destructive trade-off:

* **Scaling for the Max (80.0):** The scale factor expands to accommodate the peak token. Consequently, the standard tokens with activations near 3.0 suffer severe resolution collapse, similar to the per-tensor problem.
* **Scaling for the Typical Range (3.0):** The scale factor tightens to preserve resolution for standard tokens. Consequently, the peak outlier tokens are aggressively clipped, saturating the activation to $\pm 127$ and destroying critical high-magnitude features.

The fundamental limitation is that activation outliers are a two-dimensional problem—**channel-local** and **token-variant**. Outliers are structural because they concentrate in specific channels, but they are dynamic because their amplitudes fluctuate across the sequence. Per-channel quantization solves the spatial distribution but fails to handle the temporal variance.

*Canonical category: Tail Clipping vs Budget Waste trade-off across the token dimension.*

---

## Attention Score Distributions

Transformers compute attention scores that pass through a softmax function. The output of softmax is highly concentrated near 0 with a small number of dominant values near 1 â€” a spiky distribution rather than a smooth one.

A uniform quantization grid distributes grid points evenly across the range. For softmax outputs:

- The region near 0 (values 0.00 to 0.05) contains a large fraction of values â€” the "not attending" scores. These need fine resolution to distinguish between "not attending at all" and "attending slightly."
- The region near 1 (values 0.95 to 1.00) contains a few dominant attention scores. These also need fine resolution.
- The region from 0.1 to 0.9 contains very few values but receives the majority of the grid points.

**Worked example: softmax output quantization.** An 8-head attention layer produces softmax outputs for a sequence. A typical distribution for one head, one query token attending to 8 key tokens:

$$[0.001, 0.002, 0.005, 0.012, 0.03, 0.15, 0.35, 0.45]$$

These sum to 1.0. Under int8 quantization over [0, 1] with \\(S = 1.0/255 \approx 0.00392\\):

| Region | Value count | Int8 codes allocated | Utilization |
|---|---|---|---|
| [0, 0.05) | 5 values | ~13 codes | 38% |
| [0.05, 0.5) | 3 values | ~115 codes | 2.6% |
| [0.5, 1.0] | 0 values | ~127 codes | 0% |

115 codes are wasted on the [0.05, 0.5) region where only 3 values exist, and 127 codes cover the [0.5, 1.0] range that has no values at all. The 5 near-zero values â€” the â€œnot attendingâ€ scores â€” get 13 codes, which means 0.001 and 0.002 both map to code 0 (indistinguishable). Whether a token is â€œnot attending at allâ€ vs â€œattending very slightlyâ€ is lost.

The uniform grid allocates its budget uniformly, but the data distribution is spiky â€” concentrated near the extremes of [0, 1]. Most of the grid is wasted on an empty region, and the two regions where values actually cluster get inadequate resolution.

This is representation error (Chapter 5) at a structural level: the uniform grid is fundamentally mismatched to the softmax output distribution.

*Canonical category: Distribution Mismatch / Budget Waste (grid wasted in [0.1, 0.9]); Resolution Collapse near 0 and near 1 where precision matters most.*

---

## The Scale of the Problem

These are not edge cases. In sufficiently large transformers (multi-billion parameters and above):

- Activation outliers are present in every linear layer following a LayerNorm
- Outlier magnitudes grow with model scale
- The max/normal ratio can grow dramatically â€” exceeding 100:1 in the largest models
- Softmax distributions are spiky in every attention head

**Concrete example at 100:1 ratio.** In a 70B model, a linear layer with 512 channels: channels 1â€“510 peak at 1.8â€“2.1, channels 511â€“512 peak at 180 and 210. Per-tensor scale: \\(S = (210 - (-210)) / 255 = 420 / 255 \approx 1.65\\). For normal channels with range [-2.1, 2.1]: usable levels \\(= 4.2 / 1.65 \approx 2.5\\) â€” effectively 2â€“3 distinct representable values per channel. The model cannot distinguish an activation of 0.5 from 1.5 in those channels. With 510 out of 512 channels at 2â€“3 levels, the layerâ€™s output is essentially random for normal-magnitude channels. This is not â€œslightly degradedâ€ â€” it is complete representational collapse.

Standard int8 PTQ on these models can cause severe quality collapse (task- and metric-dependent), often far beyond the small degradations seen in CNNs. The model's outputs become incoherent. QAT (Chapter 11) can help, but full-model fine-tuning at tens of billions of parameters is operationally expensive; many deployments therefore prefer distribution-shaping or selective precision strategies.

The standard quantization machinery from Chapters 1â€“13 â€” designed for smooth, bounded, channel-balanced distributions â€” encounters distributions that are none of these things. Different approaches are required.

---

## Conceptual Consolidation

Transformers break standard quantization because they produce activation distributions with structural outliers (specific channels, 10â€“100Ã— typical magnitude) and spiky attention scores (concentrated at 0 and 1). These properties violate the assumption of bounded, approximately uniform distributions that per-tensor and per-channel quantization rely on.

The core assumption that breaks: standard affine uniform quantization assumes a single scale can cover the tensor while preserving useful resolution for the bulk of values. Transformers violate this because the bulk and the extremes are structurally separated â€” by channel and by token.

The question is no longer "what scale should we use?" It is: how do we transform the distributions to be quantizable before applying the standard machinery? We need distribution-shaping or granularity changes (channel Ã— token) before the standard quantization tools from Chapters 1â€“13 can work.

**Failure Signals**

- Attention quality drops sharply under quantization
- Outputs become incoherent at longer contexts
- Extreme sensitivity to outlier channels

