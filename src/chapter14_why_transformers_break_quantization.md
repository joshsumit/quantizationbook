# Chapter 14: Why Transformers Break Quantization

In this chapter, we analyze the structural behavior of transformer activations under quantization and isolate the specific mechanisms that cause standard techniques to fail.

## A Different Failure Regime

The quantization methodologies detailed in Chapters 1–13 deliver excellent results for convolutional neural networks and classic feedforward architectures. For instance, standard Post-Training Quantization (PTQ) compresses networks like ResNet-50 to int8 precision while maintaining an accuracy drop well below 1%. These models exhibit compact weight distributions and strictly bounded activation ranges, allowing standard linear quantization grids to map them cleanly.

In contrast, large language models and vision transformers generate activation distributions that systematically violate the foundational assumptions of uniform quantization. Transformers introduce a structurally distinct optimization challenge. Rather than presenting well-behaved parameters, they generate extreme, systematic activation outliers that standard quantization frameworks lack the architectural capacity to resolve.

---

## Activation Outliers

> **Systems Note:** Throughout this section, the term **channels** refers directly to the feature dimensions (hidden units or neurons) of the transformer. While the term traditionally originates from convolutional neural networks, hardware profiling tools, execution kernels, and quantization literature use it because these dimensions map directly to the contiguous memory columns of the activation tensor during matrix multiplication operations.

In transformer models, a small, predictable subset of activation channels consistently produces values that scale multiple orders of magnitude larger than typical features. This systematic variance stems directly from structural properties of the computation.

In a typical large language model layer with 512 output channels, 510 channels produce activations with maximum absolute values below 2.0. The remaining 2 channels regularly generate values reaching 60.0 or 80.0—exceeding the baseline magnitude by 30 to 40 times. These outlier channels exhibit strict spatial consistency, maintaining extreme values across entirely diverse input prompts. They emerge predictably as transformers cross the multi-billion parameter threshold and grow progressively more severe as model scale increases. These attributes emerge directly from the transformer architecture and its internal training dynamics, independent of specific datasets or isolated training runs.

Outliers are most frequently observed in activations immediately following linear projections that take LayerNorm-normalized inputs, since the normalization removes scale and the subsequent linear layers selectively amplify specific directions.

### Origin of Activation Outliers

Transformer architectures execute Layer Normalization (LayerNorm) immediately before each linear projection. LayerNorm rescales activations to enforce unit variance; however, it executes this operation per-token across the hidden dimensions instead of per-channel.

To visualize this interaction at the tensor level, consider an activation tensor entering a LayerNorm block with a 2D shape of `[Sequence Length (Tokens), Hidden Dimension (Channels)]`. The matrix below demonstrates how tokens normalize across their rows, leaving specific channel columns unnormalized vertically:

| Token Index | Channel 1 | Channel 2 | Channel 3 (Outlier) | ... | Channel 512 | Row Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Token 1** | 1.2 | 0.1 | **75.0** | ... | 0.4 | Normalized together as a single row |
| **Token 2** | 0.8 | 0.3 | **82.0** | ... | 0.2 | Normalized together as a single row |
| **Token 3** | 1.1 | 0.2 | **79.0** | ... | 0.5 | Normalized together as a single row |
| **Token N** | 0.9 | 0.1 | **80.0** | ... | 0.3 | Normalized together as a single row |

Looking down the structural column of Channel 3, every single value remains consistently massive because LayerNorm only operates horizontally across the channel vector for each individual token. This horizontal execution completely bypasses vertical normalization across the sequence for an individual channel. Consequently, if a specific channel column consistently produces massive values for every token in a sequence, LayerNorm lacks the mechanism to scale down that column relative to its neighbors.

Specific channels that encode persistent structural information—such as syntax markers or sequence delimiters—accumulate high magnitudes during training. The per-token normalization loop preserves these channel-wise variations, allowing a small subset of features to scale up without triggering a normalization penalty. This behavior serves as a structural mechanism to maintain representational capacity across layers, operating completely apart from training anomalies.

*Canonical category: Distribution Mismatch / Budget Waste.*

---

### Precision Collapse under Per-Tensor Quantization

Per-tensor quantization assigns a single scale factor \\(S\\) to an entire activation tensor. This scale factor must accommodate the absolute maximum value present across all dimensions of the tensor, forcing the uniform quantization grid to span the extreme outliers.

To understand the mathematical collapse this layout causes, consider a concrete structural profiling example. Suppose profiling an activation tensor reveals that its 512 channels split into two distinct behavioral zones:
* **The Normal Range (510 Channels):** The vast majority of channels (99.6% of the layer width) contain standard features. Their values reside completely within a compact baseline range of \\([-2.0, 2.0]\\). This baseline spans a total floating-point window width of 4.0 units (\\(2.0 - (-2.0) = 4.0\\)).
* **The Outlier Range (2 Channels):** A tiny subset of dimensions (0.4% of the layer width) generates extreme values peaking at an absolute maximum of 60.0.

Because symmetric uniform quantization requires a balanced grid centered around zero, the runtime must scale the entire tensor's grid budget using the global absolute peak. The system must therefore construct a quantization window that stretches from -60.0 to +60.0, establishing a total dynamic range window of 120.0 units (\\(60.0 - (-60.0) = 120.0\\)).

An signed int8 data type provides exactly 256 discrete integer levels, spanning from -128 to 127. In a symmetric setup, the dynamic range maps across 255 available steps. The runtime calculates the uniform scale factor \\(S\\) (the real-world value step size between each discrete integer code) by dividing the total outlier-dominated window width by the total step budget:

\\(S = \frac{2 \times 60.0}{255} = \frac{120.0}{255} \approx 0.4706\\)

This scale factor applies uniformly to every channel in the tensor. Each integer point on the int8 grid now stands approximately 0.4706 units apart. 

This massive step size creates a catastrophic resolution crisis for the 510 normal-range channels. Because those standard channels only possess a total real-world window width of 4.0 units (\\[-2.0, 2.0]\\), the number of discrete integer grid levels available to describe them drops sharply. We calculate the usable levels by dividing their entire baseline window by the step size \\(S\\):

\\(\text{Usable Levels with Outliers} = \frac{\text{Normal Window Width}}{\text{Scale Factor } S} = \frac{4.0}{0.4706} \approx 8.5\\)

The post-quantization runtime truncates these 510 channels into just eight or nine discrete integer values (such as codes -4, -3, -2, -1, 0, 1, 2, 3, 4). The original floating-point model relies on millions of fine-grained fractional states within that \\([-2.0, 2.0]\\) neighborhood to isolate subtle behavioral tokens. Compressing 99.6% of the network's features into nine crude levels introduces overwhelming quantization noise and destroys representational capacity.

To isolate the structural damage caused by these two outlier channels, calculate the ideal quantization resolution if they did not exist. Without outliers, the global maximum value drops to the normal channel limit of 2.0. The runtime would calculate an optimized scale factor tailored strictly to the standard baseline range:

\\(S_{\text{ideal}} = \frac{2 \times 2.0}{255} = \frac{4.0}{255} \approx 0.0157\\)

With an ideal step size of 0.0157, the standard channels utilize the complete dynamic range of the int8 data type:

\\(\text{Usable Levels without Outliers} = \frac{\text{Normal Window Width}}{\text{Ideal Scale Factor } S_{\text{ideal}}} = \frac{4.0}{0.0157} \approx 255\\)

Comparing these two scenarios reveals the exact scale of the destruction. We find the resolution reduction ratio by dividing the clean, outlier-free level count by the outlier-degraded level count:

\\(\text{Resolution Reduction Ratio} = \frac{\text{Usable Levels without Outliers}}{\text{Usable Levels with Outliers}} = \frac{255}{8.5} = 30\\)

The 2 outlier channels represent less than 0.4% of the layer's total width:

\\(\frac{\text{Total Outliers}}{\text{Total Channels}} = \frac{2}{512} \approx 0.39\%\\)

Yet, because per-tensor quantization forces a shared grid scale, these two dimensions inflict a 30-fold reduction in resolution across the remaining 99.6% of the features. This structural mismatch makes the activation outlier problem the primary failure mode for naive uniform quantization in multi-billion parameter transformers.

*Canonical category: Resolution Collapse (in normal channels) caused by Distribution Mismatch / Budget Waste.*

---

## Per-Channel Quantization: The Residual Failure Mode

Per-channel quantization assigns a unique scale factor to each output channel. By decoupling the quantization grids, an extreme value in a single outlier channel can no longer contaminate the resolution of neighboring, normal channels. Each channel scales strictly to its own dynamic range.

However, transformer outliers fluctuate dynamically across the token (sequence) dimension rather than remaining statically bound to a single spatial coordinate. While a specific channel may consistently harbor outliers, the magnitude of those outliers shifts aggressively from token to token. 

For example, in a given outlier channel, Token A (a high-signal word or punctuation mark) might fire with an activation magnitude of 80.0, while Token B (a standard filler word) only reaches a magnitude of 3.0. Because standard per-channel quantization applies a single static scale factor across the entire sequence length, it forces a destructive trade-off:

* **Scaling for the Max (80.0):** The scale factor expands to accommodate the peak token. Consequently, the standard tokens with activations near 3.0 suffer severe resolution collapse, mimicking the per-tensor problem.
* **Scaling for the Typical Range (3.0):** The scale factor adapts to preserve resolution for standard tokens. Consequently, the system aggressively clips the peak outlier tokens, saturating the activation to \\(\pm 127\\) and destroying critical high-magnitude features.

Activation outliers present a two-dimensional challenge: they remain channel-local yet vary by token. Outliers exhibit a structural nature because they concentrate in specific channels, but they retain dynamic properties because their amplitudes fluctuate across the sequence. Per-channel quantization solves the spatial distribution but fails to handle the temporal variance.

*Canonical category: Tail Clipping vs Budget Waste trade-off across the token dimension.*

---

## Attention Score Distributions

Transformers compute attention scores that pass through a softmax function. The softmax function concentrates output values heavily near 0 while forcing a small number of dominant values near 1, creating a highly spiky distribution.

A uniform quantization grid distributes grid points evenly across its range. For softmax outputs, this uniform spacing causes immediate resolution imbalances:

* The region near 0 (values 0.00 to 0.05) contains a massive fraction of values representing "not attending" scores. These require high resolution to distinguish between complete non-attention and slight attention.
* The region near 1 (values 0.95 to 1.00) contains a few dominant attention scores that similarly demand high precision.
* The intermediate region from 0.1 to 0.9 contains very few values but receives the vast majority of the uniform grid points.

**Worked Example (Softmax Output Quantization):** An 8-head attention layer produces softmax outputs for a sequence. A typical distribution for one head, where one query token attends to 8 key tokens, yields:

\\([0.001, 0.002, 0.005, 0.012, 0.03, 0.15, 0.35, 0.45]\\)

These values sum to 1.0. Under int8 quantization over the \\([0, 1]\\) range with $S = \frac{1.0}{255} \approx 0.00392$, the grid budget allocates as follows:

| Region | Value Count | Int8 Codes Allocated | Budget Utilization |
| :--- | :--- | :--- | :--- |
| **[0, 0.05)** | 5 values | ~13 codes | 38% |
| **[0.05, 0.5)** | 3 values | ~115 codes | 2.6% |
| **[0.5, 1.0]** | 0 values | ~127 codes | 0% |

The system wastes 115 codes on the \\([0.05, 0.5)\\) region where only 3 values exist, and squanders 127 codes on the \\([0.5, 1.0]\\) range that contains no values at all. The 5 near-zero values receive only 13 codes. Consequently, \\(0.001\\) and \\(0.002\\) both map to code 0, obliterating the model's ability to distinguish between complete non-attention and minor attention signals.

This structural limitation causes a representation error (Chapter 5), as the uniform grid fundamentally mismatches the spiky softmax output distribution.

*Canonical category: Distribution Mismatch / Budget Waste (grid wasted in [0.1, 0.9]); Resolution Collapse near 0 and near 1 where precision matters most.*

---

## The Scale of the Problem

These phenomena represent systemic characteristics of the architecture rather than isolated edge cases. In sufficiently large transformers (multi-billion parameters and above):

* Activation outliers emerge in every linear layer following a LayerNorm block.
* Outlier magnitudes grow linearly with model scale.
* The max-to-normal channel ratio grows dramatically, exceeding 100:1 in massive models.
* Softmax distributions remain highly spiky across every attention head.

**Concrete Example at a 100:1 Ratio:** In a 70B parameter model, a linear layer contains 512 channels. Channels 1–510 peak between 1.8 and 2.1, while channels 511–512 peak at 180 and 210. A per-tensor symmetric scale evaluates to:

\\(S = \frac{210 - (-210)}{255} = \frac{420}{255} \approx 1.65\\)

For the 510 normal channels operating within the \\([-2.1, 2.1]\\) range, the available resolution collapses:

\\(\text{Usable Levels} = \frac{4.2}{1.65} \approx 2.5\\)

This leaves only 2 or 3 distinct representable integer values per channel. The model cannot distinguish an activation of 0.5 from 1.5 within these dimensions. With 99.6% of the channels restricted to 2 or 3 levels of resolution, the layer's output turns into random noise for normal-magnitude features. This catastrophic drop triggers complete representational collapse.

Applying standard int8 PTQ to these models causes severe quality collapse across evaluation metrics, far exceeding the minor degradations observed in convolutional neural networks. The model's outputs become entirely incoherent. While Quantization-Aware Training (QAT) (Chapter 11) mitigates this behavior, full-model fine-tuning at tens of billions of parameters incurs massive operational costs. Production deployments therefore rely on distribution-shaping or selective precision strategies.

---

## Conceptual Consolidation

Transformers break standard quantization because they generate activation distributions with structural outliers (concentrated in specific channels at 10 to 100 times typical magnitudes) and spiky attention scores (clustered at 0 and 1). These properties directly violate the bounded, uniform assumptions that per-tensor and per-channel quantization rely on to maintain precision.

Standard affine uniform quantization incorrectly assumes a single scale can span an entire tensor while preserving useful resolution for the bulk of its values. Transformers break this assumption because the typical features and the extreme outliers remain structurally separated across the channel and token dimensions.

To resolve this, engineers must shift their focus away from choosing alternative static scales. Instead, the core engineering challenge requires transforming the underlying distributions to be quantizable before applying standard grids. Resolving these failures requires distribution-shaping operations or granularity adjustments across the channel and token dimensions prior to running standard compression workflows.

**Critical Failure Signals:**

* Attention quality drop maps sharply to the insertion of quantization blocks.
* Output coherency degrades exponentially as text generation context lengths extend.
* Small permutations in outlier channel magnitudes trigger catastrophic performance swings.