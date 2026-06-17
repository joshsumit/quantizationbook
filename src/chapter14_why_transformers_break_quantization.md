# Chapter 14: Why Transformers Break Quantization

In this chapter, we analyze the structural behavior of transformer activations under quantization and isolate the specific mechanisms that cause standard techniques to fail.

## A Different Failure Regime

The quantization methodologies detailed in Chapters 1–13 deliver excellent results for convolutional neural networks (CNNs) and classic feedforward architectures. For instance, standard Post-Training Quantization (PTQ) compresses networks like ResNet-50 to `int8` precision while keeping the accuracy drop well below 1%. These models exhibit compact weight distributions and strictly bounded activation ranges, allowing standard uniform grids to map them cleanly.



In contrast, large language models (LLMs) and vision transformers (ViTs) generate activation distributions that systematically break the core assumptions of uniform quantization. Transformers introduce a structurally distinct optimization challenge. Instead of well-behaved parameters, they generate extreme, systematic activation outliers that standard quantization frameworks simply lack the architectural capacity to resolve.

---

## Activation Outliers

> **Systems Note:** Throughout this section, the term **channels** refers directly to the feature dimensions (hidden units or neurons) of the transformer. While this term originates from CNNs, hardware profiling tools, execution kernels, and quantization literature use it because these dimensions map directly to the contiguous memory columns of the activation tensor during matrix multiplication.

In transformer models, a small, predictable subset of activation channels consistently produces values multiple orders of magnitude larger than typical features. This systematic variance stems directly from the structural design of the attention and feedforward mechanisms.

Consider a typical LLM layer with 512 output channels. 510 of these channels produce normal activations with maximum absolute values below 2.0. The remaining 2 channels regularly generate values reaching 60.0 or 80.0—exceeding the baseline magnitude by 30 to 40 times. 

These outlier channels exhibit strict spatial consistency, maintaining extreme values across entirely diverse input prompts. They emerge predictably as transformers cross the multi-billion parameter threshold and grow progressively more severe as the model scale increases. This behavior is a structural characteristic of the transformer architecture and its internal training dynamics, independent of specific datasets or isolated training runs.

Outliers appear most frequently in activations immediately following linear projections that ingest LayerNorm-normalized inputs. This occurs because the normalization step flattens the scale across tokens, and the subsequent linear layers selectively amplify specific feature directions.

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

Per-tensor quantization enforces a single scale factor \\(S\\) across an entire activation tensor. In the presence of outliers, this global grid budget must span the absolute peak value, destroying the resolution of normal-range channels.

Consider an activation tensor with 512 channels split into two behavioral zones:
* **Normal Range (510 channels):** Values reside within \\([-2.0, 2.0]\\). 
* **Outlier Range (2 channels):** Values peak at an absolute maximum of \\(60.0\\).

Symmetric uniform quantization requires a balanced grid centered around zero. The runtime must scale the entire tensor using the absolute maximum value (\\(x_{\text{max}} = 60.0\\)), establishing a total dynamic window width of:

\\(\text{Total Window Width} = 2 \times x_{\text{max}} = 2 \times 60.0 = 120.0\\)

A signed int8 format maps this dynamic range across 255 available symmetric steps. The uniform scale factor \\(S\\) represents the real-world step size between discrete integer codes:

\\(S = \frac{2 \times x_{\text{max}}}{255} = \frac{120.0}{255} \approx 0.4706\\)

Because this scale factor applies globally, it forces a catastrophic resolution collapse on the 510 normal-range channels. We calculate the discrete levels available to represent their entire \\(4.0\\)-unit wide baseline range (\\(2.0 - (-2.0) = 4.0\\)) by dividing by the step size \\(S\\):

\\(\text{Usable Levels with Outliers} = \frac{\text{Normal Baseline Width}}{S} = \frac{4.0}{0.4706} \approx 8.5\\)



To see why this \\(8.5\\) value is a massive red flag for network precision, let's trace how distinct floating-point activations inside the normal channels map onto this coarse grid. 

#### Scenario 1: Total Collapse to Zero
Consider three small, distinct activations: \\(x_1 = 0.11\\), \\(x_2 = 0.22\\), and \\(x_3 = 0.23\\). When passing through the quantization kernel (\\(\text{round}(x / S)\\)), the massive step size of \\(0.4706\\) flattens them completely:

\\(\text{Quantized } x_1 = \text{round}\left(\frac{0.11}{0.4706}\right) = \text{round}(0.233) = \mathbf{0}\\)  
\\(\text{Quantized } x_2 = \text{round}\left(\frac{0.22}{0.4706}\right) = \text{round}(0.467) = \mathbf{0}\\)  
\\(\text{Quantized } x_3 = \text{round}\left(\frac{0.23}{0.4706}\right) = \text{round}(0.488) = \mathbf{0}\\)  

#### Scenario 2: Information Loss in Non-Zero Buckets
The destruction isn't limited to zero. Higher activations escape zero but still suffer massive information loss by getting shoved into the exact same integer bucket. Consider \\(y_1 = 1.0\\), \\(y_2 = 1.15\\), and \\(y_3 = 1.3\\):

\\(\text{Quantized } y_1 = \text{round}\left(\frac{1.0}{0.4706}\right) = \text{round}(2.125) = \mathbf{2}\\)  
\\(\text{Quantized } y_2 = \text{round}\left(\frac{1.15}{0.4706}\right) = \text{round}(2.444) = \mathbf{2}\\)  
\\(\text{Quantized } y_3 = \text{round}\left(\frac{1.3}{0.4706}\right) = \text{round}(2.762) = \mathbf{3}\\)  

Here, both \\(1.0\\) and \\(1.15\\) collapse into the integer bucket \\(\mathbf{2}\\). When dequantized back to a float via (\\(q \times S\\)), both reconstruct to exactly \\(2 \times 0.4706 = 0.9412\\). The original \\(15\%\\) variance between these features is permanently lost, introducing substantial quantization noise.

---

### The Ideal Scenario (Isolating the Outlier Impact)

To isolate the structural impact of these outliers, consider the ideal scenario if they did not exist. The tensor maximum drops to \\(x_{\text{normal\_max}} = 2.0\\), yielding an optimized step size:

\\(S_{\text{ideal}} = \frac{2 \times 2.0}{255} = \frac{4.0}{255} \approx 0.0157\\)

Without outliers, the standard channels utilize the full dynamic budget of the data type:

\\(\text{Usable Levels without Outliers} = \frac{\text{Normal Baseline Width}}{S_{\text{ideal}}} = \frac{4.0}{0.0157} \approx 255\\)

Let's re-run our previous inputs through this optimized grid to see how high-resolution representation is preserved:

#### Tracking the Low Values (\\(x_1, x_2, x_3\\)) on the Ideal Grid:
\\(\text{Quantized } x_1 = \text{round}\left(\frac{0.11}{0.0157}\right) = \text{round}(7.006) = \mathbf{7}\\)  
\\(\text{Quantized } x_2 = \text{round}\left(\frac{0.22}{0.0157}\right) = \text{round}(14.012) = \mathbf{14}\\)  
\\(\text{Quantized } x_3 = \text{round}\left(\frac{0.23}{0.0157}\right) = \text{round}(14.650) = \mathbf{15}\\)  

#### Tracking the High Values (\\(y_1, y_2, y_3\\)) on the Ideal Grid:
\\(\text{Quantized } y_1 = \text{round}\left(\frac{1.0}{0.0157}\right) = \text{round}(63.694) = \mathbf{64}\\)  
\\(\text{Quantized } y_2 = \text{round}\left(\frac{1.15}{0.0157}\right) = \text{round}(73.248) = \mathbf{73}\\)  
\\(\text{Quantized } y_3 = \text{round}\left(\frac{1.3}{0.0157}\right) = \text{round}(82.802) = \mathbf{83}\\)  

Every single float value now maps to its own dedicated, unique integer bucket. No signals collapse, and feature variance is perfectly preserved.

---

### The Structural Destruction

We quantify the structural destruction by calculating the resolution reduction ratio:

\\(\text{Resolution Reduction Ratio} = \frac{\text{Usable Levels without Outliers}}{\text{Usable Levels with Outliers}} = \frac{255}{8.5} = 30\\)

The two outlier channels constitute a tiny fraction of the layer width:

\\(\text{Outlier Ratio} = \frac{2}{512} \approx 0.39\%\\)

Yet, because per-tensor quantization forces a shared grid scale, this \\(0.39\%\\) of dimensions inflicts a 30-fold resolution drop across the remaining \\(99.6\%\\) of the features. This mathematical mismatch makes the activation outlier problem the primary failure mode for naive uniform quantization in multi-billion parameter transformers.

*Canonical category: Resolution Collapse (in normal channels) caused by Distribution Mismatch / Budget Waste.*

---

## Per-Channel Quantization: The Residual Failure Mode

Per-channel quantization assigns a unique scale factor to each individual channel. By decoupling these quantization grids, an extreme value in an outlier channel cannot contaminate the resolution of other channels across the tensor. Each channel scales strictly to its own local dynamic range.

However, while these outliers are spatially bound to specific channels, their magnitudes fluctuate dynamically across the token (sequence) dimension. While a specific channel column consistently harbors outliers, the amplitude of those spikes shifts aggressively from token to token. 

For example, inside a designated outlier channel, Token A (a high-signal token or punctuation mark) might fire with an activation magnitude of 60.0, while Token B (a standard filler word) only reaches a magnitude of 2.0. Because standard per-channel activation quantization scales the tensor across the sequence dimension, a single scale factor must span the entire token history, forcing a destructive trade-off:

* **Scaling for the Peak (60.0):** The scale factor expands to accommodate the maximum outlier token. Consequently, the standard tokens with activations near 2.0 suffer severe resolution collapse, reintroducing the original per-tensor precision problem.
* **Scaling for the Typical Range (2.0):** The scale factor shrinks to preserve resolution for standard tokens. Consequently, the system aggressively clips the peak outlier tokens, saturating the activation to \\(\pm 127\\) and obliterating critical high-magnitude features.

Activation outliers present a two-dimensional challenge: they remain channel-local yet vary by token. They are structural because they concentrate in specific channels, but they are dynamic because their amplitudes fluctuate across the sequence sequence. Per-channel quantization isolates the variance across channels but fails to handle the dynamic variance across tokens.

*Canonical category: Tail Clipping vs Budget Waste trade-off across the token dimension.*

---
## Attention Score Distributions

Transformers compute attention scores by passing them through a softmax function. This operation concentrates values heavily near 0 while forcing a small number of dominant values near 1, creating a highly "spiky" distribution.

A uniform quantization grid distributes its points evenly across its target range. For softmax outputs, this uniform spacing causes immediate resolution imbalances:

* **The near-zero region (\\(0.00\\) to \\(0.05\\)):** Contains the massive majority of values representing "non-attention" scores. These require high resolution to distinguish between complete non-attention and subtle, minor attention signals.
* **The near-one region (\\(0.95\\) to \\(1.00\\)):** Contains a few dominant attention scores that similarly demand high precision.
* **The intermediate region (\\(0.1\\) to \\(0.9\\)):** Contains very few values, yet receives the vast majority of the uniform grid points.

**Worked Example (Softmax Output Quantization):** Consider a single attention head where one query token attends to 8 key tokens. A typical post-softmax distribution yields:

\\[[0.001, 0.002, 0.005, 0.012, 0.03, 0.15, 0.35, 0.45]\\]

These values sum to 1.0. Under `int8` quantization over the \\([0, 1]\\) range with a scale factor of \\(S = \frac{1.0}{255} \approx 0.00392\\), the grid budget allocates as follows:

| Region | Value Count | Int8 Codes Allocated | Budget Utilization |
| :--- | :--- | :--- | :--- |
| **[0, 0.05)** | 5 values | ~13 codes | 5.1% |
| **[0.05, 0.5)** | 3 values | ~115 codes | 45.1% |
| **[0.5, 1.0]** | 0 values | ~127 codes | 49.8% |

The system wastes 115 codes on the \\([0.05, 0.5)\\) region where only 3 values exist, and squanders 127 codes on the \\([0.5, 1.0]\\) range that contains no values at all. Conversely, the 5 critical near-zero values must share just 13 codes. 

Consequently, \\(0.001\\) and \\(0.002\\) both map to code 0:

\\(\text{Quantized } x_1 = \text{round}\left(\frac{0.001}{0.00392}\right) = \text{round}(0.255) = \mathbf{0}\\)  
\\(\text{Quantized } x_2 = \text{round}\left(\frac{0.002}{0.00392}\right) = \text{round}(0.510) = \mathbf{0}\\)  

This obliterates the model's ability to distinguish between complete non-attention and minor attention signals. This structural limitation causes a representation error (Chapter 5), as the uniform grid fundamentally mismatches the spiky softmax output distribution.

---

## The Scale of the Problem

These phenomena are systemic characteristics of the transformer architecture rather than isolated edge cases. In sufficiently large models (multi-billion parameters and above):

* Activation outliers emerge in every linear layer following a LayerNorm block.
* Outlier magnitudes grow linearly with model scale.
* The max-to-normal channel ratio grows dramatically, exceeding 100:1 in massive models.
* Softmax distributions remain highly spiky across every attention head.

**Concrete Example at a 100:1 Ratio:** In a 70B parameter model, a linear layer contains 512 channels. Channels 1–510 peak between 1.8 and 2.1, while channels 511–512 peak at 180 and 210. A per-tensor symmetric scale evaluates to:

\\(S = \frac{210 - (-210)}{255} = \frac{420}{255} \approx 1.647\\)

For the 510 normal channels operating within the \\([-2.1, 2.1]\\) range, the available resolution collapses:

\\(\text{Usable Levels} = \frac{\text{Normal Width}}{S} = \frac{4.2}{1.647} \approx 2.55\\)



This leaves only 2 or 3 distinct integer values per channel. The model cannot distinguish an activation of 0.5 from 1.5 within these dimensions. With 99.6% of the channels restricted to 2 or 3 levels of resolution, the layer's output turns into crude quantization noise for normal-magnitude features, triggering complete representational collapse.

Applying standard `int8` PTQ to these models causes severe quality collapse across evaluation metrics, far exceeding the minor degradations observed in CNNs. The model's outputs become entirely incoherent. While Quantization-Aware Training (QAT) (Chapter 11) can mitigate this behavior, full-model fine-tuning at tens of billions of parameters incurs massive operational costs. Production deployments therefore rely on distribution-shaping or mixed-precision strategies.

---

## Conceptual Consolidation

Transformers break standard quantization because they generate activation distributions with structural outliers (concentrated in specific channels at 10 to 100 times typical magnitudes) and spiky attention scores (clustered heavily at 0 and 1). These properties directly violate the bounded, uniform assumptions that per-tensor and per-channel quantization schemes rely on to maintain precision.

Standard uniform quantization assumes a single scale factor can span an entire tensor while preserving useful resolution for the bulk of its values. Transformers break this assumption because the typical features and the extreme outliers remain structurally separated across the channel and token dimensions.

To resolve this, engineers must shift their focus away from simply searching for alternative static scales. Instead, the core systems challenge requires transforming the underlying distributions to be "quantizable" before applying standard grids. Resolving these failures requires distribution-shaping operations or granularity adjustments across the channel and token dimensions prior to running standard compression workflows.

**Critical Failure Signals in Production Pipelines:**

* Attention quality drop maps sharply to the insertion of quantization blocks.
* Output coherency degrades exponentially as text generation context lengths extend.
* Small permutations in outlier channel magnitudes trigger catastrophic performance swings.