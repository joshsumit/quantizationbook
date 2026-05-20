# Chapter 5: Where Error Is Born

In this chapter, we track errors produced when weights and activations are quantized.

## Three Kinds of Error

When a continuous value is quantized, a loss of information occurs. However, these losses are not uniform. Quantization introduces three distinct types of error, each characterized by a different cause, a different magnitude, and a different method of mitigation. Conflating these errors leads to incorrect system diagnosis. This chapter provides the precise definition for each type.

---

## Rounding Error

Rounding error is the simplest form of quantization error. It occurs when a continuous value falls between two adjacent grid points and is mapped to the nearest one.

Consider a range [-1.0, 1.0] quantized to int8 with 256 levels. The step size (Chapter 2) is:

$$S = \frac{2.0}{255} \approx 0.00784$$

The grid points are spaced at intervals of 0.00784. The value 0.5023 falls between grid points at 0.4980 and 0.5059. Mapping explicitly:

$$q = \text{round}\!\left(\frac{0.5023 - (-1.0)}{0.00784}\right) = \text{round}(191.54) = 192$$

$$r' = -1.0 + 192 \times 0.00784 \approx 0.5059$$

The error is:

$$|0.5023 - 0.5059| = 0.0036$$

This error is strictly bounded; the worst case for any rounding operation is exactly half the step size:

$$\text{max rounding error} = \frac{S}{2} \approx 0.0039$$

Because no value within the representable range can exceed this limit, rounding error remains small, symmetric (equally likely to round up or down), and predictable. It represents the inherent penalty of operating on a discrete grid—a baseline distortion that cannot be eliminated, but can be mitigated.

*Accuracy pattern: Rounding Noise — smooth, gradual degradation that improves with more bits or per-channel scales.*

### Quantizing the Noise: A First-Order Model

Rounding error can be modeled as additive noise. When a value \\(r\\) is quantized, the reconstructed value is \\(\hat{r} = r + \epsilon\\), where \\(\epsilon\\) represents the rounding error. Assuming the distribution of \\(r\\) is not systematically aligned with the quantization grid—a valid assumption for neural network weights and activations—\\(\epsilon\\) behaves approximately as a uniform random variable on the interval \\([-S/2, +S/2]\\). The variance of this uniform distribution is defined by the classical signal processing quantization noise formula:

\\[\text{Var}(\epsilon) = \frac{S^2}{12}\\]

This formula provides a direct method for estimating error magnitude across different bit-widths. For example, an int8 representation over the range \\([-1, 1]\\) yields a step size of \\(S = 0.00784\\), resulting in a variance of \\(\text{Var}(\epsilon) = 5.12 \times 10^{-6}\\) and a standard deviation of \\(\sigma \approx 0.0023\\). Conversely, an int4 representation over the same range increases the step size to \\(S = 0.133\\), which escalates the variance to \\(\text{Var}(\epsilon) = 1.48 \times 10^{-3}\\) and the standard deviation to \\(\sigma \approx 0.038\\)—approximately a 17-fold increase in noise magnitude.

When executing a dot product of \\(N\\) independently quantized values, such as a single output channel in a linear layer, the output noise variance accumulates according to the following relationship:

\\[\text{Var}(\epsilon_{\text{output}}) \approx N \times \text{Var}(\epsilon_w) \times \mathbb{E}[x^2] + N \times \text{Var}(\epsilon_x) \times \mathbb{E}[w^2]\\]

Here, \\(\epsilon_w\\) and \\(\epsilon_x\\) represent the weight and activation rounding errors, respectively. The terms \\(\mathbb{E}[x^2]\\) and \\(\mathbb{E}[w^2]\\) denote the expected values (the average squared magnitudes) of the unquantized activations and weights. 
This relationship demonstrates that the accumulated output noise is driven not only by the precision of the quantization grid itself, but also by the signal scaling factors: weight noise is amplified by the magnitude of the incoming activations, while activation noise is amplified by the magnitude of the static weights. 

Because the output noise variance scales linearly with the dot-product width \\(N\\), a 4096-wide vector accumulates 4096 times more noise variance than a single scalar multiplication. To isolate and illustrate this growth pattern, the table below tracks the accumulated variance and standard deviation across varying dimensions, assuming \\(\mathbb{E}[x^2] \approx 1.0\\) and isolating weight quantization noise (\\(S = 0.00784\\), \\(\sigma = 0.0023\\)) for simplicity.


| Dot-product width \\(N\\) | Accumulated \\(\text{Var}\\) | Accumulated \\(\sigma\\) |
| :--- | :--- | :--- |
| 1 | \\(5.12 \times 10^{-6}\\) | 0.0023 |
| 64 | \\(3.28 \times 10^{-4}\\) | 0.018 |
| 512 | \\(2.62 \times 10^{-3}\\) | 0.051 |
| 1024 | \\(5.24 \times 10^{-3}\\) | 0.072 |
| 4096 | \\(2.10 \times 10^{-2}\\) | 0.145 |

While a single quantized scalar introduces a baseline noise standard deviation of only 0.0023, a 4096-wide dot product amplifies that standard deviation to 0.145—an increase of over 60 times. For a network layer where typical output values span the range $[-1, 1]$, a noise standard deviation of 0.145 constitutes roughly 15% of the total signal amplitude. This compounding effect explains why wider layers in large-scale models exhibit a higher sensitivity to quantization degradation than narrower architectures.



### What Quantization Noise Is and What It Is Not

The word "noise" invites a misconception. In other machine learning contexts, noise is sometimes injected deliberately â€” as regularization, as data augmentation, or as a source of stochasticity. Quantization noise is none of those things.

Quantization noise is a *measurement of damage*. When a value \\(r\\) becomes \\(\hat{r} = r + \epsilon\\), the \\(\epsilon\\) is not stored, not fed into another module, and not corrected later. It is baked into the output â€” permanently and invisibly. At inference time, the quantized model simply runs; there is no mechanism that detects or compensates for the accumulated error.

So what is this noise actually *used for*? Not by the model â€” by the engineer. Quantization noise is an engineering diagnostic:

- **To decide where quantization is safe.** Measuring noise per layer tells you which layers tolerate int8 and which must stay in FP16. This is the basis of mixed-precision policies (Chapter 12).
- **To choose scales and formats.** Calibration (Chapter 9) runs candidate configurations and compares the noise they produce â€” per-tensor vs per-channel, symmetric vs asymmetric, percentile vs min-max. The configuration with less noise wins.
- **To predict accuracy drop before deployment.** Metrics like signal-to-quantization-noise ratio (SQNR) or per-layer activation MSE let you estimate accuracy degradation without deploying the model.

  **SQNR worked example.** For the int8 case with \\(\sigma_{\text{noise}} = 0.0023\\) and a signal with std \\(\sigma_{\text{signal}} = 0.2\\):

  $$\text{SQNR} = 20 \log_{10}\!\left(\frac{0.2}{0.0023}\right) = 20 \log_{10}(87.0) = 20 \times 1.939 = 38.8 \text{ dB}$$

  For int4 with \\(\sigma_{\text{noise}} = 0.038\\):

  $$\text{SQNR} = 20 \log_{10}\!\left(\frac{0.2}{0.038}\right) = 20 \log_{10}(5.26) = 20 \times 0.721 = 14.4 \text{ dB}$$

  For reference: audio engineers consider SNR below 20 dB noticeably degraded. In neural networks, 38 dB (int8) is typically comfortable â€” the noise is negligible relative to the signal. At 14 dB (int4), the noise is a significant fraction of the signal, and accuracy drops become likely. The ~24 dB gap between int8 and int4 corresponds to a ~16Ã— difference in noise power.

- **To train robustness.** QAT (Chapter 11) deliberately injects fake quantization noise during training so the model learns to produce outputs that survive it. The noise is treated as adversarial irritation, not as a learned signal.

The one-sentence version: *we observe quantization noise to decide how far we can push low precision before the model breaks â€” the model itself never sees or uses the noise as a feature.*

**When this model breaks.** The uniform noise assumption requires that values are not aligned with the grid. It holds well for weights (which are continuous-valued after training) and for activations in intermediate layers. It breaks in two cases:

1. **After QAT** (Chapter 11): weights cluster near grid points, so rounding errors are systematically near zero â€” the noise is no longer uniformly distributed.
2. **Across layers**: the rounding error at layer \\(k\\) becomes part of the input to layer \\(k+1\\). The error and the signal are no longer independent. The simple variance-addition model underestimates cumulative error in deep networks because it ignores correlation between layers.

The \\(S^2/12\\) model is a first-order estimate â€” useful for comparing quantization configurations (int8 vs int4, per-tensor vs per-channel) and for sanity-checking whether observed accuracy drops are consistent with expected noise levels. It is not a precise predictor for deep networks, where error propagation across dozens of layers introduces correlations that the independent-noise assumption cannot capture.

> **ðŸ“Š INSERT DIAGRAM: Error Propagation Through a Multi-Layer Pipeline**
>
> A vertical flow showing 4 sequential layers of a neural network:
>
> ```
> Layer 1 input (float) â”€â”€â†’ [Quantize] â”€â”€â†’ int8 Ã— int8 matmul â”€â”€â†’ int32 acc â”€â”€â†’ [Requant] â”€â”€â†’ int8 output
>     â”‚ rounding error Îµâ‚                                               â”‚ requant error Î´â‚
>     v                                                                v
> Layer 2 input = signal + Îµâ‚ + Î´â‚  â”€â”€â†’ [Quantize] â”€â”€â†’ ... â”€â”€â†’ [Requant] â”€â”€â†’ output
>     â”‚ now Îµâ‚ and Î´â‚ are part of the input â€” new error Îµâ‚‚ is added ON TOP of the old error
>     v
> Layer 3 input = signal + accumulated(Îµâ‚+Î´â‚+Îµâ‚‚+Î´â‚‚) â”€â”€â†’ ...
>     v
> Layer 4 input = signal + accumulated errors from all prior layers
> ```
>
> Key annotations:
> - At each layer boundary, show the error budget growing (e.g., Â±0.004 â†’ Â±0.012 â†’ Â±0.03 â†’ Â±0.06)
> - Highlight that errors from early layers get multiplied by weights of all subsequent layers
> - Show that early-layer errors affect MORE downstream layers than late-layer errors
> - Include a callout: "This is why early layers are more sensitive to quantization than late layers"

---

## Clipping Error

Clipping error occurs when a real value falls outside the representable range entirely. The quantization grid covers \\([r_{\min}, r_{\max}]\\). Any value beyond this range is clamped â€” forced to the nearest boundary value.

Using the same range [-1.0, 1.0]: the value 1.7 lies outside the grid. It is clamped to 1.0. The error is:

$$|1.7 - 1.0| = 0.7$$

Compare this to the maximum rounding error of 0.0039. Clipping error here is 180Ã— larger than the worst rounding error. And unlike rounding, clipping error is not bounded by the step size. A value at 10.0 clamped to 1.0 has an error of 9.0 â€” over 2,300Ã— the maximum rounding error. The further the value is from the range boundary, the worse the error.

Clipping error is caused by a mismatch between the quantization range and the actual values being quantized. If the range is set too narrow, values in the tails are crushed to the boundary. If the range is set wide enough to include all values, clipping error is zero â€” but step size increases, making rounding error worse everywhere.

This is the rangeâ€“precision trade-off from Chapter 2 manifesting as a concrete engineering problem: widen the range to avoid clipping, and you increase rounding error for every value inside the range. Narrow the range for better resolution, and you clip the tails.

*Accuracy pattern: Tail Clipping â€” sharp failures and outlier sensitivity; improves with calibration range adjustment or percentile observers.*

---

## Representation Error

Representation error is the most subtle of the three. It does not come from any single value being mapped incorrectly. It comes from a mismatch between the shape of the value distribution and the uniformity of the quantization grid.

The quantization grid is uniform â€” grid points are evenly spaced across the entire range. But value distributions are rarely uniform. In a typical neural network layer, the vast majority of activation values might cluster in a narrow region near zero, with only a few values extending into the tails.

Consider a layer where 95% of activations fall in [-0.1, 0.1] and the remaining 5% are spread across [-1.0, -0.1] and [0.1, 1.0]. Under int8 quantization of the full range [-1.0, 1.0]:

- The entire range gets 256 grid points with step size 0.00784.
- The dense region [-0.1, 0.1] spans 0.2 units of range. It receives approximately \\(0.2 / 0.00784 \approx 26\\) grid points.
- 95% of the values must be represented using 26 out of 256 levels. The remaining 230 grid points cover regions with almost no data.

No individual value is clipped. No individual rounding error exceeds \\(S/2\\). Yet the quantized representation is spending 90% of its budget on regions where almost nothing exists, and only 10% on the region where the data actually lives. If the model's behavior depends on fine distinctions within [-0.1, 0.1] â€” and it often does, since these are the values near zero where ReLU activations transition and small differences determine network behavior â€” then 26 levels may be too coarse.

Representation error is not about any one value being wrong. It is about the grid being poorly matched to the distribution it represents.

*Accuracy pattern: Distribution Mismatch â€” cannot be fixed by adjusting range alone; improves with per-channel scales, groupwise quantization, or non-uniform schemes.*

---

## The Three Errors Are Distinct

These three error types have different causes, different magnitudes, and different remedies:

| Error Type | Cause | Bounded? | Remedy Direction |
|---|---|---|---|
| Rounding | Value falls between grid points | Yes, by \\(S/2\\) | Use more bits (finer grid) |
| Clipping | Value falls outside the range | No â€” grows with distance | Widen the range (at the cost of resolution) |
| Representation | Uniform grid mismatches non-uniform distribution | Not directly | Adjust range or use non-uniform quantization |

When accuracy degrades after quantization, the diagnostic question is: which of these three is dominant? If clipping is the problem, widening the range helps â€” but it makes rounding and representation error worse. If representation mismatch is the problem, the grid itself is the wrong shape for the data, and adjusting the range within a uniform grid can only partially help.

Blaming "quantization" generically is not a diagnosis. Naming the specific error type is.

---

## Conceptual Consolidation

Quantization error is not one thing. It is three things â€” rounding, clipping, and representation mismatch â€” that coexist in every quantized layer with different magnitudes and different causes.

Whenever accuracy degrades in a quantized model, the first question is: is the problem rounding (step size too large), clipping (range too narrow), or representation (grid shape mismatched to data shape)? The answer determines what to fix and what fixing it will cost.

*In Appendix B, these three error types will appear in a single model â€” you'll compute each one numerically and see which layers are limited by rounding, which by clipping, and how that drives the overall accuracy drop.*

**Failure Signals**

- Sudden saturation at boundary values
- Layer outputs become noisy after quantization
- Accuracy drops without obvious architecture changes


