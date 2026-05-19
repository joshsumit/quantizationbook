# Chapter 3: Scale and Zero-Point

In this chapter, we quantize weights and activations by defining their shared mapping parameters.

> **Notation used in this chapter:** \\( r \\) denotes a real floating-point value; \\(q\\) denotes a quantized integer; \\(S\\) (scale) is the real-valued width of one quantization step; \\(Z\\) (zero-point) is the integer that represents real zero; \\( q_{\min}\\) and \\(q_{\max}\\) are the integer endpoints (e.g., 0 and 255 for uint8, or -128 and 127 for int8).

## The Mapping Problem

A quantization grid provides a fixed number of integer levels spanning a target range. However, given a continuous range of real values—for example, activations distributed between -2.3 and 5.1—and a discrete budget of 256 integers (0 through 255), we must establish a deterministic rule to map real values to specific integers.

To solve this mapping problem, two parameters must be determined. First, we define the width of each step on the grid, which controls the resolution and spacing between representable values. Second, we determine where the grid sits relative to real zero, which ensures that floating-point zero maps precisely to an integer value.

These two quantities are called *scale* and *zero-point*.
(Scale is the real value represented by one integer step.)
Together, they fully define the affine mapping between real numbers and quantized integers. Once calculated during calibration, these parameters are fixed; every subsequent hardware instruction in the quantized model depends on their exact values.

---

## Scale

Scale (\\(S\\)) is the real-valued width of one quantization step — the distance between two adjacent representable values. It is the step size from Chapter 2 made precise.

Given a real-valued range \\([r_{\min}, r_{\max}]\\) and an integer range \\([q_{\min}, q_{\max}]\\):

To see exactly how this mapping works, we quantify it:

$$S = \frac{r_{\max} - r_{\min}}{q_{\max} - q_{\min}}$$

For activations in [-2.3, 5.1] mapped to uint8 [0, 255]:

$$S = \frac{5.1 - (-2.3)}{255 - 0} = \frac{7.4}{255} \approx 0.02902$$

Each integer step corresponds to a real-valued increment of 0.02902. A smaller scale means finer resolution but a narrower range. A larger scale means coarser resolution but a wider range. This is the range–precision trade-off from Chapter 2, now expressed as a single number.

---

## Zero-Point

The scale determines spacing, but not position. A grid with step size 0.02902 could start anywhere on the real number line. Zero-point (\\(Z\\)) anchors the grid by specifying which integer represents real zero.

$$Z = \text{round}\!\left(\frac{-r_{\min}}{S}\right) = \text{round}\!\left(\frac{2.3}{0.02902}\right) = \text{round}(79.26) = 79$$

We use round-to-nearest (rather than floor or ceiling functions) to minimize representation error around zero. Because 79.26 is closer to 79 than to 80, selecting 79 yields the lowest possible quantization error for zero. Consequently, the integer 79 represents real zero, integers below 79 represent negative real values, and integers above 79 represent positive real values, fully positioning the grid.

---

## The Full Mapping

With \\(S\\) and \\(Z\\) determined, the mapping from real values to integers and back is:

**Quantize** (real → integer):

$$q = \text{clamp}\!\left(q_{\min},\; q_{\max},\; \text{round}\!\left(\frac{r}{S}\right) + Z\right)$$

The \\(\text{clamp}(a, b, x)\\) function means: if \\(x < a\\), return \\(a\\); if \\(x > b\\), return \\(b\\); otherwise return \\(x\\). It ensures values outside the calibrated range saturate to the nearest endpoint rather than overflowing. All examples in this chapter use round-to-nearest (ties to even), the default rounding mode in most quantization toolkits.

**Dequantize** (integer → real):

$$r = S \cdot (q - Z)$$

Walking through concrete values for the range [-2.3, 5.1] with \\(S = 0.02902\\) and \\(Z = 79\\):

**Real value 0.0:**
$$q = \text{round}(0.0 / 0.02902) + 79 = 0 + 79 = 79$$

Dequantized: \\(0.02902 \times (79 - 79) = 0.0\\). Zero maps to integer 79 and reconstructs exactly. No error.

**Real value 3.7:**
$$q = \text{round}(3.7 / 0.02902) + 79 = \text{round}(127.5) + 79 = 128 + 79 = 207$$

Dequantized: \\(0.02902 \times (207 - 79) = 0.02902 \times 128 = 3.715\\). The original value was 3.7; the reconstructed value is 3.715. The difference — 0.015 — is the rounding error for this specific value, well within one step size of 0.029.

**Real value -2.3:**
$$q = \text{round}(-2.3 / 0.02902) + 79 = \text{round}(-79.26) + 79 = -79 + 79 = 0$$

Dequantized: \\(0.02902 \times (0 - 79) = -2.292\\). The boundary value reconstructs with a small rounding error of 0.008.

---

## Invariant: Zero Must Be Exactly Representable

In the previous example, real 0.0 mapped perfectly to integer 79 and dequantized back to exactly 0.0. This exact mapping is a strict mathematical requirement rather than a statistical coincidence. If real zero does not map to an exact integer, systematic offsets introduce errors throughout the computation graph.

### Accumulation of Systematic Offsets
Consider multiplying an activation tensor by a weight tensor where the weight should be zero. In standard float32, this operation evaluates to exactly zero. However, if the quantized representation of zero shifts to a non-zero value (e.g., 0.003 because zero falls between two discrete grid points), every zeroed multiplication injects a minor offset. Across thousands of MAC (Multiply-Accumulate) operations within a single dense layer, these small residual errors compound into significant noise.

### Shift in Non-Linear Boundaries
Non-linear activation functions like ReLU depend on sharp zero boundaries, outputting zero for negative inputs and passing positive inputs through unchanged. If quantized zero carries an error, the boundary between an active and inactive neuron shifts. This shift causes inactive neurons to leak signals, while weakly active neurons are suppressed prematurely.

### Bias Term Drift
Bias values are added directly to accumulation registers. If the zero-point representation introduces a systematic error, every bias addition injects an unintended constant offset. Across deep architectures, these layer-by-layer offsets compound, degrading model accuracy. Exact-zero mapping serves as a fundamental correctness invariant for affine quantization; without it, every operation touching zero would require explicit, hardware-inefficient compensation terms.

---

## Symmetric vs Asymmetric Quantization

The choice of zero-point creates two fundamentally different quantization modes.

### Asymmetric Quantization

This is what we have been computing. The range is \\([r_{\min}, r_{\max}]\\), and \\(Z\\) is chosen so that zero maps to an integer within \\([q_{\min}, q_{\max}]\\). The grid matches the actual data range.

For our example: range [-2.3, 5.1], \\(Z = 79\\). All 256 levels span the observed data range. No representational budget is wasted.

### Symmetric Quantization

Symmetric quantization forces \\(Z = 0\\) and centers the range around zero: \\([-\alpha, +\alpha]\\), where \\(\alpha = \max(|r_{\min}|, |r_{\max}|)\\).

For the same data: \\(\alpha = \max(2.3, 5.1) = 5.1\\). The range becomes [-5.1, 5.1], mapped to signed int8 [-128, 127].

$$S = \frac{5.1}{127} \approx 0.04016$$

The step size is 0.04016, compared to 0.02902 under asymmetric. Resolution is 38% coarser. And the range below -2.3 — from -5.1 to -2.3 — contains no actual data. Roughly 27% of the grid points represent values that never occur.

**Concrete comparison — mapping three values through both schemes:**

| Real value | Asymmetric (\\(S\\) = 0.02902, \\(Z\\) = 79) | Symmetric (\\(S\\) = 0.04016, \\(Z\\) = 0) |
|---|---|---|
| 0.5 | \\(q = 96\\), \\(\hat{r} = 0.493\\), error = 0.007 | \\(q = 12\\), \\(\hat{r} = 0.482\\), error = 0.018 |
| 1.5 | \\(q = 131\\), \\(\hat{r} = 1.509\\), error = 0.009 | \\(q = 37\\), \\(\hat{r} = 1.486\\), error = 0.014 |
| 3.0 | \\(q = 182\\), \\(\hat{r} = 2.989\\), error = 0.011 | \\(q = 75\\), \\(\hat{r} = 3.012\\), error = 0.012 |

Asymmetric quantization produces roughly half the error on these representative values. Symmetric wastes 27% of its codes on a range with no data, and the remaining 73% of codes must cover the same region that asymmetric covers with 100% of its codes.

So why would anyone choose symmetric? Because \\(Z = 0\\) removes entire correction terms from the computation.

When two quantized values are multiplied and accumulated — the core operation in every linear layer and convolution — the full integer arithmetic expands as follows. Let \\(q_w = \text{round}(w/S_w) + Z_w\\) and \\(q_x = \text{round}(x/S_x) + Z_x\\). The real-valued dot product \\(\sum w_i x_i\\) becomes:

$$S_w S_x \left[\sum q_{w_i} q_{x_i} - Z_x \sum q_{w_i} - Z_w \sum q_{x_i} + N \cdot Z_w Z_x\right]$$

The integer dot product is only the first term; zero-points introduce three additional correction terms. The first term — \\(\sum q_{w_i} q_{x_i}\\) — is the integer dot product that hardware accelerates. The remaining three are *correction terms* introduced by the zero-points:

- \\(Z_x \sum q_{w_i}\\): a vector-scalar multiplication over all quantized weights, required for every output element.
- \\(Z_w \sum q_{x_i}\\): a vector-scalar multiplication over all quantized inputs.
- \\(N \cdot Z_w Z_x\\): a constant offset per output element.

These corrections add two extra vector reductions and a scalar add to *every single matrix multiply in the model*. For a [4096 × 4096] weight matrix, each output row requires summing 4096 quantized weights and 4096 quantized inputs — per output element, per layer, per inference.

### Correction Terms in Practice
Consider a small-scale dot product where \\(N = 3\\), utilizing weights \\(q_w = [8, -6, 3]\\) with \\(Z_w = 2\\), and inputs \\(q_x = [4, 7, -5]\\) with \\(Z_x = 1\\). The calculation resolves as follows:

* **Integer Dot Product (Term 1):** \\(8 \times 4 + (-6) \times 7 + 3 \times (-5) = 32 - 42 - 15 = -25\\)
* **Correction 1:** \\(-Z_x \sum q_{w_i} = -1 \times (8 + (-6) + 3) = -1 \times 5 = -5\\)
* **Correction 2:** \\(-Z_w \sum q_{x_i} = -2 \times (4 + 7 + (-5)) = -2 \times 6 = -12\\)
* **Correction 3:** \\(N \cdot Z_w \cdot Z_x = 3 \times 2 \times 1 = 6\\)
* **Full Resolution:** \\(-25 - 5 - 12 + 6 = -36\\)

Omitting the zero-point correction terms yields -25 instead of the mathematically correct value of -36, introducing an error of approximately 31% on a single output element. At production scale (\\(N = 4096\\)), managing these corrections becomes a critical performance consideration.

While the weight reduction term (\\(\sum q_{w_i}\\)) depends entirely on static parameters and can be precomputed once during model compilation, the activation reduction term (\\(\sum q_{x_i}\\)) depends on dynamic model inputs and must be evaluated at runtime for every inference pass.

When both zero-points are forced to zero (\\(Z_w = 0\\) and \\(Z_x = 0\\)), all three correction terms vanish. The entire equation collapses to a single optimized operation:

$$S_w S_x \sum q_{w_i} q_{x_i}$$

By removing the correction overhead, the underlying hardware can execute a single fused multiply-accumulate instruction per value pair, leveraging the maximized throughput of integer execution pipelines. Hardware preference for symmetric layouts stems directly from this structural efficiency rather than conceptual simplicity.

Asymmetric quantization captures the dynamic range more precisely but introduces three arithmetic correction terms to every multiply-accumulate operation. Conversely, symmetric quantization sacrifices a portion of the representational range but eliminates these computational corrections entirely. In production backends, the correction overhead in a 24-layer transformer utilizing fully asymmetric quantization on both weights and activations can account for a significant percentage of the total int8 execution time. The precise performance penalty depends heavily on the specific kernel implementation (such as FBGEMM, QNNPACK, or TensorRT) and how aggressively the static weight-sums are precomputed during model compilation.

### The Practical Convention: Symmetric Weights, Asymmetric Activations

In most deployed systems, the choice is not "symmetric everywhere" or "asymmetric everywhere." The standard recipe is a **hybrid**: symmetric quantization for weights, asymmetric for activations. This is the default in PyTorch's quantization stack, in TensorRT, and in most mobile backends. Understanding why requires examining each side separately.

**Weights → Symmetric.** Neural network weights are typically distributed roughly symmetrically around zero — a bell curve centered at (or very near) 0.0. Symmetric quantization fits this distribution naturally without wasting codes. More importantly, setting \\(Z_w = 0\\) eliminates two of the three correction terms in the dot-product expansion above: the \\(Z_w \sum q_{x_i}\\) term (which depends on the input and must be computed at runtime) and the \\(N \cdot Z_w Z_x\\) constant both vanish. Only the \\(Z_x \sum q_{w_i}\\) term remains — and \\(\sum q_{w_i}\\) depends only on weights, so it can be precomputed once during model preparation and stored as a constant. This makes the per-inference cost of the remaining correction nearly free.

**Activations → Asymmetric.** Activations after ReLU are strictly non-negative: their range is \\([0, r_{\max}]\\). If you apply symmetric quantization, the range becomes \\([-r_{\max}, +r_{\max}]\\), and *half* of your 256 int8 codes represent negative values that can never occur — you are left with effectively ~128 usable levels instead of 256. Asymmetric quantization maps the zero-point to code 0 (since the minimum is 0.0), giving you all 256 codes for the actual \\([0, r_{\max}]\\) range. For non-ReLU activations (GELU, SiLU, etc.) the distribution may extend below zero, in which case the advantage is smaller — but even there, activations are rarely perfectly symmetric, so asymmetric usually fits better.

**The net effect.** With \\(Z_w = 0\\) (weights symmetric) and \\(Z_x \neq 0\\) (activations asymmetric), the dot-product formula becomes:

$$S_w S_x \left[\sum q_{w_i} q_{x_i} - Z_x \sum q_{w_i}\right]$$

One correction term instead of three. And \\(\sum q_{w_i}\\) is a precomputed constant per output channel — so the runtime cost is one extra addition per output element, which hardware handles trivially.

This is why Appendix B uses exactly this setup: asymmetric uint8 for activations, symmetric int8 for weights. It is not an arbitrary choice — it is the best trade-off between representational precision (asymmetric activations waste no codes) and compute efficiency (symmetric weights eliminate runtime correction overhead).

---

## Scale and Zero-Point Are Contracts

Once scale and zero-point are determined, they are fixed. They are compiled into the quantized model alongside the quantized weights. Every subsequent layer, every accumulation, every output interpretation depends on these exact values.

In static quantization, they cannot be adjusted at inference time — they cannot be tuned per-input. (Dynamic quantization, covered later, recomputes activation scales per input or per batch, but weight scales remain fixed even there.) If the scale is wrong — if it was computed from unrepresentative data, or if the true range of values extends beyond what the scale covers — every inference will carry that error. Their permanence is the point: scale and zero-point are binding contracts between the quantization scheme and every computation that follows.

---

## Conceptual Consolidation

Scale controls resolution. Zero-point anchors the grid to real zero. Together they define the complete mapping between continuous values and the finite set of integers that a quantized model operates over.

Whenever you encounter a quantized layer, two questions should be immediate: What is the scale — and is it fine enough to preserve the distinctions this layer depends on? What is the zero-point — and does zero map exactly?

This chapter defined the mapping for one tensor with one scale and one zero-point. In practice, different output channels can have very different value distributions, and a single scale is a blunt instrument. Per-channel scales and groupwise quantization — covered in later chapters — address this by giving each channel or group its own contract.

*These scale and zero-point computations will be composed into a full end-to-end pipeline in Appendix B, where you'll compute them for every layer of a small model and trace the errors all the way to the output.*
