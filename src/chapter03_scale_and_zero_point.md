# Chapter 3: Scale and Zero-Point

> **Notation used in this chapter:** $r$ denotes a real floating-point value; $q$ denotes a quantized integer; $S$ (scale) is the real-valued width of one quantization step; $Z$ (zero-point) is the integer that represents real zero; $$q_{\min}$$ and $$q_{\max}$$ are the integer endpoints (e.g., 0 and 255 for uint8, or -128 and 127 for int8).

## The Mapping Problem

A quantization grid provides a fixed number of integer levels spanning a range. But a question remains unanswered. Given a range of real values — say, activations that fall between -2.3 and 5.1 — and 256 integers (0 through 255), how do you decide which integer represents which real value?

Two things must be determined. First, how wide each step on the grid is — this controls the spacing between representable values. Second, where the grid sits relative to real zero — this controls which real value maps to which integer.

These two quantities are called *scale* and *zero-point*. Together they fully define the mapping between real numbers and quantized integers. Once set, they become permanent. Every subsequent computation in the quantized model depends on their exact values.

---

## Scale

Scale ($S$) is the real-valued width of one quantization step — the distance between two adjacent representable values. It is the step size from Chapter 2 made precise.

Given a real-valued range $[r_{\min}, r_{\max}]$ and an integer range $[q_{\min}, q_{\max}]$:

$$S = \frac{r_{\max} - r_{\min}}{q_{\max} - q_{\min}}$$

For activations in [-2.3, 5.1] mapped to uint8 [0, 255]:

$$S = \frac{5.1 - (-2.3)}{255 - 0} = \frac{7.4}{255} \approx 0.02902$$

Each integer step corresponds to a real-valued increment of 0.02902. A smaller scale means finer resolution but a narrower range. A larger scale means coarser resolution but a wider range. This is the range–precision trade-off from Chapter 2, now expressed as a single number.

---

## Zero-Point

The scale determines spacing, but not position. A grid with step size 0.02902 could start anywhere on the real number line. Zero-point ($Z$) anchors the grid by specifying which integer represents real zero.

$$Z = \text{round}\!\left(\frac{-r_{\min}}{S}\right) = \text{round}\!\left(\frac{2.3}{0.02902}\right) = \text{round}(79.26) = 79$$

We use round-to-nearest (not floor or ceil) because it minimizes the error in representing zero: 79.26 is closer to 79 than to 80, so rounding to 79 gives the smallest zero-point error.

Integer 79 represents real zero. Integers below 79 represent negative real values. Integers above 79 represent positive real values. The grid is now positioned.

---

## The Full Mapping

With $S$ and $Z$ determined, the mapping from real values to integers and back is:

**Quantize** (real → integer):

$$q = \text{clamp}\!\left(q_{\min},\; q_{\max},\; \text{round}\!\left(\frac{r}{S}\right) + Z\right)$$

The $\text{clamp}(a, b, x)$ function means: if $x < a$, return $a$; if $x > b$, return $b$; otherwise return $x$. It ensures values outside the calibrated range saturate to the nearest endpoint rather than overflowing. All examples in this chapter use round-to-nearest (ties to even), the default rounding mode in most quantization toolkits.

**Dequantize** (integer → real):

$$r = S \cdot (q - Z)$$

Walking through concrete values for the range [-2.3, 5.1] with $S = 0.02902$ and $Z = 79$:

**Real value 0.0:**
$$q = \text{round}(0.0 / 0.02902) + 79 = 0 + 79 = 79$$

Dequantized: $0.02902 \times (79 - 79) = 0.0$. Zero maps to integer 79 and reconstructs exactly. No error.

**Real value 3.7:**
$$q = \text{round}(3.7 / 0.02902) + 79 = \text{round}(127.5) + 79 = 128 + 79 = 207$$

Dequantized: $0.02902 \times (207 - 79) = 0.02902 \times 128 = 3.715$. The original value was 3.7; the reconstructed value is 3.715. The difference — 0.015 — is the rounding error for this specific value, well within one step size of 0.029.

**Real value -2.3:**
$$q = \text{round}(-2.3 / 0.02902) + 79 = \text{round}(-79.26) + 79 = -79 + 79 = 0$$

Dequantized: $0.02902 \times (0 - 79) = -2.292$. The boundary value reconstructs with a small rounding error of 0.008.

---

## Invariant: Zero Must Be Exactly Representable

In the example above, real 0.0 mapped to integer 79 and dequantized back to exactly 0.0. This is not a coincidence — it is a requirement.

If real zero does not map to an exact integer, systematic offsets appear throughout the computation:

**Multiplication by zero produces nonzero results.** Consider multiplying an activation by a weight that should be zero. In float32, the result is exactly zero. If the quantized representation of zero is actually 0.003 (because zero falls between two grid points), every "zero" multiplication produces 0.003 instead. Across thousands of such operations in a single layer, these phantom values accumulate.

**ReLU boundaries shift.** ReLU outputs zero for negative inputs and passes positive inputs through. If quantized zero is not exact, the boundary between "active" and "inactive" shifts by the zero-point error. Neurons that should be inactive leak signal. Neurons that should be active are suppressed.

**Bias terms drift.** Bias values are added directly to accumulations. If the zero representation carries error, every bias addition injects a small offset. Over many layers, these offsets compound.

Exact-zero mapping is a correctness invariant for affine quantization. Without it, every operation that touches zero — and in a typical network, that is most of them — requires explicit compensation terms.

---

## Symmetric vs Asymmetric Quantization

The choice of zero-point creates two fundamentally different quantization modes.

### Asymmetric Quantization

This is what we have been computing. The range is $[r_{\min}, r_{\max}]$, and $Z$ is chosen so that zero maps to an integer within $[q_{\min}, q_{\max}]$. The grid matches the actual data range.

For our example: range [-2.3, 5.1], $Z = 79$. All 256 levels span the observed data range. No representational budget is wasted.

### Symmetric Quantization

Symmetric quantization forces $Z = 0$ and centers the range around zero: $[-\alpha, +\alpha]$, where $\alpha = \max(|r_{\min}|, |r_{\max}|)$.

For the same data: $\alpha = \max(2.3, 5.1) = 5.1$. The range becomes [-5.1, 5.1], mapped to signed int8 [-128, 127].

$$S = \frac{5.1}{127} \approx 0.04016$$

The step size is 0.04016, compared to 0.02902 under asymmetric. Resolution is 38% coarser. And the range below -2.3 — from -5.1 to -2.3 — contains no actual data. Roughly 27% of the grid points represent values that never occur.

**Concrete comparison — mapping three values through both schemes:**

| Real value | Asymmetric ($S$ = 0.02902, $Z$ = 79) | Symmetric ($S$ = 0.04016, $Z$ = 0) |
|---|---|---|
| 0.5 | $q = 96$, $\hat{r} = 0.493$, error = 0.007 | $q = 12$, $\hat{r} = 0.482$, error = 0.018 |
| 1.5 | $q = 131$, $\hat{r} = 1.509$, error = 0.009 | $q = 37$, $\hat{r} = 1.486$, error = 0.014 |
| 3.0 | $q = 182$, $\hat{r} = 2.989$, error = 0.011 | $q = 75$, $\hat{r} = 3.012$, error = 0.012 |

Asymmetric quantization produces roughly half the error on these representative values. Symmetric wastes 27% of its codes on a range with no data, and the remaining 73% of codes must cover the same region that asymmetric covers with 100% of its codes.

So why would anyone choose symmetric? Because $Z = 0$ removes entire correction terms from the computation.

When two quantized values are multiplied and accumulated — the core operation in every linear layer and convolution — the full integer arithmetic expands as follows. Let $q_w = \text{round}(w/S_w) + Z_w$ and $q_x = \text{round}(x/S_x) + Z_x$. The real-valued dot product $\sum w_i x_i$ becomes:

$$S_w S_x \left[\sum q_{w_i} q_{x_i} - Z_x \sum q_{w_i} - Z_w \sum q_{x_i} + N \cdot Z_w Z_x\right]$$

The integer dot product is only the first term; zero-points introduce three additional correction terms. The first term — $\sum q_{w_i} q_{x_i}$ — is the integer dot product that hardware accelerates. The remaining three are *correction terms* introduced by the zero-points:

- $Z_x \sum q_{w_i}$: a vector-scalar multiplication over all quantized weights, required for every output element.
- $Z_w \sum q_{x_i}$: a vector-scalar multiplication over all quantized inputs.
- $N \cdot Z_w Z_x$: a constant offset per output element.

These corrections add two extra vector reductions and a scalar add to *every single matrix multiply in the model*. For a [4096 × 4096] weight matrix, each output row requires summing 4096 quantized weights and 4096 quantized inputs — per output element, per layer, per inference.

**Worked example: the correction terms in practice.** Suppose a tiny dot product with $N = 3$: weights $q_w = [8, -6, 3]$ with $Z_w = 2$, inputs $q_x = [4, 7, -5]$ with $Z_x = 1$.

- The integer dot product (first term): $8 \times 4 + (-6) \times 7 + 3 \times (-5) = 32 - 42 - 15 = -25$
- Correction 1: $-Z_x \sum q_{w_i} = -1 \times (8 + (-6) + 3) = -1 \times 5 = -5$
- Correction 2: $-Z_w \sum q_{x_i} = -2 \times (4 + 7 + (-5)) = -2 \times 6 = -12$
- Correction 3: $N \cdot Z_w \cdot Z_x = 3 \times 2 \times 1 = 6$
- Full result: $-25 - 5 - 12 + 6 = -36$

The integer dot product alone gives $-25$. The true result (after corrections) is $-36$. Getting it wrong by omitting corrections means a 31% error — on every output element. Now imagine this at $N = 4096$: the three correction terms involve a 4096-element weight sum, a 4096-element input sum, and a scalar, all computed per output element.

One partial mitigation: $\sum q_{w_i}$ depends only on weights and can be precomputed once per output channel during model preparation. But $\sum q_{x_i}$ depends on the input and must be computed at runtime for every inference.

When both zero-points are zero ($Z_w = 0$, $Z_x = 0$), all three correction terms vanish. The entire operation collapses to:

$$S_w S_x \sum q_{w_i} q_{x_i}$$

A single integer dot product. No corrections. The hardware executes one fused multiply-accumulate instruction per pair of values — exactly what the int8 datapath in Chapter 4 was built for.

Asymmetric quantization captures the range more precisely but adds three correction terms to every multiply-accumulate operation. Symmetric quantization wastes some range but eliminates those corrections entirely. In some backends, the correction overhead in a 24-layer transformer with asymmetric quantization on both weights and activations can account for a meaningful fraction of total int8 compute time — the exact cost depends on the kernel implementation (FBGEMM, QNNPACK, TensorRT, etc.) and how aggressively weight-sums are precomputed.

Hardware prefers symmetric quantization not because it is conceptually simpler, but because it is structurally cheaper to execute.

### The Practical Convention: Symmetric Weights, Asymmetric Activations

In most deployed systems, the choice is not "symmetric everywhere" or "asymmetric everywhere." The standard recipe is a **hybrid**: symmetric quantization for weights, asymmetric for activations. This is the default in PyTorch's quantization stack, in TensorRT, and in most mobile backends. Understanding why requires examining each side separately.

**Weights → Symmetric.** Neural network weights are typically distributed roughly symmetrically around zero — a bell curve centered at (or very near) 0.0. Symmetric quantization fits this distribution naturally without wasting codes. More importantly, setting $Z_w = 0$ eliminates two of the three correction terms in the dot-product expansion above: the $Z_w \sum q_{x_i}$ term (which depends on the input and must be computed at runtime) and the $N \cdot Z_w Z_x$ constant both vanish. Only the $Z_x \sum q_{w_i}$ term remains — and $\sum q_{w_i}$ depends only on weights, so it can be precomputed once during model preparation and stored as a constant. This makes the per-inference cost of the remaining correction nearly free.

**Activations → Asymmetric.** Activations after ReLU are strictly non-negative: their range is $[0, r_{\max}]$. If you apply symmetric quantization, the range becomes $[-r_{\max}, +r_{\max}]$, and *half* of your 256 int8 codes represent negative values that can never occur — you are left with effectively ~128 usable levels instead of 256. Asymmetric quantization maps the zero-point to code 0 (since the minimum is 0.0), giving you all 256 codes for the actual $[0, r_{\max}]$ range. For non-ReLU activations (GELU, SiLU, etc.) the distribution may extend below zero, in which case the advantage is smaller — but even there, activations are rarely perfectly symmetric, so asymmetric usually fits better.

**The net effect.** With $Z_w = 0$ (weights symmetric) and $Z_x \neq 0$ (activations asymmetric), the dot-product formula becomes:

$$S_w S_x \left[\sum q_{w_i} q_{x_i} - Z_x \sum q_{w_i}\right]$$

One correction term instead of three. And $\sum q_{w_i}$ is a precomputed constant per output channel — so the runtime cost is one extra addition per output element, which hardware handles trivially.

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
