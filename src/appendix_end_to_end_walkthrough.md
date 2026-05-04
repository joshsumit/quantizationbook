# Appendix B: End-to-End Numeric Walkthrough

## From Float to Quantized: One Model, All the Way Through

**If you have read Chapters 1–19, you already have every concept needed to follow this walkthrough.** This appendix is the test: if you can predict the output at each step before reading the answer, you have internalized the material. If a step surprises you, the chapter reference tells you exactly where to revisit.

This appendix takes a small model through the entire quantization pipeline — calibration, scale computation, boundary counting, error prediction, and accuracy comparison — composing every concept from the book into a single worked example.

---

## The Model

A 3-layer feedforward classifier:

- **Input:** 4 features (float32)
- **Layer 1:** Linear(4 → 8) + ReLU
- **Layer 2:** Linear(8 → 8) + ReLU
- **Layer 3:** Linear(8 → 2) (output logits)

Total parameters: \\((4 \times 8 + 8) + (8 \times 8 + 8) + (8 \times 2 + 2) = 40 + 72 + 18 = 130\\) weights and biases.

Float32 model size: \\(130 \times 4 = 520\\) bytes.
Int8 model size (weights only): \\(130 \times 1 + \text{scales} \approx 140\\) bytes.

(We ignore scale/zero-point metadata and packing overhead for simplicity. In this toy model, metadata is a handful of bytes; in real models it is a small fraction of weight data — see Chapter 16.)

---

## Step 1: Calibrate

Run 64 calibration inputs through the float32 model. At each quantization boundary, an observer records the activation range.

| Boundary | Location | Observed Min | Observed Max | Range |
|---|---|---|---|---|
| B1 | After Layer 1 + ReLU | 0.0 | 3.42 | 3.42 |
| B2 | After Layer 2 + ReLU | 0.0 | 5.17 | 5.17 |
| B3 | After Layer 3 (logits) | -2.83 | 4.61 | 7.44 |

Weight ranges (computed directly, no calibration needed):

*Output artifact: observer table with per-boundary activation ranges.*

| Layer | Weight Min | Weight Max | Range |
|---|---|---|---|
| W1 | -0.62 | 0.71 | 1.33 |
| W2 | -0.48 | 0.53 | 1.01 |
| W3 | -0.81 | 0.77 | 1.58 |

---

## Step 2: Compute Scales and Zero-Points

Using asymmetric uint8 quantization for activations (post-ReLU values are non-negative) and symmetric int8 for weights:

**Activation scales:**

| Boundary | \\(S_{\text{act}}\\) | \\(Z_{\text{act}}\\) |
|---|---|---|
| B1 | \\(3.42 / 255 = 0.01341\\) | 0 (min is 0.0) |
| B2 | \\(5.17 / 255 = 0.02027\\) | 0 |
| B3 | \\(7.44 / 255 = 0.02918\\) | \\(\text{round}(2.83 / 0.02918) = 97\\) |

**Weight scales (symmetric, \\(Z = 0\\)):**

| Layer | \\(S_w\\) | \\(S_{\text{acc}} = S_w \times S_{\text{input}}\\) |
|---|---|---|
| W1 | \\(0.71 / 127 = 0.005591\\) | \\(0.005591 \times 0.01 = 0.00005591\\) (input scale assumed 0.01) |
| W2 | \\(0.53 / 127 = 0.004173\\) | \\(0.004173 \times 0.01341 = 0.00005598\\) |
| W3 | \\(0.81 / 127 = 0.006378\\) | \\(0.006378 \times 0.02027 = 0.0001293\\) |

*Output artifact: quantization parameter table \\((S, Z)\\) for every activation boundary and weight layer.*

---

## Step 3: Count Boundaries

Trace the data flow through the quantized graph:

1. Input quantized to int8 → **Boundary 0** (input quantization)
2. Layer 1: int8 × int8 → int32 accumulator. Bias added in int32. ReLU in int32. → **Boundary 1**: int32 → int8
3. Layer 2: int8 × int8 → int32 accumulator. Bias added in int32. ReLU in int32. → **Boundary 2**: int32 → int8
4. Layer 3: int8 × int8 → int32 accumulator. Bias added in int32. → **Boundary 3**: int32 → int8 (output)

**Total boundaries: 4** (including input quantization).

Each boundary introduces rounding error (round-to-nearest throughout). With Conv-ReLU fusion (Chapter 8), the ReLU does not add a separate boundary — it is applied within the int32 domain before the requantization step (typically fused into the requant epilogue).

*Output artifact: a boundary ledger listing B0–B3 with their locations and scales.*

---

## Step 4: Predict Error

Using the noise model from Chapter 5 (\\(\text{Var}(\epsilon) = S^2/12\\)):

**Per-boundary rounding noise variance:**

| Boundary | Scale \\(S\\) | \\(\text{Var}(\epsilon)\\) | \\(\sigma\\) |
|---|---|---|---|
| B0 (input) | assume 0.01 | \\(8.33 \times 10^{-6}\\) | 0.0029 |
| B1 | 0.01341 | \\(1.50 \times 10^{-5}\\) | 0.0039 |
| B2 | 0.02027 | \\(3.42 \times 10^{-5}\\) | 0.0059 |
| B3 | 0.02918 | \\(7.10 \times 10^{-5}\\) | 0.0084 |

**Layer 2 output noise estimate:**

Layer 2 has a dot product of width \\(N = 8\\). The output noise from weight quantization alone:

$$\text{Var}(\epsilon_{\text{L2}}) \approx N \times \frac{S_w^2}{12} \times \mathbb{E}[x^2]$$

With \\(S_w = 0.004173\\), \\(N = 8\\), and \\(\mathbb{E}[x^2] \approx 1.0\\) (typical for post-ReLU activations in this range):

$$\text{Var}(\epsilon_{\text{L2}}) \approx 8 \times \frac{0.004173^2}{12} \times 1.0 = 8 \times 1.45 \times 10^{-6} = 1.16 \times 10^{-5}$$

This noise then passes through Boundary 2's requantization, adding its own \\(\text{Var} = 3.42 \times 10^{-5}\\).

**Cumulative noise at the output (rough estimate):**

The noise from each boundary compounds through subsequent layers. A conservative upper bound: sum the variances at each boundary (ignoring amplification by weights, which can increase or decrease the noise depending on weight magnitudes):

$$\text{Var}_{\text{total}} \approx 8.33 \times 10^{-6} + 1.50 \times 10^{-5} + 3.42 \times 10^{-5} + 7.10 \times 10^{-5} \approx 1.29 \times 10^{-4}$$

$$\sigma_{\text{total}} \approx 0.0114$$

For output logits with a typical range of [-2.83, 4.61] (7.44 units), a noise standard deviation of 0.011 is roughly 0.15% of the range. This is small — a classification task should tolerate this level of noise without significant accuracy loss.

*Output artifact: variance ledger per boundary + cumulative \\(\sigma_{\text{total}}\\).*

---

## Step 5: Compare Prediction with Reality

Run the same 64 calibration inputs through both the float32 and int8 models. Measure the actual output differences:

| Metric | Predicted | Observed (typical) |
|---|---|---|
| Output noise \\(\sigma\\) | 0.011 | 0.008 – 0.015 |
| Max output deviation | ~3σ ≈ 0.034 | 0.02 – 0.04 |
| Classification accuracy drop | < 0.5% | 0.1 – 0.3% |

The prediction and observation are consistent. The noise model provides the right order of magnitude. The small discrepancy arises because:

- The independent-noise assumption slightly overestimates (errors partially cancel across the dot product due to sign mixing).
- Weight quantization error interacts with activation quantization error in ways the simple additive model does not capture.
- ReLU clips negative noise, which can slightly reduce noise variance after activation.

**Sanity checks:**
- All zero-points are within [0, 255]: \\(z_{B1}=0\\), \\(z_{B2}=0\\), \\(z_{B3}=97\\). ✓
- Clamp rate during calibration is ~0% (no values outside observed range by definition). ✓
- Predicted \\(\sigma\\) (0.011) roughly matches observed range (0.008–0.015). ✓

*Output artifact: measured diff table + sanity check confirmation.*

---

## Step 6: What If It Had Failed?

Suppose this model had an outlier problem. If Layer 1's activations had a single channel reaching 50.0 (instead of 3.42), the min-max scale would be:

$$S_{\text{B1}} = 50.0 / 255 = 0.196$$

The noise variance at B1 jumps to \\(0.196^2 / 12 = 0.0032\\) — a 213× increase. The output noise \\(\sigma\\) would rise from 0.011 to approximately 0.057, and the classification accuracy drop would be 3–5%.

**Diagnosis using Chapter 13's order:**
1. Graph boundaries: 4, as expected. ✓
2. Silent fallbacks: none (all ops are linear + ReLU). ✓
3. Scale alignment: no elementwise ops, no mismatch. ✓
4. Calibration: check the 99th-percentile-to-max ratio. Here: \\(3.1 / 50.0 = 0.062\\). Ratio is 16:1 — well above the 10:1 outlier explosion threshold. **Diagnosis: Outlier Explosion (Pattern 1).**
5. Fix: switch to percentile observer. Set range to [-0.0, 3.1]. The outlier clips, but 99.99% of values get full resolution.

*Canonical category: Distribution Mismatch / Budget Waste, manifesting as Outlier Explosion (Pattern 1, Chapter 13).*

*Output artifact: diagnosis + fix recommendation.*

---

## The Complete Pipeline

This walkthrough composed every major concept:

| Step | Concept | Chapter |
|---|---|---|
| Calibrate | Observers collect activation ranges | 9 |
| Compute scales | \\(S = \text{range} / (q_{\max} - q_{\min})\\) | 3 |
| Count boundaries | int32 → int8 at each layer output | 6, 7 |
| Predict error | \\(\text{Var}(\epsilon) = S^2/12\\), accumulates across boundaries | 5, 7 |
| Diagnose failure | Check graph → fallbacks → alignment → calibration → sensitivity | 13 |
| Fix | Percentile observer, SmoothQuant, mixed precision, or don't quantize | 9, 15, 12, 13 |

A practitioner who can execute this pipeline — calibrate, count, predict, measure, diagnose — can handle any quantization deployment. The individual chapters provide the depth. This walkthrough provides the composition.
