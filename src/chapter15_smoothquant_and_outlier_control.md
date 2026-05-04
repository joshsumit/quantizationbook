# Chapter 15: SmoothQuant and Outlier Control

## Redistributing Difficulty

Transformer activations contain channel-level outliers that destroy per-tensor quantization resolution. The obvious responses — widen the range (lose resolution), clip the outliers (lose the outlier values), or use per-channel activation quantization (doesn't handle token-level variation) — are all inadequate.

SmoothQuant introduces a different approach: instead of fighting the outliers, mathematically redistribute them from activations (where they cause damage) to weights (where they can be absorbed).

---

## The Asymmetry Between Activations and Weights

Quantizing activations and quantizing weights are not equally hard for a given layer. In a typical transformer linear layer:

- **Activations** have extreme outlier channels — 2 out of 512 channels with values reaching 60, while the rest stay below 2. The max/median ratio is 30:1 or higher.
- **Weights** are well-behaved — values are approximately Gaussian, centered near zero, with a max/median ratio of 3:1 to 5:1 per channel.

Weights have headroom. Their distributions are compact enough that expanding them by a moderate factor (say 5×) would not push them outside the range that int8 can represent with adequate resolution. Activations, on the other hand, are at their limit — the outlier channels are already forcing catastrophic range expansion.

SmoothQuant exploits this asymmetry: shrink the activations where they are extreme, expand the weights where they are compact. The mathematical result is identical. The quantization behavior is dramatically better.

> **📊 INSERT DIAGRAM: SmoothQuant Before and After — Range Migration**
>
> Two side-by-side histograms showing the same linear layer, before and after SmoothQuant:
>
> ```
> BEFORE SmoothQuant:                         AFTER SmoothQuant:
>
> Activations (X):                            Smoothed Activations (X / s):
> │                    █                      │
> │                    █                      │     █████
> │  ███████           █  outlier             │   █████████
> │██████████         █  channels            │  ███████████  ← outliers shrunk
> └──────────────────────     └─────────────────────
> Range: [-2, 60]  (60:1 ratio)               Range: [-2, 5]  (2.5:1 ratio)
> int8 step: 0.24  ← too coarse               int8 step: 0.027  ← much finer
>
> Weights (W):                                 Scaled Weights (s × W):
> │    ████                                     │       ██
> │  ████████                                  │    ████████
> │ ██████████                                 │  ████████████  ← weights expanded
> └──────────────                               └─────────────────
> Range: [-0.1, 0.1]  (compact)               Range: [-1.2, 1.2]  (wider, still fine)
> int8 step: 0.00078                          int8 step: 0.0094
> ```
>
> Key annotations:
> - "The output Y = X × W is identical — division and multiplication cancel"
> - "Activation range shrank 12× → step size improved 9×"
> - "Weight range grew 12× — but weights had room to spare"
> - "Net effect: the HARDER-to-quantize tensor (activations) improved, the EASIER tensor (weights) absorbed the cost"

---

## The Mathematical Transformation

For a linear layer computing \\(Y = XW\\), SmoothQuant inserts a per-channel scaling factor \\(s\\) between the input and the weights:

$$Y = X W = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)$$

Here \\(\text{diag}(s)\\) is a diagonal matrix: a matrix that is zero everywhere except on its main diagonal, which holds the values of the vector \\(s\\). Multiplying by \\(\text{diag}(s)\\) simply scales each column \\(j\\) by \\(s_j\\). Multiplying by \\(\text{diag}(s)^{-1}\\) divides each column \\(j\\) by \\(s_j\\). The two operations cancel, so the output \\(Y\\) is mathematically identical — but now the quantization properties of the inputs and weights are different.

Let \\(\tilde{X} = X \cdot \text{diag}(s)^{-1}\\) and \\(\tilde{W} = \text{diag}(s) \cdot W\\).

The output \\(Y\\) is mathematically identical — \\(\tilde{X} \tilde{W} = X W\\). But the quantization properties of \\(\tilde{X}\\) and \\(\tilde{W}\\) are different from those of \\(X\\) and \\(W\\).

For a specific channel \\(j\\):
- \\(\tilde{X}_j = X_j / s_j\\) — the activation channel is divided by \\(s_j\\)
- \\(\tilde{W}_j = W_j \times s_j\\) — the corresponding weight channel is multiplied by \\(s_j\\)

If \\(s_j\\) is large, channel \\(j\\)'s activations shrink (becoming easier to quantize) while its weights expand (using more of their quantization range but still fitting).

---

## Choosing the Scaling Factor

The scaling factor \\(s\\) is chosen to balance quantization difficulty between activations and weights. A common formula:

$$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$$

The parameter \\(\alpha \in [0, 1]\\) controls the migration strength — how aggressively difficulty is moved from activations to weights.

- \\(\alpha\\) closer to 1 emphasizes activation outlier suppression: each activation channel is divided by a value close to its own maximum, making all channels roughly uniform in range. Weights absorb the corresponding expansion.
- \\(\alpha\\) closer to 0 emphasizes weight normalization influence: the scaling is driven primarily by weight magnitudes, and activations see less compression.
- \\(\alpha = 0.5\\): balanced. Difficulty is split equally between activations and weights.

Typical values are \\(\alpha \approx 0.5\\), found empirically to work well across a range of transformer models.

---

## A Concrete Example

A linear layer with 512 input channels. Channel 137 is an outlier.

**Before smoothing:**
- Activation channel 137: max |value| = 62.0
- All other activation channels: max |value| < 2.0
- Weight channel 137: max |value| = 0.3

Per-tensor activation quantization with range [-62, 62]:
$$S_{\text{act}} = \frac{124}{255} \approx 0.486$$

For normal channels (values in [-2, 2]): usable grid points \\(\approx 4.0 / 0.486 \approx 8\\). Severe resolution loss.

**After smoothing (\\(\alpha = 0.5\\)):**

$$s_{137} = \frac{62.0^{0.5}}{0.3^{0.5}} = \frac{7.87}{0.548} \approx 14.4$$

- Smoothed activation channel 137: \\(62.0 / 14.4 \approx 4.3\\)
- Adjusted weight channel 137: \\(0.3 \times 14.4 \approx 4.3\\)

The activation outlier has been absorbed. Channel 137's activation now peaks at 4.3 — comparable to other channels. The per-tensor activation range shrinks to approximately [-4.3, 4.3]:

$$S_{\text{act}} = \frac{8.6}{255} \approx 0.034$$

For normal channels (values in [-2, 2]): usable grid points \\(\approx 4.0 / 0.034 \approx 118\\). A 15× improvement in resolution compared to the unsmoothed case.

The weight channel 137 expanded from max 0.3 to max 4.3. Per-channel weight quantization can accommodate the expanded range without clipping, but the step size widens; the question is whether the added weight quantization error is acceptable. In this case the weight range was compact to begin with, so the expanded range is well within int8's capacity.

SmoothQuant is commonly applied to QKV projections and the first MLP linear layer in each transformer block — the layers most affected by activation outliers. The per-channel scaling can often be folded into surrounding LayerNorm parameters or bias terms, adding no runtime overhead.

*Canonical category: Distribution Mismatch / Budget Waste — SmoothQuant directly addresses the outlier-driven range inflation that wastes most of the quantization grid on empty space.*

Note that SmoothQuant inherits Calibration Mismatch risk: the per-channel activation scales \\(\max(|X_j|)\\) are gathered from calibration data. If calibration is unrepresentative, the scaling factors \\(s\\) will be wrong, and the smoothed distributions may not match production behavior.

---

## Limits

SmoothQuant assumes outliers are *channel-consistent*: the same channels produce outliers across all tokens. The per-channel scaling factor \\(s\\) is computed once and applied identically to every token.

If outlier locations vary per-token — channel 137 is extreme for token 5 but normal for token 20 — the fixed \\(s_{137}\\) either oversmooths (suppressing token 20's normal values) or undersmooths (failing to tame token 5's outlier). The per-channel assumption breaks down.

Additionally, if weights are already near their quantization limits, absorbing more range from activations may push weight values past their own clipping thresholds. The migration is not free — it shifts the problem, and the target must have capacity to absorb it.

---

## The Conservation Principle

SmoothQuant does not change the underlying function. It redistributes dynamic range between operands. Reducing activation range by a factor \\(s_j\\) necessarily increases weight range by the reciprocal factor for the affected channels. The total "difficulty" of quantizing a layer — the combined range that activations and weights must cover — is determined by the mathematical function the layer computes. SmoothQuant does not reduce this total difficulty. It redistributes it.

Activations are the overloaded side — dominated by outlier channels that force catastrophic range expansion. Weights are the underloaded side — compact, with room to expand. SmoothQuant compresses the activation side (dividing by \\(s\\)) and expands the weight side (multiplying by \\(s\\)).

This clarifies the limits. If the weight side is already near its quantization boundary — weights already span most of the int8 range — there is nowhere for the transferred activation difficulty to go. SmoothQuant degrades the weight quantization while only partially fixing the activation quantization.

*When migration exhausts weight headroom, the consequence is Resolution Collapse in weights (the same pattern that Chapter 14 describes for activations, now shifted to the other operand) or Tail Clipping if expanded weights are clamped.*

---

## When SmoothQuant Fails

SmoothQuant is not universal. It fails predictably in three scenarios:

**Scenario 1: Token-varying outlier channels.** Some transformer architectures produce outliers that shift channels across tokens. A per-channel \\(s\\) computed from calibration data matches the average behavior but misses per-token extremes. Symptom: accuracy improves slightly over unsmoothed quantization but remains far below float16. Diagnostic: compare outlier channel indices across different input sequences — if the set of top-5 outlier channels changes by more than 20%, channel-consistency has broken down.

**Scenario 2: Weight saturation.** If a layer's weights already span most of the int8 range before smoothing, the absorbed activation difficulty pushes weight quantization error beyond acceptable levels. Symptom: weight quantization error (MSE or max error) increases post-smoothing, and the layer's output error is *worse* than before smoothing. Diagnostic: after applying \\(s\\), check whether per-channel weight step size increases sharply relative to the unsmoothed baseline, and whether per-channel weight quantization MSE grows. If the weight range expansion exceeds the headroom the weights originally had, SmoothQuant is transferring more difficulty than the weights can absorb.

**Scenario 3: Extreme α sensitivity.** For some models, no single α value works — α = 0.5 oversmooths weights while α = 0.3 undersmooths activations. The "balanced" point does not exist. This typically occurs when different layers have radically different activation-to-weight difficulty ratios. A global α fails; per-layer α tuning helps but adds calibration complexity.

**Worked example: global vs. per-layer α.** Two layers in the same model:

- *Layer A:* activation range [-1, 50] (outlier ratio 50:1), weight range [-0.1, 0.1]. Needs aggressive smoothing.
- *Layer B:* activation range [-1, 2] (ratio 2:1), weight range [-0.5, 0.5]. Nearly normal — needs minimal smoothing.

With global \\(\alpha = 0.5\\):
- Layer A: \\(s = 50^{0.5} / 0.1^{0.5} = 7.07 / 0.316 = 22.4\\). Smoothed activation max \\(= 50/22.4 = 2.23\\). Smoothed weight max \\(= 0.1 \times 22.4 = 2.24\\). Per-tensor activation scale \\(= 4.46/255 = 0.0175\\) — reasonable.
- Layer B: \\(s = 2^{0.5} / 0.5^{0.5} = 1.41 / 0.707 = 2.0\\). Smoothed activation max \\(= 2/2 = 1.0\\). Smoothed weight max \\(= 0.5 \times 2.0 = 1.0\\). Weight range expanded from 0.5 to 1.0 — weight step size doubled from \\(1.0/255 = 0.0039\\) to \\(2.0/255 = 0.0078\\). Layer B’s weights are now 2× coarser for no benefit (its activations were already quantizable).

With per-layer tuning: Layer A gets \\(\alpha = 0.7\\), Layer B gets \\(\alpha = 0.3\\):
- Layer A: \\(s = 50^{0.7} / 0.1^{0.3} = 18.0 / 0.501 = 35.9\\). Smoothed activation max \\(= 50/35.9 = 1.39\\). Excellent compression.
- Layer B: \\(s = 2^{0.3} / 0.5^{0.7} = 1.23 / 0.616 = 2.0\\). Smoothed weight max \\(= 0.5 \times 2.0 = 1.0\\) — similar to global, but Layer A is much better controlled.

Per-layer α tuning adds calibration cost (one α per layer, searched by grid sweep) but avoids the one-size-fits-all penalty.

When SmoothQuant hits these limits, the next options are weight-only quantization (Chapter 16) — which avoids activation quantization entirely — or more sophisticated algorithms like GPTQ and AWQ (Chapter 17) that optimize weight values directly.

*Failure pattern tags: Scenario 1 is residual Distribution Mismatch along the token dimension. Scenario 2 is Resolution Collapse or Tail Clipping transferred to weights. Scenario 3 is a sign that layer-level heterogeneity requires per-layer α or a different strategy entirely.*

---

## Conceptual Consolidation

SmoothQuant is not a hack. It is a mathematical equivalence — the float output is identical before and after the transformation. The only thing that changes is how the values are distributed between activations and weights, and therefore how well they survive quantization.

The core insight is asymmetry: activations have outliers and no headroom; weights are compact and have headroom. SmoothQuant migrates difficulty from where it hurts to where it can be absorbed. It works when outliers are channel-consistent and weights have room to expand. When these conditions hold, it converts a catastrophically unquantizable layer into a normally quantizable one — without retraining.
