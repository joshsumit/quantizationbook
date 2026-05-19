# Chapter 11: Quantization-Aware Training

In this chapter, we quantize weights and activations during training-time simulation.

## When Passive Fails

PTQ has a structural ceiling: when weight and activation distributions are hostile to quantization, no post-hoc calibration can fix them. The weights were never trained to survive quantization. They produce ranges that int8 cannot represent without severe error.

Quantization-aware training (QAT) takes the opposite approach. Instead of accepting the model's distributions and hoping they fit, QAT modifies the training process so that the model learns to produce values that survive quantization. The model actively adapts to the constraint.

---

## Fake Quantization

QAT works by simulating quantization during training. At every point where quantization will eventually be applied — weight tensors and activation boundaries — a *fake quantization* node is inserted into the forward pass.

A fake quantization node does the following:

1. Takes a floating-point input value.
2. Quantizes it: maps it to the nearest int8 grid point.
3. Immediately dequantizes it: maps it back to floating-point.

The output is still floating-point, but it has been rounded to a value that is exactly representable in int8. The network's forward pass sees values as if they were quantized, but the computation remains in floating-point so that gradients can be computed normally.

Consider a weight value of 0.3021 with scale \\(S = 0.00784\\). Fake quantization maps it to the nearest grid point:

To see exactly how this simulated quantization is applied, we quantify it:

$$q = \text{round}(0.3021 / 0.00784) = \text{round}(38.54) = 39$$
$$\hat{r} = 39 \times 0.00784 = 0.3058$$

The original value 0.3021 becomes 0.3058 in the forward pass. The difference of 0.0037 is the simulated quantization error. The loss function now reflects this error. The optimizer sees it. The gradients account for it.

Over the course of training, the model adjusts its weights to minimize loss under these simulated quantization effects. Weights that would have landed far from grid points — causing large rounding errors — are gradually pushed toward values that align with the grid. The model learns to avoid the regions where quantization error is largest.

---

## The Straight-Through Estimator

There is a mathematical problem with fake quantization: the rounding operation is not differentiable. Its gradient is zero almost everywhere (between grid points, the output is constant) and undefined at the grid points themselves. If the gradient of rounding is zero, no learning signal passes through the fake quantization node, and the model cannot learn from the quantization simulation.

The *Straight-Through Estimator* (STE) solves this by replacing the rounding operation's gradient with the identity function during the backward pass. The forward pass rounds the value to the nearest grid point. The backward pass acts as if the rounding never happened — the gradient flows through unchanged.

This is mathematically wrong. The true gradient of a step function is zero between steps, not one. STE pretends the rounding operation is an identity, which it is not. The gradient signal that reaches the weights does not accurately reflect the function that was applied in the forward pass.

But STE works in practice, and understanding *why* it works despite being wrong is essential.

### Why STE Works Despite Being Wrong

The key insight is that STE does not need to be a correct gradient estimator. It only needs to be a *useful descent direction* — a direction that, when followed, reduces the loss.

An analogy: imagine navigating a staircase in the dark. You can’t see the stairs (the true gradient is a step function — zero between stairs, undefined at edges). But you can feel that the floor slopes slightly forward (STE says gradient = 1). You take a small step forward. Eventually your foot hits a stair edge and you step up. The slope direction was “wrong” in detail but “right” in that it moved you upward. STE works the same way: approximate direction, correct sign, good enough for iterative optimization.

Consider what the optimizer actually needs. It needs to know: "if I increase this weight slightly, will the loss go up or down?" The true gradient of the rounding function cannot answer this question — it says zero everywhere, meaning "changing the weight does nothing." But that is obviously false: changing a weight *does* change the loss, because the weight eventually moves past a grid-point boundary and snaps to a different integer.

STE provides the answer the optimizer needs by computing the gradient of the loss *with respect to the fake-quantized output* (which is a valid, nonzero gradient because the loss changes as the quantized output changes) and passing it directly through to the weight. The direction is correct — the sign of the gradient tells you whether the weight should increase or decrease to reduce loss. The magnitude is approximate — STE says "the sensitivity is 1.0 per unit change" when the true sensitivity is a staircase. But an approximate magnitude with a correct sign is enough for iterative optimization.

Formally, STE replaces:

$$\frac{\partial \text{round}(x)}{\partial x} = 0 \quad \text{(true but useless)}$$

with:

$$\frac{\partial \text{round}(x)}{\partial x} \approx 1 \quad \text{(false but useful)}$$

The loss function \\(L\\) is computed using the quantized values (the forward pass is correct). The gradient \\(\partial L / \partial \hat{w}\\) — where \\(\hat{w}\\) is the fake-quantized weight — is exact. STE's approximation only affects the chain rule step from \\(\hat{w}\\) back to \\(w\\). Because the forward quantization error is bounded (within \\(S/2\\)), the mismatch between the true staircase sensitivity and the STE surrogate does not explode; in practice it provides a stable descent signal.

In many implementations, a *clipped STE* is used: gradients pass through normally within the representable range but are set to zero for saturated values (those clamped to \\(q_{\min}\\) or \\(q_{\max}\\)). This prevents the optimizer from receiving misleading signals for values that are already at the clamp boundary.

Over thousands of iterations, these approximately-correct updates accumulate into weights that genuinely minimize the quantization-affected loss. The model converges not because each gradient step is exact, but because each step moves in roughly the right direction.

### A Worked Example: STE in Action

Consider a single weight \\(w = 0.34\\) with scale \\(S = 0.1\\) and the current loss gradient \\(\partial L / \partial y = -0.5\\) (the loss wants the output to increase).

**Forward pass:**
$$\hat{w} = \text{round}(0.34 / 0.1) \times 0.1 = \text{round}(3.4) \times 0.1 = 3 \times 0.1 = 0.30$$

The weight snapped from 0.34 to 0.30 — it rounded down to the nearest grid point.

**Backward pass (true gradient):**
$$\frac{\partial \hat{w}}{\partial w} = 0 \quad \Rightarrow \quad \frac{\partial L}{\partial w} = -0.5 \times 0 = 0$$

No learning signal. The weight is frozen.

**Backward pass (STE):**
$$\frac{\partial \hat{w}}{\partial w} \approx 1 \quad \Rightarrow \quad \frac{\partial L}{\partial w} \approx -0.5 \times 1 = -0.5$$

The optimizer receives a gradient of -0.5, meaning "increase this weight to reduce loss."

**Weight update** (learning rate \\(\eta = 0.01\\)):
$$w \leftarrow 0.34 - 0.01 \times (-0.5) = 0.34 + 0.005 = 0.345$$

After the update, \\(w = 0.345\\). Fake quantization still rounds to 0.30 (since \\(\text{round}(0.345/0.1) = 3\\)). More updates accumulate. Eventually \\(w\\) crosses 0.35 — the boundary between grid points 3 and 4. At that point:

$$\hat{w} = \text{round}(0.35 / 0.1) \times 0.1 = 4 \times 0.1 = 0.40$$

The quantized weight jumps from 0.30 to 0.40. The loss changes. The optimizer sees the effect and continues adjusting. Over many iterations, the weight settles at a value where the quantized output minimizes the loss — typically very close to a grid point.

---

## Fixed vs Learned Quantization Parameters

In standard QAT, the scale and zero-point are **not** learned by gradient descent. They are computed by an internal observer — the same kind of observer from Chapter 9 — that tracks the running statistics (min, max, or histogram) of the values passing through the fake quantization node. The observer updates \\(S\\) and \\(Z\\) as statistics change during training, but these updates are based on observed ranges, not on gradients from the loss function.

This means standard QAT learns *where the weights should be* (via STE), but the *grid itself* — its spacing and offset — is fixed by the observer. The weights adapt to fit a grid that is determined by statistics, not by the task loss.

For int8, this works well. With 256 grid points, the grid is fine enough that its exact placement rarely matters — moving one grid point left or right changes almost nothing. The dominant improvement comes from pushing weights toward grid points, which standard QAT handles.

For int4 (16 levels) or int2 (4 levels), the grid is so coarse that its exact placement matters enormously. Shifting the scale by 5% can move every grid point by a full step size, potentially changing which code every weight snaps to. At 4 bits, the observer-derived grid may not be optimal — a slightly different scale could produce lower task loss.

**Learnable fake quantization** addresses this by treating \\(S\\) and \\(Z\\) as trainable parameters with `requires_grad=True`. During the backward pass, gradients flow not only through the STE to the weights, but also to the scale and zero-point themselves. The optimizer adjusts the grid boundaries to minimize the loss — the grid adapts to the weights at the same time the weights adapt to the grid.

| Aspect | Standard (observer-derived \\(S\\), \\(Z\\)) | Learnable (\\(S\\), \\(Z\\) as parameters) |
|---|---|---|
| Scale/zero-point source | Computed from running statistics | Learned via gradient descent |
| What gradients update | Weights only (via STE) | Weights *and* quantization parameters |
| Grid placement | Fixed by data statistics | Optimized for task loss |
| Stability | High — observer statistics are smooth | Requires careful learning-rate tuning for \\(S\\), \\(Z\\) |
| Best for | int8 QAT (grid is fine enough) | int4 / int2 QAT (grid placement is critical) |

**When to use which.** Standard observer-derived QAT is the default for int8 — it is stable, well-supported, and sufficient. Learnable quantization parameters become valuable when pushing below 8 bits: if standard QAT at int4 still leaves an accuracy gap, making scale and zero-point learnable is the next lever before giving up on that bit width.

In PyTorch, these correspond to `FakeQuantize` (observer-derived) and `LearnableFakeQuantize` (gradient-learned) modules. Other frameworks have analogous distinctions.

---

## Where STE Breaks Down

STE has a specific blind spot: clamping.

When a value exceeds the quantization range and is clamped, the true gradient with respect to that value is zero — the output is flat at the boundary regardless of how the input changes. STE does not handle this correctly. It still passes the gradient through as if the clamp did not happen. The optimizer does not receive a direct signal that the value is being saturated.

The model does eventually learn to avoid saturated regions — not from the gradient through the clamp, but from the loss increasing when too many values are clamped. This is an indirect and slower learning signal. In practice, if a significant fraction of values in a layer are being clamped during QAT, convergence is slower and the final accuracy may be worse than if the model had been initialized closer to the quantization-friendly range.

This is why QAT is often applied as *fine-tuning* on a pre-trained model rather than training from scratch. The pre-trained weights start in a reasonable range, and QAT nudges them to align with the grid. Training from scratch with fake quantization active is significantly harder because the early random weights may be far from any quantization-friendly region.

---

## What QAT Produces

After QAT, the model's weight distributions look measurably different from the floating-point baseline:

- Weight values cluster closer to grid points. The "between grid points" region is less populated.
- Per-channel weight ranges are tighter. Outlier weights that would have forced wide scales have been pushed inward during training.
- Activation distributions are more bounded. The model has learned to avoid producing extreme values that would cause clipping.

**QAT vs failure patterns:**

- *Cumulative Rounding Noise*: QAT makes the model robust to repeated requantization rounding by training under those projections. The fake-quantization noise injected during training is not a signal the model learns to exploit — it is adversarial damage the model learns to withstand. After QAT, the model produces outputs that survive quantization not because the noise is useful, but because the weights have been pushed to positions where the noise does minimal harm.
- *Tail Clipping*: QAT reduces saturation by discouraging values outside the representable range.
- *Distribution Mismatch / Budget Waste*: QAT shrinks outliers and tightens per-channel ranges, reducing wasted codes.
- *Calibration Mismatch*: QAT does **not** fix this if the fine-tuning dataset is also unrepresentative of production.

The effect is visible if you compare histograms of weight values before and after QAT. The pre-QAT distribution is smooth and continuous. The post-QAT distribution shows subtle clustering around grid points — the model has learned the quantization grid.

### Worked Example: Same Weight, PTQ vs QAT

To see concretely what "learning the grid" means, trace one weight through both strategies.

**Setup.** A layer's per-channel weight scale is \\(S_w = 0.015\\) (symmetric int8, grid points at multiples of 0.015). One weight has the float32 value \\(w = 0.052\\).

**PTQ (passive).** The weight is rounded to the nearest grid point:

$$q = \text{round}\!\left(\frac{0.052}{0.015}\right) = \text{round}(3.467) = 3 \qquad w_{\text{PTQ}}' = 3 \times 0.015 = 0.045$$

PTQ error: \\(|0.052 - 0.045| = 0.007\\). That is \\(47\%\\) of a step size — nearly worst-case rounding.

**QAT (active).** During QAT fine-tuning, the optimizer sees (via the STE) that this weight is stuck between grid points 3 and 4. Over several training steps, the gradient nudges the weight toward the grid point that produces lower task loss. Suppose the optimizer shifts it to \\(w = 0.046\\):

$$q = \text{round}\!\left(\frac{0.046}{0.015}\right) = \text{round}(3.067) = 3 \qquad w_{\text{QAT}}' = 3 \times 0.015 = 0.045$$

QAT error: \\(|0.046 - 0.045| = 0.001\\). Same integer code (3), but the float weight moved closer to the grid point.

Alternatively, if grid point 4 (\\(= 0.060\\)) produces better task loss, the optimizer might push the weight to \\(w = 0.059\\):

$$q = \text{round}\!\left(\frac{0.059}{0.015}\right) = \text{round}(3.933) = 4 \qquad w_{\text{QAT}}' = 4 \times 0.015 = 0.060$$

QAT error: \\(|0.059 - 0.060| = 0.001\\). The weight jumped to a different code, but with minimal quantization error.

**The point.** PTQ accepts the weight at 0.052 and eats the 0.007 error. QAT adjusts the weight during training so that whichever grid point it lands on, the error is small *and* the task loss is low. Multiplied across millions of weights, this is why QAT recovers accuracy that PTQ cannot.

When this model is then quantized to int8, the accuracy drop is typically much smaller than PTQ on the same architecture — on quantization-hostile models, QAT often reduces the PTQ accuracy gap substantially (task- and backend-dependent).

Note that fake quantization for weights is commonly applied per-channel, while activation fake quantization is typically per-tensor. The granularity must match what the target backend supports at deployment.

---

## The Cost

QAT requires:

- Access to the training pipeline and training infrastructure
- A representative training dataset (or at least a fine-tuning dataset)
- Additional training epochs — often on the order of 10–30% of the original training budget for fine-tuning, depending on the recipe
- Modifications to the model definition (inserting fake quantization nodes)

For a model that took weeks to train on a GPU cluster, QAT adds days of additional compute. For a model that cannot be retrained — because the training data is unavailable, the training recipe is proprietary, or the compute budget is exhausted — QAT is not an option.

To put the cost in concrete terms: a ResNet-50 that trained in 2 hours might need 30 minutes of QAT fine-tuning. A BERT-base that trained in 4 days might need 8–12 hours of QAT. A 70B LLM that took months is practically infeasible to QAT.
These are advanced methods that improve basic rounding (explained earlier): GPTQ and AWQ (Chapter 17).

QAT is a last resort for models where PTQ's structural ceiling has been reached. It is not a default because it is expensive, and for quantization-friendly models, PTQ works just as well without the cost.

---

## QAT End-to-End: A Two-Layer Example

To see the full QAT process, trace it through a minimal network: two linear layers with ReLU, processing a single input.

**Setup:** Input \\(x = [1.0, -0.5]\\). Layer 1 weights: \\(W_1 = \begin{bmatrix} 0.72 & -0.41 \\ 0.33 & 0.89 \end{bmatrix}\\), bias \\(b_1 = [0.1, -0.2]\\). Layer 2 weights: \\(W_2 = \begin{bmatrix} 0.55 & -0.28 \end{bmatrix}\\), bias \\(b_2 = [0.05]\\). Scale \\(S = 0.01\\) for all quantization points (simplified for clarity). Target output: \\(y_{\text{target}} = 0.6\\).

**Step 1: Floating-point forward pass (no quantization).**

Layer 1 output: \\(h = W_1 x + b_1 = [0.72(1.0) + (-0.41)(-0.5) + 0.1,\; 0.33(1.0) + 0.89(-0.5) + (-0.2)] = [1.025, -0.315]\\)

After ReLU: \\([1.025, 0.0]\\)

Layer 2 output: \\(y = 0.55(1.025) + (-0.28)(0.0) + 0.05 = 0.614\\)

Loss (MSE): \\((0.614 - 0.6)^2 = 0.000196\\)

**Step 2: QAT forward pass (with fake quantization).**

Each weight is fake-quantized: \\(\hat{w} = S \cdot \text{round}(w / S)\\).

\\(\hat{W}_1 = \begin{bmatrix} 0.72 & -0.41 \\ 0.33 & 0.89 \end{bmatrix}\\) (these happen to be on grid points with \\(S = 0.01\\))

Layer 1 output before ReLU: \\([1.025, -0.315]\\). After fake quantization of activations: \\([1.03, -0.32]\\) (rounded to nearest 0.01). After ReLU: \\([1.03, 0.0]\\).

Layer 2 with fake-quantized weights: \\(y = 0.55(1.03) + (-0.28)(0.0) + 0.05 = 0.6165\\). Fake-quantized output: \\(0.62\\).

QAT loss: \\((0.62 - 0.6)^2 = 0.0004\\) — higher than float loss because quantization introduced error.

**Step 3: Backward pass with STE.**

The gradient \\(\partial L / \partial y = 2(0.62 - 0.6) = 0.04\\) flows backward. At each fake quantization node, STE passes the gradient through as-is. The optimizer updates all weights to reduce this quantization-affected loss.

**Step 4: After many iterations.**

The weights shift to values that produce outputs closer to the target *after* fake quantization. The final weight distribution differs measurably from the floating-point optimum — the QAT-trained weights are slightly "wrong" in float but produce better results when actually quantized to int8.

This is the fundamental output of QAT: weights that are optimized not for floating-point accuracy, but for *post-quantization* accuracy.

---

## Conceptual Consolidation

PTQ observes the model and hopes its distributions survive quantization. QAT trains the model to produce distributions that survive by construction. The mechanism is fake quantization: simulating quantization in the forward pass so that the loss function and optimizer can account for rounding and clamping errors during training.

The decision between PTQ and QAT is not ideological. It is diagnostic: does the model's existing distribution survive int8 quantization with acceptable accuracy? If yes, use PTQ. If no — if the accuracy drop exceeds tolerance and the structural signals from Chapter 10 are present — QAT is the tool that can reshape the distributions to fit. If PTQ fails due to Tail Clipping or Distribution Mismatch that calibration cannot tame, QAT is the mechanism that can address those patterns — at the cost of training compute.

### QAT Is Static Quantization

Both PTQ and QAT produce statically quantized models. The difference is *how* the quantization parameters are determined — not *when* they are applied.

- **PTQ** discovers parameters after training via calibration (Chapter 9–10).
- **QAT** learns parameters during training via fake quantization (this chapter).

In both cases, the final deployed model has the same structure: weights stored as int8 integers, activation scales and zero-points frozen as constants, no range computation at runtime. Both are static.

QAT is almost never combined with dynamic quantization. The reason is a conflict of purpose: QAT trains the model to be robust to a *specific fixed* scale at each boundary — that is its entire value. Dynamic quantization computes a *new* scale for every input, so the model already adapts to each input's range automatically. Training a model to tolerate a fixed constraint that will never be applied at inference is redundant. The accuracy benefit of QAT becomes negligible when scales adapt on the fly, and the training cost is not justified.

If you see QAT, assume the target is static quantization.

**Failure Signals**

- Quantized validation remains unstable after fine-tuning
- High clamp rate in fake-quant layers
- Accuracy collapses when switching from fake-quant to real inference
