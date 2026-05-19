# Chapter 10: Post-Training Quantization

In this chapter, we quantize weights and activations after training is complete.

## The Simplest Strategy

The full machinery is now in place: grids, scales, boundaries, requantization, fusion, and calibration. The first complete quantization strategy is *post-training quantization* (PTQ) — quantizing an already-trained model without any retraining.

PTQ is widely used because it is typically the cheapest route to deployment. The model trains in floating point as usual (often mixed precision). After training is complete, quantization is applied as a post-processing step. No gradients are computed. No training loop is modified. The weights are mapped directly to integers using calibrated scales.

---

## How PTQ Works, End to End

The complete PTQ workflow:

1. **Start with a trained floating-point model.** Training is finished. The weights are fixed.
2. **Fuse operators** where possible (Chapter 8). This reduces the number of requantization boundaries.
3. **Insert observers** at every remaining boundary in the fused graph (Chapter 9).
4. **Run calibration data** through the model. Observers collect activation statistics.
5. **Freeze observers.** Each observer produces a final scale and zero-point for its boundary.
6. **Quantize weights.** Each weight tensor is mapped to int8 (or int4) using per-tensor or per-channel scale, computed directly from the weight values.
7. **Replace float operators with quantized operators.** The model is now a quantized execution graph.

The entire process touches the model once, after training. No weight is updated. No gradient is computed. The weights are taken as-is and mapped to the nearest grid points.

### What the deployed model looks like

After PTQ, the model is fully static. Weights are stored as int8 integers — they are not floating-point values that get converted at runtime. Activation scales and zero-points are stored as constants baked into the graph at each quantization boundary.

**Concrete memory layout.** A layer with [4096 × 4096] weights under per-channel int8 quantization:

- Weight data: \\(4096 \times 4096 \times 1\\) byte = 16 MB (int8)
- Scale metadata: 4096 scales (one per output channel) × 2 bytes (float16) = 8 KB
- Per-layer footprint: 16.008 MB
- Compared to float32: \\(4096 \times 4096 \times 4\\) bytes = 64 MB
- Compression: \\(64 / 16.008 \approx 4.0\times\\)

For a 24-layer model: \\(24 \times 16.008 \approx 384\\) MB weights + 192 KB scales. Float32 equivalent: \\(24 \times 64 = 1{,}536\\) MB. The scale metadata (192 KB) is 0.05% of weight data — negligible overhead for per-channel quantization.

During inference, the hardware converts each layer's floating-point activations to int8 using the stored constants, executes the int8 matmul, and produces results — with no range computation at runtime. This is the performance advantage of static quantization: the hardware never pauses to scan a tensor for its min/max. Every parameter was decided during steps 3–6 above, and the inference engine simply reads them from memory.

The observers from step 3–5 are gone. They were temporary tools used during calibration. In the deployed model, no observer logic exists — only the constants they produced.

---

## PTQ Is Passive Range Ownership

PTQ does not modify the model to fit quantization. It observes the model's existing distributions — weights and activations — and sets ranges to match what it finds. The calibration data tells the model what the valid range is. The model has no say in the matter.

By *passive* we mean: PTQ accepts the model exactly as trained and hopes its weight and activation distributions happen to survive quantization. Contrast this with QAT (Chapter 11), which is *active*: it modifies the training process so the model learns distributions that deliberately survive quantization.

This passivity is both the strength and the limitation of PTQ.

**Strength:** PTQ is fast, requires no training infrastructure, and works on models where retraining is impossible — third-party models, proprietary architectures, or models where training data is unavailable.

**Limitation:** PTQ can only work within the bounds of the existing weight and activation distributions. If those distributions are not quantization-friendly, PTQ cannot fix them.

**Worked example: why PTQ has a ceiling.** Consider a layer with 4096 output channels. Per-channel weight ranges for 4095 channels fall in [0.01, 0.05] (range 0.04). One outlier channel has range [0.01, 5.2] (range 5.19). Under per-channel int8 quantization:

- Normal channels: step size \\(= 0.04 / 255 \approx 0.000157\\) — fine resolution
- Outlier channel: step size \\(= 5.19 / 255 \approx 0.0204\\) — 130× coarser

A weight of 0.03 in a normal channel maps to code \\(\text{round}((0.03 - 0.01) / 0.000157) = 127\\), with error \\(< 0.000079\\). The same weight of 0.03 in the outlier channel maps to code \\(\text{round}((0.03 - 0.01) / 0.0204) = 1\\), dequantized to \\(0.01 + 1 \times 0.0204 = 0.0304\\), with error 0.0004 — 5× larger, but still small. The real problem: a weight of 0.035 in the outlier channel also maps to code 1, making 0.030 and 0.035 indistinguishable. In the normal channel, they map to codes 127 and 159 — clearly distinct.

PTQ accepts this. QAT (Chapter 11) could push the outlier channel’s values toward a tighter range during training, reducing its step size. PTQ cannot — it works with the distributions as they exist.

---

## When PTQ Works

PTQ works well when the model's distributions cooperate. Specifically, when:

- **Weight distributions are compact.** Weights cluster near zero with relatively uniform per-channel ranges. The max/min ratio across channels is moderate (say, 3:1 to 5:1). Each channel can be quantized with good resolution.
- **Activation distributions are bounded.** Activations do not produce extreme outliers. The range covered by a min-max or percentile observer captures the meaningful values without sacrificing resolution.
- **The architecture is fusion-friendly.** Conv-BN-ReLU chains fuse cleanly, reducing the boundary count (Chapter 8).

Many mainstream vision CNNs meet these conditions. Models like ResNet-50, MobileNet, and EfficientNet can typically achieve sub-1% top-1 accuracy drop under int8 PTQ with competent calibration (see references). The weight distributions were shaped by architectures and training practices (batch normalization, ReLU activations) that happen to produce quantization-friendly ranges.

---

## When PTQ Fails

PTQ fails when the model's distributions are structurally hostile to quantization.

Consider a model where certain weight channels contain values 50× larger than the median weight value. Under per-tensor quantization, the scale is set by the extreme channel. Every other channel — the vast majority — loses resolution. Under per-channel quantization, each channel gets its own scale, which helps — but if the extreme channel's values are genuinely needed for the model's behavior, its int8 representation at that scale may still be too coarse.

Now consider the activations. If certain activation channels consistently produce values at 60.0 while all other channels peak below 2.0, the per-tensor scale is dominated by the outlier channels. The common channels get a handful of representable levels (Chapter 5's representation error in action).

In both cases, recalibrating with different data does not help. The problem is not the calibration — the problem is the distributions themselves. The scales are set correctly given the data; the data just doesn't fit an int8 grid well.

This is PTQ's structural ceiling. PTQ can only work within the distributions the model produces. It cannot reshape those distributions. When the distributions are structurally hostile, calibration alone usually cannot recover accuracy — you typically need distribution-shaping methods (Chapter 15) or retraining (QAT, Chapter 11). Between PTQ and full QAT sit intermediate approaches that reshape distributions without full retraining; later chapters cover these.

**How to identify which layers are responsible.** In practice, PTQ failures are rarely uniform across the model. A small number of layers — often 3–5 out of dozens — dominate the total accuracy loss. The standard diagnostic is a *per-layer sensitivity analysis*: quantize one layer at a time while keeping all others in float, measure the accuracy drop, and rank layers by their individual contribution to error.

> **📊 INSERT DIAGRAM: Per-Layer Sensitivity Analysis (Bar Chart)**
>
> A horizontal bar chart for a 24-layer model:
>
> ```
> Layer  | Accuracy drop when this layer alone is quantized to int8
> -------|---------------------------------------------------------
>   1    | ██████████████  1.4%     <-- first layer (large impact)
>   2    | ████  0.4%
>   3    | ██  0.2%
>  4-20  | █  0.05% each               <-- bulk of layers: negligible
>  21    | ███  0.3%
>  22    | ██  0.2%
>  23    | ███████████  1.1%      <-- near-output layer (large impact)
>  24    | ████████████████  1.6%  <-- output layer (largest impact)
> ```
>
> Annotations:
> - Highlight that ~3 layers (1, 23, 24) account for >70% of total accuracy loss
> - Label these as "keep in FP16" candidates (mixed precision, Chapter 12)
> - Show that the middle layers (4–20) can safely be int8
> - Add callout: "Sensitivity analysis takes minutes and saves weeks of debugging"

Layers at the input and output of the network are typically most sensitive — early layers because their errors propagate through every subsequent layer (Chapter 5), and output layers because their errors directly affect the final prediction with no opportunity for averaging.

When sensitivity analysis reveals that only a few layers dominate the error, the remedy is *mixed precision* (Chapter 12): keep those layers in float16 while quantizing the rest. When sensitivity analysis shows broadly distributed error across many layers, the problem is structural and requires a different approach entirely.

Heuristic signals that PTQ has hit its ceiling (backend and task dependent):
- Top-line metric drop > ~2–3% after careful calibration
- Per-channel weight range ratio (max range / median range) > ~10×
- Activation channels with persistent outliers an order of magnitude above the bulk
- Switching from min-max to percentile observers helps but doesn't resolve the drop

**PTQ failure patterns:**

- *Calibration Mismatch*: ranges wrong because calibration data doesn't represent production distribution.
- *Tail Clipping*: range too tight; saturation events frequent at boundary clamps.
- *Distribution Mismatch / Budget Waste*: range too wide due to outliers; most codes wasted in empty regions.
- *Resolution Collapse*: even per-channel resolution is too coarse for the model's required distinctions.

If PTQ looks slow rather than inaccurate, first suspect Fusion Loss or Silent Fallback (Chapter 4).

When these signals appear, the next step is not better calibration. It is a fundamentally different strategy — one where the model actively adapts to the quantization constraint during training.

> **📊 INSERT DIAGRAM: PTQ Failure Decision Flowchart**
>
> A decision tree that an engineer follows when PTQ accuracy is unacceptable:
>
> ```
> PTQ accuracy drop > threshold?
>   │
>   ├─ No → Deploy. PTQ is sufficient.
>   │
>   └─ Yes → Run per-layer sensitivity analysis
>              │
>              ├─ Few layers dominate? → Mixed Precision (Ch.12)
>              │     Keep sensitive layers in FP16, quantize the rest.
>              │
>              └─ Error spread across many layers? → QAT (Ch.11)
>                   Retrain with fake quantization nodes.
>                   (Requires training infrastructure and data.)
> ```
>
> Include a side note: "This flowchart follows escalating cost — each step down requires more compute, more expertise, and more infrastructure."

---

## Conceptual Consolidation

PTQ is passive: it observes the model's existing distributions and maps them to integers. When distributions are compact and well-behaved, PTQ is the right tool — cheap, effective, and sufficient. When distributions are hostile — high dynamic range, extreme outliers, large per-channel variance — PTQ hits a structural ceiling that no calibration improvement can overcome.

The diagnostic question after PTQ is: is the remaining accuracy loss caused by range estimation (fixable with better calibration) or by the distribution shape itself (not fixable without retraining)? If the loss is Tail Clipping, calibration or observer choice can help. If the loss is Distribution Mismatch or Resolution Collapse, calibration won't fix it — you need distribution shaping or retraining.

**Failure Signals**

- Accuracy drops sharply after conversion despite stable float baseline
- Many values clamp to min/max codes
- Small set of layers dominates sensitivity analysis
