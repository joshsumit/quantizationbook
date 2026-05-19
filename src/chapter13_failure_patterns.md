# Chapter 13: Failure Patterns

## A Diagnostic Atlas

The complete quantization machinery is now in place: the representational grid, the scale contract, hardware constraints, three error types, graph boundaries, requantization, fusion, calibration, and three quantization strategies.

It is now possible to diagnose quantization failures systematically. Before examining the individual failure patterns, here is the diagnostic order — the sequence in which to investigate when a quantized model misbehaves:

> **📊 INSERT DIAGRAM: Quantization Failure Diagnostic Flowchart**
>
> A top-down decision tree for diagnosing a failing quantized model:
>
> ```
> Quantized model underperforming?
>   │
>   ├─ Is it SLOWER than float? (throughput problem)
>   │     ├─ Check backend logs for dequant pairs → Silent Fallback (Pattern 3)
>   │     └─ Check kernel profiler for unfused ops → Fusion Loss (Ch.8)
>   │
>   └─ Is it LESS ACCURATE? (quality problem)
>         │
>         ├─ Uniform degradation across all inputs?
>         │     ├─ Check activation histograms for outlier channels → Outlier Explosion (Pattern 1)
>         │     └─ Check per-layer sensitivity → Resolution Collapse / Scale Mismatch (Pattern 5)
>         │
>         ├─ Degradation only on production data (not calibration)?
>         │     └─ Calibration Drift (Pattern 4)
>         │
>         ├─ Depth-dependent degradation (deeper = worse)?
>         │     └─ Residual Ghost (Pattern 2) — check scale alignment at skip connections
>         │
>         └─ Nothing works (PTQ, QAT, all fail)?
>               └─ Unquantizable Model (Pattern 6) — raise precision floor
> ```
>
> Annotate: "Start with structural/throughput checks (cheap and fast) before diving into accuracy diagnosis (requires profiling and sensitivity sweeps)."

---

## Diagnostic Order

When accuracy drops after quantization and the cause is not obvious, check in this order:

1. **Graph boundaries**: Count the boundaries in the fused graph. Are there more than the layer count would predict? Export the post-fusion execution graph and count requant/rescale nodes. (Chapters 6, 8)
2. **Silent fallbacks**: Verify every operation executes in int8 on the target backend. Confirm kernel coverage via backend logs or profiler; search for unexpected dequantize/requantize pairs. (Chapter 4)
3. **Scale alignment**: At every elementwise operation, check whether input scales match. Record domain parameters at each elementwise op and count inserted rescale operations. (Chapter 8)
4. **Calibration**: Compare calibration data statistics against production data statistics. Compare exceedance and saturation rates between calibration and production. (Chapter 9)
5. **Per-layer sensitivity**: If the above checks pass, identify which specific layers contribute most to the accuracy drop. Sweep precision per-layer or disable quantization per-layer to compute each layer's contribution. (Chapter 12)

This order prioritizes structural problems (wrong graph, wrong backend support) before statistical problems (wrong calibration) before layer-level problems (individual layer sensitivity). Structural problems are cheap to check and have outsized impact. Calibration issues are next. Per-layer debugging is the most expensive and is a last resort.

Each of the five patterns below maps to one or more of these checkpoints. Use the order above as a triage sequence, and the patterns below as a reference for identifying and naming what you find.

---

## Pattern 1: Outlier Explosion

*Canonical category: Distribution Mismatch / Budget Waste; Tail Clipping when outliers are clamped.*

**Symptom:** Accuracy degrades uniformly across the model. No single layer appears broken, but every layer is slightly worse. Output quality drops in a way that feels like resolution loss rather than catastrophic failure.

**Root cause:** A few extreme values in the activation or weight distribution force the scale wide, destroying resolution for the majority of values. This is representation error (Chapter 5) caused by a range set too wide to accommodate outliers.

**Mechanism:** Consider a layer where 99.9% of activations fall in [-1.0, 3.0] but one channel produces values reaching 50.0. If the observer uses min-max (Chapter 9), the scale is set for [-1.0, 50.0] — a range of 51 units. The step size is \\(51/255 \approx 0.20\\). Values in the common range [-1.0, 3.0] get approximately 20 grid points. The layer effectively operates at 4–5 bit resolution instead of 8 bit.

**Diagnostic question:** What is the ratio between the 99.9th percentile and the maximum activation value? The *99.9th percentile* (written p99.9) is the value below which 99.9% of observations fall — it is robust to single extreme outliers. A p99.9/max ratio below 0.1 (i.e., the maximum is more than 10× the 99.9th percentile) strongly suggests outlier explosion.

**Fix / Mitigation:**

- Switch the min–max observer to a percentile or histogram observer (Chapter 9) — this trades a small amount of clipping for much better resolution in the common range.
- Use per-channel quantization for weights; per-group for activations if the backend supports it.
- Apply outlier-handling transforms that redistribute outlier magnitude across channels (Chapter 15).
- If outliers are concentrated in a few layers, keep those layers in higher precision (mixed precision, Chapter 12).

---

## Pattern 2: Residual Ghost

*Canonical category: Cumulative Rounding Noise; may trigger Distribution Mismatch at scale alignment and Fusion Loss at inserted rescale boundaries.*

**Symptom:** Accuracy degrades specifically in architectures with skip connections. The model performs worse on tasks that depend on fine-grained features, while coarse predictions remain roughly intact. The degradation scales with network depth.

**Root cause:** The main branch and the residual branch are independently requantized with different scale parameters. When they merge at the elementwise addition, their rounding errors combine (Chapter 7). If the scales are mismatched, a rescale insertion adds another boundary (Chapter 8), injecting additional error at every skip connection.

**Mechanism:** In a ResNet with 25 residual blocks, each block contributes one error-merging site. If the main and residual paths have mismatched scales, each merge point also contributes a rescale boundary. Over 25 blocks, the accumulated error from independently quantized paths reshapes the representation — particularly in the higher layers where the model relies on subtle activation differences.

**Worked example: how small errors amplify through depth.** Consider a 4-layer network where each layer multiplies its input by a weight matrix (simplified as a scalar multiplier \\(m\\) for illustration).

- Layer 1 receives input \\(x = 1.000\\), multiplier \\(m_1 = 1.5\\). Quantization introduces error \\(\epsilon_1 = 0.004\\).
  - Output: \\(1.5 \times 1.000 + 0.004 = 1.504\\) (true: 1.500).
- Layer 2 receives 1.504, multiplier \\(m_2 = 2.0\\). Adds its own error \\(\epsilon_2 = 0.003\\).
  - Output: \\(2.0 \times 1.504 + 0.003 = 3.011\\) (true: \\(2.0 \times 1.500 = 3.000\\)). Cumulative error: 0.011.
- Layer 3 receives 3.011, multiplier \\(m_3 = 1.8\\). Adds \\(\epsilon_3 = 0.005\\).
  - Output: \\(1.8 \times 3.011 + 0.005 = 5.425\\) (true: \\(1.8 \times 3.000 = 5.400\\)). Cumulative error: 0.025.
- Layer 4 receives 5.425, multiplier \\(m_4 = 2.5\\). Adds \\(\epsilon_4 = 0.004\\).
  - Output: \\(2.5 \times 5.425 + 0.004 = 13.567\\) (true: \\(2.5 \times 5.400 = 13.500\\)). Cumulative error: **0.067**.

Each layer's individual error is tiny (0.003–0.005). But Layer 1's error of 0.004 was multiplied by \\(m_2 \times m_3 \times m_4 = 2.0 \times 1.8 \times 2.5 = 9.0\\), contributing 0.036 to the final error alone. The errors don't just add — they get *amplified* by every subsequent layer's weights. This is why residual-ghost failures scale with depth, and why early layers are more sensitive: their errors pass through more amplification stages.

**Diagnostic question:** At each residual addition, do the main and residual paths share the same \\((S, Z)\\)? How many rescale insertions exist in the fused graph?

**Fix / Mitigation:**

- Enforce scale alignment at residual merge points (compiler or quantizer pass) so both branches share the same output domain.
- Fuse where possible to reduce independent requantization events before the merge.
- If alignment widens the range too much, move the merge to higher precision (mixed precision localized to merge sites, Chapter 12).
- Use QAT to train robustness specifically to merge-point quantization noise (Chapter 11).

---

## Pattern 3: Silent Fallback

*Canonical category: Silent Fallback (runtime pattern); causes boundary explosion and throughput collapse.*

**Symptom:** The quantized model runs slower than expected. In some cases, it is slower than the floating-point original. Accuracy may be fine — the failure is in throughput.

**Root cause:** One or more operations are not supported by the hardware's int8 capability envelope (Chapter 4). These operations silently fall back to float32, incurring dequantize-compute-requantize overhead at each fallback boundary.

**Mechanism:** A model uses GELU activation, which is not in the int8 capability envelope of the target backend. At each GELU layer, the runtime dequantizes int8 to float32, computes GELU in float, and requantizes back to int8. Each fallback adds two memory-format conversions and two requantization boundaries. If GELU appears after every linear layer in a 24-layer transformer, that is 48 additional boundaries and 48 additional memory round trips — none of which were in the quantization plan.

**Concrete latency cost.** For a [1, 4096] activation tensor in int8 (4 KB), each fallback boundary requires: dequantize int8 → float32 (4 KB → 16 KB, kernel launch ~5 µs), GELU compute in float32 (~20 µs), requantize float32 → int8 (16 KB → 4 KB, ~5 µs). Per-layer fallback cost: ~30 µs. If the fused int8 Linear-ReLU alternative takes ~25 µs for compute, the GELU fallback adds 5 µs overhead per layer. Across 24 layers: \\(24 \times 5 = 120\\) µs extra per inference. For a model that should complete inference in 2 ms, this is a 6% slowdown from a single unsupported activation function — and the model appears fully quantized in the graph.

The model appears quantized. The weights are int8. But the critical path runs through float operations surrounded by conversions, and the net effect is worse than running the entire model in float.

**Diagnostic question:** Does every operation in the graph execute in int8 on the target backend? Are there any dequantize/requantize pairs that were not present in the original quantized graph?

**Fix / Mitigation:**

- Replace unsupported ops with backend-friendly approximations or fused variants (e.g., GELU approximations, or SiLU if supported).
- Re-export the graph with fusion-friendly patterns (Conv/Linear + activation fused before quantization).
- Verify backend kernel selection via logs or profiler and block deployment if any float fallback appears in the critical path.

---

## Pattern 4: Calibration Drift

*Canonical category: Calibration Mismatch (initial mismatch) and Calibration Drift (time-varying subtype); often leads to Tail Clipping downstream.*

**Symptom:** The quantized model passes validation with acceptable accuracy but degrades on production data. The degradation may appear gradually as production data distribution shifts, or it may be immediate if the calibration data was fundamentally unrepresentative.

**Root cause:** The calibration dataset (Chapter 9) does not match the deployment distribution. The computed scales and zero-points are correct for the calibration data but wrong for the data the model actually encounters.

**Mechanism:** A speech recognition model is calibrated on clean studio-recorded audio. In production, it encounters noisy phone recordings with higher dynamic range. Activation values regularly exceed the calibrated range, causing clipping at every affected layer. The model was never calibrated for this input distribution, and the immutability of the parameters (Chapter 3) means the wrong ranges persist for every inference.

**Diagnostic question:** Was the calibration data drawn from the same distribution as production data? Do production activation ranges exceed the calibrated ranges?

**Fix / Mitigation:**

- Expand calibration dataset coverage to include production segments; stratify by known regimes (noise levels, device types, data sources).
- Track saturation rate and range exceedance in production telemetry — this is the actionable invariant.
- When drift is expected: consider dynamic quantization for activations or periodic recalibration (Chapter 12).

---

## Pattern 5: Scale Mismatch at Boundaries

*Canonical category: Cumulative Rounding Noise (from extra boundaries); may trigger Fusion Loss and Silent Fallback at inserted rescale ops.*

**Symptom:** Unexpected additional requantization boundaries appear in the execution graph — more than the layer count would predict. Latency is higher than expected, and accuracy drops are concentrated at specific points in the graph.

**Root cause:** Elementwise operations (add, concat) with inputs from branches that have different scales force rescale insertions (Chapter 8). Each insertion adds a requantization boundary that was not in the original plan.

**Mechanism:** A model with a feature pyramid network concatenates feature maps from different stages. Each stage has been quantized with different scales. The concatenation requires all inputs to share one \\((S, Z)\\). Rescale operations are inserted on two of the three input branches, adding two boundaries at each pyramid level. Across four pyramid levels, eight additional boundaries appear — each introducing rounding error.

**Diagnostic question:** At each elementwise operation, do all inputs share the same \\((S, Z)\\)? How many rescale insertions does the fused graph contain?

**Fix / Mitigation:**

- Force shared domain for branches feeding elementwise ops (alignment pass in the quantizer).
- Restructure the graph to move concatenation or addition to a point where domains already match (common graph rewrite).
- Fuse rescale into producer epilogues where possible to avoid extra materialization.

---

## Pattern 6: The Unquantizable Model

*Canonical category: Resolution Collapse (dominant); may also involve Silent Fallback and capability envelope limits.*

Not every quantization failure has a fix. Sometimes the correct diagnosis is: *do not quantize this model below a certain precision without architectural change.*

**Canonical example:** A small language model (350M parameters) with:
- LayerNorm before every attention block (producing activation outliers per Chapter 14)
- GELU activations throughout (outside the int8 capability envelope on most backends)
- No grouped-query attention (full KV-heads, maximizing the KV-cache cost)
- Narrow hidden dimension (1024) — meaning each dot product has only 1024 terms, giving less averaging to smooth out per-weight quantization error

Quantization results (illustrative numbers):

| Method | Precision | Accuracy Drop | Outcome |
|---|---|---|---|
| PTQ | int8 | 12% | Collapse |
| PTQ + percentile observer | int8 | 8% | Still unacceptable |
| SmoothQuant + PTQ | int8 | 5% | Better but still degraded |
| QAT | int8 | 1.5% | Acceptable — but QAT cost 3 GPU-days |
| PTQ | int4 | 38% | Total collapse |
| GPTQ | int4 | 18% | Still incoherent |
| AWQ | int4 | 15% | Marginally better, still broken |

**The diagnosis:** At 350M parameters, the model is too small for aggressive quantization. Its representations are already compact — there is less redundancy for quantization to exploit. Each weight carries more information per parameter than in a 70B model (a heuristic, not a theorem — architecture matters as much as size), so the same per-weight error has a larger relative impact.

**The correct answer:** Do not quantize this model below int8 without QAT. And do not quantize to int4 at all — at some point the precision floor is reached; beyond it, even advanced methods cannot reliably recover accuracy for a model this compact.

**The signal to look for:** When *every* quantization approach at a given precision produces unacceptable accuracy drops, the model is telling you that the precision floor has been reached. The fix is not a better algorithm — it is a different precision or a different (larger) model whose representations are more redundant and therefore more compressible.

**Fix / Mitigation:**

- Raise precision (int8 instead of int4), or use mixed precision on the most sensitive blocks (Chapter 12).
- Consider architectural changes: replace ops that produce structural outliers, adjust hidden dimension for better quantization headroom.
- If economics allow, use QAT — as the table above shows, it can recover int8 accuracy at the cost of training compute (Chapter 11).

---

## The Standard Model vs. The Transformer Reality

The five failure patterns above — and the diagnostic machinery that identifies them — were developed against convolutional networks and feedforward architectures. These models have smooth activation distributions, bounded value ranges, and well-behaved weight statistics. The quantization framework from Chapters 1–12 handles them well, and the failure patterns above capture the standard ways things go wrong.

Transformers rewrite the rules. Large language models and vision transformers produce activation distributions with *structural* outliers — not random extremes, but specific channels that consistently fire at 30–100× the magnitude of their neighbors. These outliers are not noise to be clipped or calibrated around. They are a fundamental property of the architecture.

The standard observer (Chapter 9) cannot fix this — no choice of percentile eliminates outliers that are structurally necessary. PTQ (Chapter 10) collapses entirely. QAT (Chapter 11) is theoretically possible but economically prohibitive for 70-billion-parameter models. The diagnostic patterns above still apply, but the dominant failure mode in transformers — outlier explosion at architectural scale — requires a different class of solutions entirely.

The next four chapters explore this frontier: why transformers break (Chapter 14), how to redistribute outlier difficulty (Chapter 15), why quantizing only weights changes the game (Chapter 16), and how to choose weight values more intelligently than naive rounding (Chapter 17).

---

## Conceptual Consolidation

Quantization failures have names. Outlier explosion, residual ghost, silent fallback, calibration drift, scale mismatch, and unquantizable-model behavior are not exotic edge cases — they are the standard failure modes that account for the vast majority of quantization accuracy and performance regressions.

When a quantized model misbehaves, the question is not "why did quantization fail?" It is: which specific pattern is this, and which invariant from the foundational chapters was violated? The answer determines the fix.

**Pattern interactions.** These patterns are not independent. Fixing Outlier Explosion by widening the observer range can worsen Resolution Collapse in normal-range layers. Aligning scales at residual merges (fixing Residual Ghost) can widen the shared domain and introduce Budget Waste. Adding rescale ops to fix Scale Mismatch adds boundaries that create Fusion Loss. Effective quantization debugging means monitoring for secondary patterns introduced by each fix.

---

## Pattern Index

| Pattern | Symptom Type | What to Inspect | Primary Fix Lever |
|---|---|---|---|
| Outlier Explosion | Accuracy (uniform degradation) | Activation histograms, p99.9/max ratio | Observer choice, per-channel/groupwise, outlier transforms (Ch.15) |
| Residual Ghost | Accuracy (depth-scaled, fine features) | Scale params at merge points, rescale count | Scale alignment, fusion, merge-point mixed precision (Ch.7, Ch.8) |
| Silent Fallback | Throughput (slower than float) | Backend logs, kernel profiler, dequant pairs | Op replacement, fusion-friendly export, backend verification (Ch.4) |
| Calibration Drift | Accuracy (production-only) | Range exceedance, saturation telemetry | Calibration coverage, dynamic quant (Ch.12), recalibration (Ch.9) |
| Scale Mismatch | Accuracy + latency (at boundaries) | Domain params at elementwise ops, rescale count | Domain alignment, graph restructuring, epilogue fusion (Ch.8) |
| Unquantizable Model | Accuracy (all methods fail) | Per-layer sensitivity sweep, precision floor | Raise precision, mixed precision (Ch.12), QAT (Ch.11), architectural change |
