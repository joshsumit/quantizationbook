# Chapter 13: Failure Patterns

In this chapter, we diagnose failures arising from quantized weights, activations, and boundary conversions.

## A Diagnostic Atlas

With the foundational elements of quantization established—including the representational grid, the scale contract, hardware constraints, error types, graph boundaries, requantization, fusion, calibration, and optimization strategies—you can systematically diagnose quantization failures. Investigate post-quantization performance anomalies using the following top-down diagnostic sequence:

Quantized model underperforming?
│
├── Is it SLOWER than float? (throughput problem)
│    ├── Check backend logs for dequant pairs ──> Silent Fallback (Pattern 3)
│    └── Check kernel profiler for unfused ops ──> Fusion Loss (Ch.8)
│
└── Is it LESS ACCURATE? (quality problem)
│
├── Uniform degradation across all inputs?
│    ├── Check activation histograms for outlier channels ──> Outlier Explosion (Pattern 1)
│    └── Check per-layer sensitivity ──> Resolution Collapse / Scale Mismatch (Pattern 5)
│
├── Degradation only on production data (not calibration)?
│    └── Calibration Drift (Pattern 4)
│
├── Depth-dependent degradation (deeper = worse)?
│    └── Residual Ghost (Pattern 2) ── check scale alignment at skip connections
│
└── Nothing works (PTQ, QAT, all fail)?
└── Unquantizable Model (Pattern 6) ── raise precision floor

**Annotate:** Start with structural and throughput checks (which are cheap and fast) before diving into accuracy diagnosis (which requires profiling and sensitivity sweeps).

---

## Diagnostic Order

When accuracy drops after quantization and the cause remains unclear, evaluate the system in this specific order:

1. **Graph boundaries**: Export the post-fusion execution graph and count the requantize and rescale nodes to ensure they match architectural predictions.
2. **Silent fallbacks**: Verify that every operation executes natively in int8 on the target backend by inspecting backend logs or kernel profilers to identify unexpected dequantize/requantize pairs.
3. **Scale alignment**: Record domain parameters at each elementwise operator and count inserted rescale operations to ensure input scales match.
4. **Calibration**: Measure and cross-reference exceedance and saturation rates between calibration and production environments to catch data telemetry mismatches.
5. **Per-layer sensitivity**: Isolate the specific layers driving the accuracy drop by sweeping precision per-layer or selectively disabling quantization to measure each block's isolated error contribution.

This triage order prioritizes structural flaws (graph errors, backend capability mismatches) over statistical issues (calibration errors), leaving expensive layer-level debugging as a last resort. Structural problems are cheap to verify and exert an outsized impact on performance. Use this sequence to isolate anomalies, and reference the six core patterns below to identify and resolve the root causes.

---

## Pattern 1: Outlier Explosion

*Canonical category: Distribution Mismatch / Budget Waste; Tail Clipping when outliers are clamped.*

**Symptom:** Accuracy degrades uniformly across all model inputs. No single layer appears catastrophic, but every layer suffers minor precision loss. Output degradation manifests as a general drop in representational resolution rather than a complete functional collapse.

**Root cause:** A small subset of extreme values in the activation or weight distributions forces the quantization scale wide, destroying numerical resolution for the remaining majority of values. This represents a severe representation error where the grid range is too wide to accommodate sparse outliers efficiently.

**Mechanism:** Consider a layer where 99.9% of activations fall within \\([-1.0, 3.0]\\), but a single outlier channel produces values reaching \\(50.0\\). A min-max observer sets the scale to encompass the entire \\([-1.0, 50.0]\\) span—a range of 51 units. This yields an unacceptably large quantization step size:

\\[S = \frac{51}{255} \approx 0.20\\]

Values residing within the common range of \\([-1.0, 3.0]\\) compress into approximately 20 discrete grid points. Consequently, the layer effectively operates at a coarse 4-to-5-bit resolution instead of its intended 8-bit capacity.

**Diagnostic question:** What is the ratio between the 99.9th percentile (p99.9) and the maximum absolute activation value? The p99.9 metric isolates the value below which 99.9% of observations fall, remaining robust against isolated anomalies. A p99.9-to-max ratio below \\(0.1\\) indicates that the maximum value is more than \\(10\times\\) larger than the p99.9 threshold, confirming an outlier explosion.

**Fix / Mitigation:**
* Switch from a min-max observer to a percentile or histogram observer to trade bounded clipping for vastly superior resolution across the common range.
* Implement per-channel quantization for weights, and utilize per-group quantization for activations if the execution backend supports it.
* Apply outlier-handling transforms (such as those outlined in Chapter 15) to redistribute outlier magnitude across adjacent channels.
* Isolate layers with concentrated outlier profiles and preserve them in higher precision via mixed-precision scheduling.

---

## Pattern 2: Residual Ghost

*Canonical category: Cumulative Rounding Noise; may trigger Distribution Mismatch at scale alignment and Fusion Loss at inserted rescale boundaries.*

**Symptom:** Accuracy degrades specifically in architectures incorporating skip connections. The model underperforms on tasks demanding fine-grained feature resolution, while coarse top-level predictions remain intact. This degradation scales directly with network depth.

**Root cause:** The system quantizes the main branch and the residual branch independently using different scale parameters. When these branches merge at an elementwise addition, their discrete rounding errors combine. Mismatched scales force the compiler to insert a rescale operation, adding an extra boundary that injects error at every skip connection.

**Mechanism:** In a network featuring 25 residual blocks, each block introduces an error-merging site. If the paths utilize mismatched scales, each merge point inserts a rescale boundary. Across all 25 blocks, this accumulated error reshapes the intermediate representations, severely degrading the higher layers that depend on subtle activation differences.

**Worked Example (Error Amplification Through Depth):** Consider a 4-layer network where each layer multiplies its input by a weight matrix, simplified here as a scalar multiplier \\(m\\):
* **Layer 1:** Receives input \\(x = 1.000\\) with multiplier \\(m_1 = 1.5\\). Quantization introduces an error of \\(\epsilon_1 = 0.004\\).
  \\[\text{Output} = 1.5 \times 1.000 + 0.004 = 1.504 \quad (\text{True Value} = 1.500)\\]
* **Layer 2:** Receives \\(1.504\\) with multiplier \\(m_2 = 2.0\\) and injects its own error \\(\epsilon_2 = 0.003\\).
  \\[\text{Output} = 2.0 \times 1.504 + 0.003 = 3.011 \quad (\text{True Value} = 3.000)\\]
  The cumulative error grows to \\(0.011\\).
* **Layer 3:** Receives \\(3.011\\) with multiplier \\(m_3 = 1.8\\) and injects error \\(\epsilon_3 = 0.005\\).
  \\[\text{Output} = 1.8 \times 3.011 + 0.005 = 5.425 \quad (\text{True Value} = 5.400)\\]
  The cumulative error grows to \\(0.025\\).
* **Layer 4:** Receives \\(5.425\\) with multiplier \\(m_4 = 2.5\\) and injects error \\(\epsilon_4 = 0.004\\).
  \\[\text{Output} = 2.5 \times 5.425 + 0.004 = 13.567 \quad (\text{True Value} = 13.500)\\]
  The final cumulative error reaches **\\(0.067\\)**.

While each layer's isolated error is tiny (\\(0.003\\) to \\(0.005\\)), the error from Layer 1 undergoes amplification by every subsequent multiplier: \\(m_2 \times m_3 \times m_4 = 2.0 \times 1.8 \times 2.5 = 9.0\\). This factor alone contributes \\(0.036\\) to the final deviation. Quantization errors do not merely accumulate linearly; subsequent weight matrices amplify them. This explains why residual-ghost anomalies scale with network depth and why early layers display acute sensitivity.

**Diagnostic question:** Do the main and residual paths share identical scale and zero-point parameters \\((S, Z)\\) at each residual addition? How many explicit rescale operations exist within the fused execution graph?

**Fix / Mitigation:**
* Enforce strict scale alignment across residual merge points during the compiler or quantizer pass so both branches share an identical output domain.
* Maximize graph fusion to eliminate independent requantization events prior to the merge point.
* If alignment expands the dynamic range excessively, preserve the merge sites in higher precision using localized mixed-precision boundaries.
* Deploy Quantization-Aware Training (QAT) to inject robustness against merge-point quantization noise directly into the model parameters.

---

## Pattern 3: Silent Fallback

*Canonical category: Silent Fallback (runtime pattern); causes boundary explosion and throughput collapse.*

**Symptom:** The quantized model executes slower than anticipated. In severe instances, throughput drops below the original floating-point model's baseline. Accuracy metrics remain perfectly acceptable, but execution latency degrades critically.

**Root cause:** The target hardware's native int8 capability envelope does not support one or more operations in the model graph. The execution runtime silently shifts these unsupported operators to a float32 fallback path, incurring massive dequantize-compute-requantize data conversion overhead at each unexpected boundary.

**Mechanism:** Suppose a model incorporates a GELU activation function that lacks int8 hardware support on the target backend. At each GELU layer, the runtime must dequantize the incoming intermediate activations from int8 to float32, compute GELU in floating-point precision, and requantize the results back to int8. Each individual fallback introduces two memory-format conversions and two requantization graph boundaries. If a 24-layer transformer features GELU after every linear layer, the model adds 48 unplanned boundaries and 48 additional round-trips to memory.

**Concrete Latency Cost:** For a standard \\([1, 4096]\\) activation tensor in int8 (4 KB), each fallback boundary demands:
1. Dequantize int8 \\(\rightarrow\\) float32 (4 KB \\(\rightarrow\\) 16 KB; kernel launch overhead \\(\approx 5\ \mu\text{s}\\)).
2. GELU computation in float32 (\\(\approx 20\ \mu\text{s}\\)).
3. Requantize float32 \\(\rightarrow\\) int8 (16 KB \\(\rightarrow\\) 4 KB; \\(\approx 5\ \mu\text{s}\\)).

This yields an added fallback cost of \\(\approx 30\ \mu\text{s}\\) per layer. If a fully fused int8 Linear-ReLU alternative requires only \\(\approx 25\ \mu\text{s}\\) for computation, the GELU fallback introduces an extra 5 \\(\mu\text{s}\\) overhead per block. Across 24 layers, this accumulates:

\\[24 \times 5\ \mu\text{s} = 120\ \mu\text{s} \text{ of extra latency}\\]

For an optimized model targeting a 2 ms total inference budget, this single unsupported activation function causes a 6% performance penalty while masking itself as a fully quantized graph. Although the stored weights are in int8, the runtime critical path routes through floating-point execution units wrapped in expensive format conversions, yielding worse overall throughput than a native floating-point pipeline.

**Diagnostic question:** Does every operator in the execution graph run natively in int8 on the target hardware backend? Are there hidden dequantize/requantize pairs present in the post-fusion execution log?

**Fix / Mitigation:**
* Replace unsupported operations with backend-compatible approximations or hardware-optimized fused variants (e.g., substituting standard GELU with supported approximations or switching to SiLU).
* Re-export the model graph using fusion-friendly structural patterns, such as ensuring Conv/Linear + activation blocks fuse prior to the quantization step.
* Audit backend kernel selection via explicit profiler logs and block deployment pipelines if any floating-point fallbacks appear in the execution path.

---

## Pattern 4: Calibration Drift

*Canonical category: Calibration Mismatch (initial mismatch) and Calibration Drift (time-varying subtype); often leads to Tail Clipping downstream.*

**Symptom:** The quantized model maintains acceptable accuracy during offline validation but degrades severely when exposed to live production workloads. This drop can manifest immediately upon deployment or surface gradually as the live data distribution shifts over time.

**Root cause:** The calibration dataset does not accurately reflect the statistical distribution of real-world deployment data. The pre-computed quantization scales and zero-points fit the calibration profile but fail when processing actual production inputs.

**Mechanism:** Consider a speech recognition model calibrated using pristine, studio-recorded audio. In production, the model encounters noisy, low-bandwidth cell phone recordings with vastly different dynamic ranges. Live activation values consistently overshoot the static, pre-calibrated grid boundaries, triggering severe saturation clipping at each layer. Because the offline parameters are fixed and immutable, these misaligned ranges persist across every inference step.

**Diagnostic question:** Did the calibration data originate from the exact same statistical distribution as the production workload? Do live production activation ranges frequently exceed the maximum bounds established during offline calibration?

**Fix / Mitigation:**
* Broaden calibration dataset coverage to incorporate diverse production data segments, explicitly stratifying samples by noise levels, hardware capture devices, and user demographics.
* Implement live production telemetry to track saturation rates and range exceedance metrics as actionable invariants.
* In deployment environments prone to high variance or continuous drift, deploy dynamic activation quantization or implement automated, periodic recalibration cycles.

---

## Pattern 5: Scale Mismatch at Boundaries

*Canonical category: Cumulative Rounding Noise (from extra boundaries); may trigger Fusion Loss and Silent Fallback at inserted rescale ops.*

**Symptom:** The final execution graph contains unexpected, redundant requantization boundaries that exceed architectural predictions. Inference latency rises, and accuracy regressions cluster around specific topological regions in the graph.

**Root cause:** Elementwise operations (such as additions or concatenations) that accept inputs from independent branches with differing scales force the compiler to insert rescale operations. Each added rescale introduces an unplanned requantization boundary into the execution graph.

**Mechanism:** Consider a feature pyramid network (FPN) that concatenates feature maps across multiple structural stages. If the quantizer processes each stage independently, the branches develop unique scale parameters. Because concatenation requires all incoming tensors to share a uniform \\((S, Z)\\) domain, the compiler must insert rescale operations on the mismatched branches. Across four pyramid levels, this inserts eight unplanned boundaries, with each boundary injecting fresh rounding noise into the features.

**Diagnostic question:** Do all input branches feeding into an elementwise operator share identical \\((S, Z)\\) scaling domains? How many explicit rescale operations does the post-fusion graph contain?

**Fix / Mitigation:**
* Force a shared quantization domain across all branches feeding into elementwise operators via an explicit scale alignment pass in the quantizer.
* Restructure the network topology to shift concatenation or addition operations to points where the input domains naturally match.
* Fuse inserted rescale operations directly into the epilogues of preceding producer kernels to avoid writing intermediate tensors back to global memory.

---

## Pattern 6: The Unquantizable Model

*Canonical category: Resolution Collapse (dominant); may also involve Silent Fallback and capability envelope limits.*

Certain model architectures resist standard low-bit quantization. In these scenarios, you must recognize that the architecture has hit a precision floor and cannot be quantized further without fundamental structural modifications.

**Canonical Example:** Consider a compact 350-million-parameter language model characterized by:
* LayerNorm blocks placed immediately before attention heads, which isolates and amplifies structural activation outliers.
* Extensive use of GELU activations, which fall outside the native int8 capability envelope of standard backends.
* Full multi-head attention (no Grouped-Query Attention), which maximizes runtime KV-cache memory pressure.
* A narrow hidden dimension window of 1024, which restricts the number of terms in each dot product and reduces the statistical averaging effect that normally dampens quantization errors.

Evaluating standard quantization methods on this architecture reveals a clear precision boundary:

| Method | Target Precision | Accuracy Regress | Operational Outcome |
| :--- | :--- | :--- | :--- |
| PTQ (Standard) | int8 | 12% | Total Model Collapse |
| PTQ + Percentile Observer | int8 | 8% | Unacceptable Quality |
| SmoothQuant + PTQ | int8 | 5% | Highly Degraded |
| Quantization-Aware Training (QAT) | int8 | 1.5% | Functional (Requires 3 GPU-days) |
| PTQ (Standard) | int4 | 38% | Complete Functional Collapse |
| GPTQ | int4 | 18% | Incoherent Outputs |
| AWQ | int4 | 15% | Broken Representation |

**The Diagnosis:** At 350M parameters, the model is too compact to absorb aggressive quantization noise. Its internal representations lack the parameters to provide structural redundancy. Each individual weight carries a higher relative informational load than its counterpart in a 70B model, meaning that equivalent rounding errors cause a much larger relative impact on performance.

**The Engineering Resolution:** Do not attempt post-training quantization on this model below int8 without deploying full QAT. Avoid 4-bit configurations entirely. When every optimization algorithm fails to deliver acceptable accuracy at a target bit-width, the architecture has hit its hard precision floor. To fix this, you must adjust the precision target or switch to a larger model that offers higher representational redundancy.

**Fix / Mitigation:**
* Raise the model's target precision floor (e.g., maintaining an int8 baseline rather than forcing 4-bit) or apply mixed-precision routing to safeguard highly sensitive structural layers.
* Modify the underlying architecture by replacing operators that yield structural outliers or expanding the hidden dimension to provide better quantization headroom.
* If project economics allow, leverage full QAT to recover int8 accuracy at the expense of training compute.

---

## The Standard Model vs. The Transformer Reality

Engineers originally developed these six failure patterns and their accompanying diagnostic procedures using convolutional neural networks and classic feedforward architectures. Those legacy models display smooth activation profiles, strictly bounded value ranges, and predictable weight statistics that align well with the foundational quantization frameworks outlined in Chapters 1–12.

Transformers discard these assumptions. Large language models and vision transformers produce activation distributions marked by severe, *structural* outliers. These are not random statistical anomalies that you can clip or calibrate around; they are persistent channels that fire at \\(30\times\\) to \\(100\times\\) the magnitude of adjacent features, serving as a core mechanism of the architecture's expressive power.

Standard observers fail here: no percentile setting can eliminate outliers that are structurally necessary to preserve performance. Standard PTQ collapses entirely, and full QAT is economically prohibitive for 70-billion-parameter models. While the baseline diagnostic patterns described above still hold true, managing transformer architectures requires an entirely different class of solutions to prevent systemic outlier explosions.

The next four chapters explore this architectural frontier:
* **Chapter 14:** Evaluates why transformers break under standard quantization constraints.
* **Chapter 15:** Details methods to redistribute outlier difficulty across channels.
* **Chapter 16:** Analyzes how weight-only quantization bypasses activation tracking hurdles.
* **Chapter 17:** Introduces advanced algorithms to choose weight values more intelligently than naive rounding.

---

## Conceptual Consolidation

Quantization failures match distinct, recurring profiles. Outlier explosion, residual ghosts, silent fallbacks, calibration drift, scale mismatches, and unquantizable precision floors represent the vast majority of real-world accuracy and throughput regressions. When a model degrades post-quantization, avoid treating it as an arbitrary failure; isolate the specific pattern at play to determine your engineering solution.

**Pattern Interactions:** These failure modes are deeply interconnected. Widening an observer's range to mitigate Outlier Explosion can compress the resolution of normal-range features and trigger Resolution Collapse elsewhere. Aligning scales to eliminate a Residual Ghost can inflate the shared domain and introduce Budget Waste. Inserting rescale operations to resolve a Scale Mismatch adds graph boundaries that cause Fusion Loss. Debugging quantization requires monitoring for secondary failure patterns introduced by your initial fixes.

---

## Pattern Index

| Failure Pattern | Primary Symptom Type | Diagnostic Focal Point | Primary Mitigation Lever |
| :--- | :--- | :--- | :--- |
| **Outlier Explosion** | Accuracy (uniform degradation) | Activation histograms, p99.9-to-max ratios | Observer tuning, per-channel/groupwise scaling, outlier transforms (Ch.15) |
| **Residual Ghost** | Accuracy (depth-scaled loss of fine features) | Scale parameters at merge locations, total rescale count | Scale alignment, kernel fusion, merge-site mixed-precision (Ch.7, Ch.8) |
| **Silent Fallback** | Throughput (latency worse than float baseline) | Backend runtime logs, kernel profiler outputs, dequant pairs | Op replacement, fusion-optimized export, backend capability verification (Ch.4) |
| **Calibration Drift** | Accuracy (degradation isolated to production) | Live range exceedance, saturation telemetry metrics | Expanded calibration coverage, dynamic quantization (Ch.12), automated recalibration (Ch.9) |
| **Scale Mismatch** | Accuracy + Latency (localized to boundaries) | Domain settings at elementwise operators, rescale count | Domain alignment passes, graph restructuring, producer epilogue fusion (Ch.8) |