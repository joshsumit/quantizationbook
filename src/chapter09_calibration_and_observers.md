# Chapter 9: Calibration and Observers

## The Parameter Problem

Scale and zero-point are binding contracts (Chapter 3). Every computation in the quantized model depends on their exact values. But how are they determined?

For weights, the answer is straightforward: weights are fixed after training, so their range can be computed directly. But activations change with every input. The range of a layer's activations depends on the specific data being processed. Scale and zero-point for activations must be estimated from representative data before deployment.

This estimation process is called *calibration*. The mechanism that collects the necessary statistics is called an *observer*. Together, they determine whether the quantized model's ranges will match the data it encounters in production — or whether they will be wrong from the start.

### What "Static" Means

This chapter covers *static* quantization — the regime where all quantization parameters are fixed before the model ever runs on real inputs. Both weights and activations are quantized, but the timing is different:

| Component | Values known before inference? | Quantization parameters (scale, zero-point) |
|-----------|-------------------------------|---------------------------------------------|
| **Weights** | Yes — weights are constant after training. | Computed directly from the weight values and baked into the model as int8 integers. |
| **Activations** | No — activation values change with every input. | Estimated during calibration from representative data, then frozen as fixed constants. |

The critical distinction: weight *values* are converted to int8 and stored before deployment — they are literally integers sitting in memory. Activation *values* cannot be pre-converted because they depend on the input. Instead, the *parameters* that will be used to convert them (scale and zero-point) are pre-calculated and frozen. At inference time, the float32 activation values are converted to int8 on the fly using those pre-determined, fixed parameters. No range computation happens at runtime — the hardware simply applies the stored scale and zero-point.

This is what makes it *static*: every quantization parameter in the model is a constant by the time inference begins. (Dynamic quantization, covered in Chapter 12, takes a different approach — it computes activation scales at runtime for each input, at the cost of additional per-inference computation.)

---

## What Calibration Does

### Boundaries: A Quick Reminder

Chapter 6 defined *quantization boundaries* — the points in the computation graph where the numeric domain changes. In a quantized model, every linear layer and convolution takes int8 inputs, accumulates in int32, and must convert back to int8 before the next operator can consume the result. Each of these conversion points is a boundary. A boundary is *not* each individual neuron or each individual activation value. It is the junction between two operators where the scale and zero-point change — typically one boundary per layer (after fusion). A 50-layer model has roughly 50 boundaries; a 3-layer model has roughly 3.

Each boundary needs its own scale and zero-point to perform the int32 → int8 conversion. Those are the parameters that calibration must determine.

### The Mechanism: The Observer

Think of an observer as a passive data-logger attached to a pipe. It does not change the data flowing through — it only watches and records.

An *observer* is a small module attached to a specific quantization boundary in the model's graph. During calibration, it watches the float32 values that flow through that boundary and records a statistical footprint — minimums, maximums, histograms, or running averages, depending on its type (covered in the next section).

Think of it concretely: if a model has three linear layers, there are roughly three boundaries. Calibration attaches one observer at each boundary — three observers total. Each observer silently records the values it sees without altering them, like a sensor on a pipe that measures pressure without impeding flow. After calibration, each produces the scale and zero-point for that specific boundary.

### The Environment: Fake-Quantization Simulation

During calibration, the model runs in float32 — the actual int8 hardware path is not used. But to get the most accurate statistics, some calibration workflows insert *fake-quantization nodes* (fake-quant nodes) at boundaries to simulate the effect of quantization while staying in float32.

A fake-quant node does not actually switch the model to integer math. It takes a float32 value, snaps it to the nearest integer grid point (quantize, using the round-and-clamp logic from Chapter 3), and then immediately converts it back to float32 (dequantize). The result is still a float32 value, but one that carries the rounding and clipping errors that real quantization would introduce (Chapter 5). The model does not "break" — it stays in float32 throughout, and gradients can still flow (this becomes critical in Chapter 11, Quantization-Aware Training).

**Why does this matter for observers?** Without fake-quant nodes, the observers at later layers see pristine float32 values — the ideal signal, undistorted by quantization. But during real quantized inference, those later layers will receive values that have already been rounded and clamped at earlier boundaries. By inserting fake-quant simulation, the data reaching each observer reflects the messy reality of the quantized world, and the observers compute scales and zero-points that account for that noise.

Not all calibration workflows need fake-quant nodes. Simple post-training calibration (Chapter 10) often runs the float32 model unmodified, with observers passively recording pristine values. Fake-quant nodes become essential when accuracy depends on accounting for how quantization error at early layers reshapes the value distributions at later layers.

### The Calibration Process

With these pieces defined, the full process:

1. **Insertion.** Identify every quantization boundary (where the numeric domain changes) and attach an observer — optionally with fake-quant nodes if the workflow requires simulated quantization effects. *Why:* each boundary will need its own scale and zero-point; without an observer there, the boundary has no data to compute them from.
2. **Observation.** Run calibration inputs through the model in float32 mode. Each observer records the values it sees — tracking range, distribution, or both. If fake-quant simulation is active, the data is rounded before it reaches downstream observers, so they see the noise-affected distribution rather than the pristine float32 signal. *Why:* the observers need enough representative data to capture the true statistical footprint at each boundary.
3. **Calculation.** After the calibration run, each observer computes the scale and zero-point that best fits the data it recorded. *Why:* this is where the statistics become the binding contracts from Chapter 3 — the concrete numbers that every subsequent quantized computation depends on.
4. **Freezing and conversion.** The observers (and fake-quant nodes, if present) are removed, the computed parameters are frozen, and the model is converted to quantized execution. *Why:* once parameters are set, they are immutable (Chapter 3). The observers have served their purpose; the model is now ready for integer-only inference.

The calibration dataset is not the training set and not the validation set. It is a separate set of inputs whose only job is to provide representative statistics for range estimation. If these inputs are not representative of the data the model will see in production, the computed ranges will be wrong — and every inference will carry that error (Chapter 3's immutability principle).

The calibration dataset is a first-class engineering decision, not a formality.

---

## What Observers Collect

The previous section described what an observer *is* — a module attached to a boundary that records values. This section covers what different observer types *record*, because the choice of observer directly determines the scale and zero-point estimates, and therefore the accuracy of the quantized model.

### Min-Max Observer

The simplest observer tracks the absolute minimum and maximum values seen across all calibration inputs. The range is set to \\([x_{\min}, x_{\max}]\\).

Consider a layer's activations across 100 calibration images. Values range from -0.8 to 47.2, but 47.2 occurs in a single image — a single outlier channel activation. Under min-max observation:

$$S = \frac{47.2 - (-0.8)}{255} = \frac{48.0}{255} \approx 0.188$$

The step size is 0.188. But 99.9% of values fall in [-0.8, 3.1] — a range of 3.9 units. That range receives approximately \\(3.9 / 0.188 \approx 21\\) grid points out of 256. The remaining 235 grid points are allocated to the range [3.1, 47.2], where almost no values exist.

One outlier has set the scale for the entire tensor. The common values — the ones that determine the model's behavior — are represented with ~21 levels instead of ~256. Resolution is destroyed for the majority to accommodate the extreme minority.
*Accuracy pattern: Observer Misfit — min-max on a long-tailed distribution wastes budget in the empty tail.*
### Histogram / Percentile Observer

A histogram observer builds a distribution of observed values and sets the range to capture a target percentile — typically 99.99% or 99.999%.

For the same layer: the 99.99th percentile falls at 3.1. The observer sets the range to [-0.8, 3.1]:

$$S = \frac{3.1 - (-0.8)}{255} = \frac{3.9}{255} \approx 0.0153$$

Step size is 0.0153 — over 12× finer than the min-max observer. The values in [-0.8, 3.1] now have the full 256 levels of resolution.

The outlier at 47.2 is clipped to 3.1. Its clipping error is \\(47.2 - 3.1 = 44.1\\) — enormous for that one value. But only one value out of hundreds of thousands is affected. The trade-off is explicit: sacrifice one extreme value to preserve resolution for everything else.

This is the rounding-vs-clipping trade-off from Chapter 5, now as a concrete calibration decision. The percentile observer intentionally accepts a small amount of clipping error to dramatically reduce representation error for the majority.

*Accuracy pattern: Tail Clipping (intentional and controlled) — the observer trades known clipping at the extremes for better resolution in the bulk.*

### Moving Average Observer

A moving average observer tracks running averages of min and max values across calibration batches. It smooths out batch-level statistical noise — a single extreme batch does not set the range for the entire model.

This observer is less sensitive to outliers than min-max but does not explicitly reason about the distribution shape the way a histogram observer does. It is a pragmatic middle ground.

---

## Observer Freezing

During calibration, observers update their statistics with each batch of data. At some point, the statistics must stabilize and the parameters must become final. This is *freezing* — the observer stops updating, and its current scale and zero-point become the permanent values for that boundary.

In static quantization (the focus of this chapter), running inference with unfrozen observers is a bug. If the observer continues updating per-input, the scale and zero-point change with every inference — violating the immutability contract from Chapter 3. Every operation downstream of that boundary was compiled assuming specific parameter values. Changing them per-input is like recalibrating a measuring tape mid-measurement: all previously recorded values become inconsistent with the new calibration. The model produces silently incorrect results — no error message, wrong answers.

Freezing is not an option. It is a requirement. (Dynamic quantization, covered in Chapter 12, intentionally recomputes activation scales at runtime — but that is a different regime with different invariants.)

*Accuracy pattern: Mutable Domain — unfrozen observers in static quantization produce silently incorrect outputs and nondeterminism.*

---

## Calibration Is Not Validation

Calibration determines the quantization parameters. Validation measures whether those parameters produce acceptable accuracy.

These are different steps with different data requirements. Calibration data should be representative of the input distribution — it does not need labels. Validation data should be drawn from the deployment distribution — it needs labels (or a quality metric) to measure accuracy.

A useful analogy: calibration is like adjusting a camera's white balance before taking photos (no subject needed, just the lighting conditions). Validation is like reviewing the actual photos to check quality. Doing calibration on training data and validation on training data is like adjusting the camera using studio lights and then complaining the outdoor photos look wrong. The calibration data must reflect the production input distribution. If production inputs have characteristics not present in calibration (different image resolutions, different text lengths, different domain distributions), the ranges will be wrong.

---

## Detecting Calibration Insufficiency

A calibration dataset that is too small or unrepresentative produces unstable scale parameters. This instability can be measured before deployment.

**The variance test:** Run calibration five times, each with a different random subset of the same size (e.g., 128 samples). After each run, record the computed scale at each boundary. Compare the scales across the five runs.

If scales at a given boundary vary by more than 5% across runs, the calibration set is too small — the observer has not converged to a stable estimate. Measure scale variability as \\(\max(S) / \min(S) - 1\\) across the five runs at each boundary.

**Worked example.** Five calibration runs at one boundary produce scales: [0.0501, 0.0498, 0.0515, 0.0492, 0.0505]. Variability: \\(0.0515 / 0.0492 - 1 = 0.047 = 4.7\%\\). Below 5%: stable — the calibration dataset is sufficient for this boundary.

Now consider an unstable case: [0.0501, 0.0450, 0.0520, 0.0390, 0.0600]. Variability: \\(0.0600 / 0.0390 - 1 = 0.538 = 53.8\%\\). The scale swings by over 50% depending on which samples are chosen. At \\(S = 0.060\\), int8 code 127 represents \\(0.060 \times 127 = 7.62\\). At \\(S = 0.039\\), code 127 represents \\(0.039 \times 127 = 4.95\\). A 54% scale swing means the clipping boundary shifts from 4.95 to 7.62 — values between 4.95 and 7.62 would be clipped in one run but not another. Diagnosis: calibration dataset too small. Remedy: increase calibration samples until variability drops below 5%.

The 5% threshold is derived from the relationship between scale error and quantization error: a 5% scale error can shift codes by ~13 grid levels near the ends of the int8 range (where code values are largest), enough to move the clipping boundary by several step sizes and cause measurable accuracy degradation. Smaller errors produce proportionally smaller shifts, especially for values near the center of the range. The remedy is straightforward: increase the calibration set until scales stabilize.

**The range exceedance test:** After calibration, run a separate batch of held-out data (not from the calibration set) through the quantized model. At each boundary, track both float-domain exceedance counts (values outside the calibrated range before quantization) and actual int8 saturation frequency (values clamped to \\(q_{\min}\\) or \\(q_{\max}\\) after quantization).

**Worked example.** A held-out batch produces 50,000,000 activation values across all boundaries. Exceedance count (values outside calibrated range): 50,045 values. Saturation count (values clamped to ±127 after quantization): 48,932 values. Rates: exceedance = \\(50{,}045 / 50{,}000{,}000 = 0.10\%\\), saturation = \\(48{,}932 / 50{,}000{,}000 = 0.098\%\\). Both below 0.1% — calibration is adequate.

If instead: exceedance = 752,000 values (1.5%), saturation = 738,000 (1.48%). The calibration range is too narrow — 1.5% of values are being clipped. At a scale of \\(S = 0.015\\), each clipped value loses at minimum one step (0.015) and potentially much more. Diagnosis: calibration data was unrepresentative (perhaps missing high-dynamic-range inputs), or the observer type is too aggressive (percentile set too tight). Remedy: add more diverse calibration samples, or widen the percentile threshold.

If more than 0.1% of values clip (a heuristic threshold — the right number depends on the model and task), the calibration range is too narrow — either the calibration data was unrepresentative, or the observer type is too aggressive (e.g., percentile with too tight a threshold).

*Accuracy pattern: Calibration Mismatch — the calibration dataset does not represent the production distribution, producing ranges that are wrong from day one.*

**The domain shift test:** Compare the mean and standard deviation of activations at each boundary between calibration data and production data. If the means differ by more than one standard deviation or the standard deviations differ by more than 2×, the calibration data does not represent the production distribution. The computed scales are valid for a distribution the model will not encounter.

These tests are cheap — they require only forward passes through the model with statistics collection, no retraining. Running them before deployment catches calibration drift (Chapter 13, Pattern 4) before it reaches production.

*Accuracy pattern: Calibration Drift — the production distribution shifts after deployment, making originally correct scales invalid over time. Distinct from Calibration Mismatch, which is wrong from day one.*

---

## Conceptual Consolidation

> **📊 INSERT DIAGRAM: Calibration Workflow Timeline**
>
> A horizontal timeline showing the calibration process:
>
> ```
> TRAINING PHASE                    CALIBRATION PHASE              DEPLOYMENT PHASE
> (weeks/months)                    (minutes/hours)                (forever)
>                                         │
> Train model    ───────────────────→ Fuse operators
> in float32                               │
>                                   Insert observers at boundaries
>                                          │
>                                   Run calibration data (100–1000 samples)
>                                          │
>                                   Observers record min/max/histogram
>                                          │
>                                   Freeze observers → fix (S, Z) per boundary
>                                          │
>                                   Quantize weights ───────────→ Deploy:
>                                          │                       int8 weights
>                                   Remove observers               fixed scales
>                                   (they are gone forever)         no observers
>                                                                   no float ops
> ```
>
> Key annotations:
> - "Observers are TEMPORARY — they exist only during calibration"
> - "After freezing, scales are baked into the graph as constants"
> - "Calibration data should represent production data, not training data"

Calibration estimates quantization parameters from data. Observers are the mechanism that collects the statistics. The calibration dataset determines whether the estimated ranges match production data — and because parameters are immutable once set, a bad calibration cannot be corrected at inference time.

When evaluating a quantized model's accuracy, the second question (after counting boundaries) is: were the calibration data representative, and did the observer type match the distribution shape? The choice of observer is fundamentally a choice of which error from Chapter 5 you are more willing to accept. A min-max observer guarantees zero clipping error — every observed value fits inside the range — but if an outlier stretches the range, the scale becomes so coarse that rounding error destroys resolution for the majority of values. A percentile observer deliberately accepts some clipping error at the extremes in exchange for a much tighter scale and far lower rounding error across the bulk of the distribution. Neither is universally correct — the choice is an engineering decision with direct accuracy consequences.
**Observer choice guide:**

- **Min–max**: use only when distributions are bounded and tails are well-controlled. Fast, simple, but vulnerable to outliers.
- **Percentile / histogram**: preferred for long-tailed activations. Tune the percentile (99.99% is a common starting point). Accepts controlled clipping for better resolution.
- **Moving average**: pragmatic middle ground for stable batch statistics with moderate tail sensitivity.

Some deployments calibrate activations per-tensor; others use per-channel or groupwise schemes when the backend supports it. The observer choice and granularity together determine the accuracy ceiling of the quantized model.