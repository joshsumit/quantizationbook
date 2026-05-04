# Chapter 19: The FP8 Revolution

## Where Int8 Breaks

Before introducing a new format, it is worth seeing precisely why the old one fails.

Consider a single activation vector from a transformer layer — 16 values from one hidden-state position:

```
[0.12, -0.34, 0.08, 0.91, -0.22, 0.45, -0.67, 1.03,
 0.15, -0.28, 0.53, -0.11, 0.77, -0.39, 0.62, 60.0]
```

Fifteen of these values cluster in the range [-1.1, 1.1]. One outlier sits at 60.0 — a pattern that appears routinely in transformer attention and MLP layers (Chapter 14).

Under int8 per-tensor quantization, the scale is determined by the range:

$$S = \frac{60.0 - (-0.67)}{255} = \frac{60.67}{255} \approx 0.238$$

Each int8 step represents 0.238 in real value. Now look at what happens to the fifteen normal values:

| Real value | Int8 code | Dequantized | Error |
|---|---|---|---|
| 0.12 | 1 | 0.238 | 0.118 |
| 0.08 | 0 | 0.0 | 0.080 |
| 0.91 | 4 | 0.952 | 0.042 |
| 0.45 | 2 | 0.476 | 0.026 |
| -0.34 | -1 | -0.238 | 0.102 |

The values 0.12 and 0.08 — a 50% difference — map to adjacent grid points, barely distinguishable. The entire range [-1.1, 1.1] — where 94% of the data lives — gets only ~9 distinct int8 codes out of 256. The remaining ~247 codes are allocated to the range [1.1, 60.0], where a single value exists.

One outlier has captured the grid. The fifteen values that carry the model's signal are crushed to a handful of codes. *This is the problem that FP8 solves.*

---

## The Grid Is No Longer Uniform

### FP8 by Counting: A Number Line Before Theory

Before defining formats or terminology, look at what an FP8 grid actually looks like near a few landmark values:

**Near 0.1:** representable values include 0.0938, 0.1016, 0.1094, 0.1172 — spaced ~0.0078 apart.

**Near 1.0:** representable values include 0.875, 0.9375, 1.0, 1.0625, 1.125 — spaced ~0.0625 apart.

**Near 10.0:** representable values include 9.0, 10.0, 11.0, 12.0 — spaced ~1.0 apart.

**Near 60.0:** representable values include 56.0, 60.0, 64.0 — spaced ~4.0 apart.

The spacing grows as the values grow. Near zero, markings are dense. Near 60.0, markings are sparse. Int8 is a ruler with identical millimeter marks from end to end. FP8 is a ruler where the marks crowd together near zero and spread apart toward the edges.

### Revisiting the Outlier Example

Apply FP8 (E4M3) to the same 16-value activation vector:

| Real value | FP8 nearest | Step size at this magnitude | Error |
|---|---|---|---|
| 0.12 | 0.1172 | 0.0078 | 0.003 |
| 0.08 | 0.0781 | 0.0078 | 0.002 |
| 0.91 | 0.875 | 0.0625 | 0.035 |
| 60.0 | 60.0 | 4.0 | 0.0 |

The values 0.12 and 0.08 — indistinguishable under int8 — are now clearly separated (0.1172 vs 0.0781). The outlier at 60.0 is still representable, just at coarser precision. FP8 did not clip it; it gave it a large step size instead. The small values got the fine resolution they need; the outlier got a "good enough" approximation.

No per-channel scaling, no SmoothQuant, no engineering heroics — the non-uniform grid handled this naturally.

*If you remember one thing from this chapter:* Int8 gives every part of the range the same precision. FP8 gives more precision to small values (where most neural network data lives) and less to large values (where outliers live). The total number of representable values is similar (~240 for FP8 vs 256 for int8) — but FP8 allocates them according to where the data actually is.

### Updating the Mental Model

Chapter 2 framed quantization as a "representational constraint" — a fixed grid imposed on continuous values. That framing remains correct: 8 bits is still 8 bits, at most ~240 distinct values. But the *character* of the constraint changes.

A uniform grid (int8) is a blunt instrument applied identically everywhere. A non-uniform grid (FP8) is a shaped instrument that allocates its budget according to where values actually fall. The total budget is the same; the allocation is smarter.

Think of it as a budget of 240 boxes. Int8 gives you 240 identical boxes — same size, evenly spaced. FP8 gives you 240 boxes where roughly 150 are tiny (for the common small values near zero) and roughly 10 are large (for the rare outliers). Same total count of boxes; different sizes arranged to match how neural network values are actually distributed.

---

## When to Reach for FP8 vs Int8

Before diving into the mechanics of FP8, it helps to know when each tool is the right choice. If you know the *purpose* of the tool, the technical details that follow will have a place to land.

**Int8 wins when:**
- Distributions are bounded and approximately uniform (CNN activations, post-ReLU values).
- The hardware has int8 support but no FP8 support (most edge devices, older GPUs).
- A full integer pipeline is needed (no float compute available).

**FP8 wins when:**
- Distributions have outliers or heavy tails (transformer activations, pre-softmax scores).
- Dynamic range spans multiple orders of magnitude (gradients, KV-cache values).
- The hardware supports native FP8 tensor operations (H100+, MI300X+).

**Quantitative comparison** on a transformer linear layer with activation outlier ratio 50:1:
- Int8 per-tensor: ~8 effective grid points for normal channels (catastrophic).
- Int8 per-channel: ~128 effective grid points (acceptable but expensive in metadata).
- FP8 E4M3 per-tensor: ~30 effective grid points for normal channels (good — the non-uniform grid naturally gives them more resolution).

FP8 does not eliminate the need for SmoothQuant or per-channel quantization in all cases. But it raises the baseline — a naive per-tensor FP8 quantization often performs comparably to a carefully calibrated per-channel int8 quantization, with less engineering effort.

---

## How FP8 Encodes a Number

### Sign, Exponent, Mantissa: The Three Parts

An FP8 number packs three fields into a single byte:

| Field | What it answers | Analogy |
|---|---|---|
| **Sign** (1 bit) | Is this positive or negative? | — |
| **Exponent** (E bits) | What power-of-2 range are we in? | Which *floor* of a building |
| **Mantissa** (M bits) | Where exactly within that range? | Which *room* on that floor |

The exponent selects a power-of-two “zone” — for example, “between 0.5 and 1.0,” or “between 8 and 16.” The mantissa pinpoints a position within that zone. More mantissa bits mean more positions per zone (finer precision). More exponent bits mean more zones (wider range).

The building analogy: more floors = wider range. More rooms per floor = finer precision within each floor. FP8 gives you 8 bits total to split between floors and rooms.

### Quantizing One Value: FP8 vs Int8

Take the value 0.352 and quantize it in both formats.

**FP8 E4M3** (4 exponent bits, 3 mantissa bits):
1. The value 0.352 falls in the zone [0.25, 0.5] (exponent = \\(2^{-2}\\) to \\(2^{-1}\\)).
2. The 3-bit mantissa provides 8 evenly spaced positions within this zone: 0.25, 0.2812, 0.3125, 0.3438, 0.375, 0.4062, 0.4375, 0.4688.
3. Nearest representable value: **0.3438**. Error: \\(|0.352 - 0.3438| = 0.0082\\).

**Int8** with a narrow range [-1, 1] (no outliers): step size \\(= 2/255 \approx 0.00784\\). Nearest value to 0.352: ~0.3490. Error: ~0.003.

**Int8** with an outlier-forced range [-1, 60]: step size \\(\approx 0.238\\). Nearest value to 0.352: 0.238 or 0.476. Error: ~0.114.

For the narrow range, int8 wins slightly — its uniform step is fine enough. But the moment an outlier forces the range wider, int8's error for 0.352 jumps to 0.114 while FP8 stays at 0.0082. FP8's precision for small values does not degrade when large values exist in the same tensor. *This is why FP8 matters for transformers.*

### Where Did the Other 16 Values Go?

An 8-bit field can represent \\(2^8 = 256\\) distinct bit patterns. But the FP8 comparison table lists only 240 representable values — fewer than int8's 256. The remaining 16 bit patterns are reserved for *special values* — representations of NaN (Not a Number) and, in some variants, infinity. This means FP8 does not have more values than int8; it has *differently distributed* values. The advantage is not more values — it is smarter placement. Different FP8 variants (OCP FP8 vs FNUZ) handle the specifics differently, but the practical effect is the same: ~240 usable values for actual numbers.

---

## The Two FP8 Formats

> **📊 INSERT DIAGRAM: Int8 vs FP8 Grid Point Distribution**
>
> Two number lines (0 to 4), showing where representable values fall:
>
> ```
> Int8 (uniform grid, range [0, 4], 256 levels):
> |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
> 0    0.5    1.0    1.5    2.0    2.5    3.0    3.5    4.0
> Step size: 0.0157 everywhere (same density at 0.1 and at 3.9)
>
> FP8 E4M3 (non-uniform grid, same range):
> |||||||||||||||||||||||||||         ||||||||       ||||      |||
> 0    0.5    1.0    1.5    2.0    2.5    3.0    3.5    4.0
> Dense near 0 (step ~0.002)   →   Sparse near 4 (step ~0.5)
> ```
>
> Annotations:
> - "Int8: same precision everywhere — wastes codes on large values that rarely appear"
> - "FP8: more precision where values are small and dense, less where they are large and rare"
> - "For a bell-curve distribution centered near 0, FP8 is a better match"
> - Overlay a typical activation distribution (bell curve near 0) to show WHY non-uniform is better

With the sign/exponent/mantissa mechanics established, the two industry-standard FP8 formats become a straightforward budget allocation choice. Both are formalized in the OCP (Open Compute Project) OFP8 specification.

### E4M3 — More Precision, Less Range

4 exponent bits, 3 mantissa bits.

- **Dynamic range:** ~±448 (much wider than int8's ±127).
- **Precision near zero:** finest step \\(\approx 2^{-9} \approx 0.00195\\).
- **Precision near 1.0:** step size ~0.0625.
- **Values per exponent zone:** \\(2^3 = 8\\) distinct positions.
- **Use case:** Weights and activations during inference. The 3-bit mantissa gives enough precision per zone for forward-pass computations.

### E5M2 — More Range, Less Precision

5 exponent bits, 2 mantissa bits.

- **Dynamic range:** ~±57,344 (extremely wide).
- **Precision near zero:** finest step \\(\approx 2^{-16} \approx 0.0000153\\).
- **Precision near 1.0:** step size ~0.25 (coarser than E4M3).
- **Values per exponent zone:** \\(2^2 = 4\\) distinct positions.
- **Use case:** Gradients during training (explained in the training section below).

### Why Two Formats?

The split is driven by what each tensor needs most:

- **Weights and activations** need precision — distinguishing 0.35 from 0.40 matters for model accuracy. Range beyond ±448 is rarely needed. → E4M3.
- **Gradients** need range — gradient magnitudes routinely span 6+ orders of magnitude, from \\(10^{-7}\\) to \\(10^{1}\\). Whether a gradient is 0.00312 or 0.00375 matters less than whether it is representable at all. → E5M2.

**Worked example: why E5M2 is necessary for gradients.** Consider gradient magnitudes across one layer during training:

$$[0.0000003,\; 0.00001,\; 0.0015,\; 0.025,\; 0.8,\; 12,\; 450,\; 6000]$$

Smallest: \\(3 \times 10^{-7}\\). Largest: \\(6 \times 10^{3}\\). Ratio: \\(2 \times 10^{10}:1\\).

- Int8 (range ±127): smallest representable nonzero = \\(S\\) (one step). To cover 6000: \\(S = 6000/127 \approx 47.2\\). The value 0.0000003 rounds to 0. In fact, anything below 47.2 rounds to 0 or ±47.2. Seven of the eight gradients are lost.
- E4M3 (range ±448): max is 448, cannot even represent 6000 without an enormous per-tensor scale. Even with scaling, min representable nonzero \\(\approx 2^{-9} = 0.00195\\) — the gradient \\(3 \times 10^{-7}\\) underflows to zero. Four of eight gradients are unrepresentable.
- E5M2 (range ±57,344): easily covers 6000. Min representable nonzero \\(\approx 2^{-16} \approx 1.5 \times 10^{-5}\\). The gradient \\(3 \times 10^{-7}\\) still underflows — but with loss scaling (multiply loss by \\(10^4\\)), it becomes \\(3 \times 10^{-3}\\), well within range. Six of eight gradients are representable without scaling; all eight with loss scaling.

This is why E5M2 is mandatory for gradient storage and E4M3 is insufficient.

This is not an arbitrary design choice. It is a direct consequence of what the data looks like at each stage of computation.

| Property | Int8 | E4M3 | E5M2 |
|---|---|---|---|
| Grid type | Uniform | Non-uniform | Non-uniform |
| Dynamic range | ±127 | ±448 | ±57,344 |
| Values near zero | Same step as everywhere | Fine step | Very fine step |
| Values near max | Same step as everywhere | Coarse step | Very coarse step |
| Representable values | 256 | 240 | 240 |

**Variant note:** FP8 encodings exist in a few variants (e.g., OCP FP8 vs FNUZ). The exact number of finite values and treatment of special values (Inf/NaN) depends on the variant. This chapter uses the common E4M3/E5M2 ranges (±448, ±57,344) as implemented in major accelerator stacks.

---

## Why FP8 Handles Outliers Better

The outlier example at the start of this chapter showed the result. This section explains the mechanism and its limits.

### The Mechanism: Per-Value Adaptation

Under int8, a single scale \\(S\\) governs the entire tensor. Every value — small or large — gets the same step size. If the tensor contains one value at 60.0 and thousands near 0.1, the step size is set by 60.0, and the values near 0.1 are crushed.

Under FP8, each value effectively gets its own step size through the exponent. A value near 0.1 lives in a low-exponent zone with a step of ~0.0078. A value near 60.0 lives in a high-exponent zone with a step of ~4.0. There is no single scale that must accommodate both — the format's structure handles the range variation internally.

The outlier at 60.0 is represented at coarse resolution, but it is represented — no clipping. The values near 0.1 retain fine resolution — no budget waste. This is the core advantage.

### FP8 Does Not Remove Scaling

FP8 is smarter than int8, but it is not magic. It still needs a scaling factor — the \\(S\\) from Chapter 3 — to position data within the representable range.

Think of it as a telescope. FP8 has better lenses than int8 (non-uniform resolution that matches real data distributions). But you still have to point the telescope at the right part of the sky. If your data lives in [0, 1] but the FP8 range covers [0, 448], most of the representable values are allocated to zones the data never visits. A scaling factor shifts the data into the FP8 "sweet spot."

Practical FP8 execution uses tensor- or block-level scaling factors to keep values within the representable range. Block scaling (e.g., MXFP8 on Blackwell) exists precisely because per-tensor scaling can still be too coarse.

### FP8 Reduces Sensitivity to Bad Scaling

The crucial difference: a suboptimal scale degrades FP8 *less* than it degrades int8.

Under int8, if the scale is 10% too wide, *every* value loses 10% of its available codes — the loss is uniform across the range. Under FP8, a 10% scale error shifts values into slightly different exponent zones, but the exponential spacing provides a form of built-in range adaptation. Small values still get fine resolution; large values still get coarse resolution. The degradation is gentler.

This is why naive per-tensor FP8 often matches carefully calibrated per-channel int8 — FP8 is more forgiving of imprecise scaling.

*Canonical categories: FP8 reduces Distribution Mismatch / Budget Waste by allocating more codes near zero where values concentrate. It reduces Tail Clipping risk because the wider dynamic range (especially E5M2) means fewer values saturate. FP8 is less brittle than int8 under suboptimal scaling, reducing Calibration Mismatch sensitivity — but it does not eliminate the need for scale computation. Cumulative Rounding Noise still exists as repeated FP8 casts; it is a different error surface than int8 requantization.*

---

## FP8 on Modern Hardware

### Why FP8 Requires New Hardware

An FP8 number and an int8 number are both 8 bits, but they are encoded differently — sign/exponent/mantissa vs a plain integer. An int8 chip cannot read FP8 data. The bit pattern means something entirely different under each format. Using FP8 is not a software trick you can apply to existing hardware. It requires new "math engines" (tensor cores, matrix cores) inside the chip that understand the floating-point encoding.

This is what makes FP8 a "revolution" rather than just a new number format. Companies like NVIDIA, AMD, and Intel redesigned their accelerators to add FP8 datapaths alongside the existing int8 and FP16 ones. If the hardware does not support FP8 (most edge devices, phones, older GPUs), FP8 is not an option — the format physically cannot execute. This is the primary reason int8 remains dominant for edge deployment.

### Current Hardware Support

FP8 is a hardware-supported format on current datacenter accelerators:

- **NVIDIA H100 (Hopper):** Native FP8 (E4M3 and E5M2) tensor core operations with per-tensor scaling.
- **NVIDIA B200 (Blackwell):** Enhanced FP8 support with higher throughput and MXFP8 block scaling.
- **AMD MI300X:** FP8 support (E4M3/E5M2) in matrix cores.
- **Intel Gaudi2/3:** FP8 acceleration with measurement-based scaling.

### What the Hardware Actually Does

On supported hardware, an FP8 matmul is a native hardware operation — not emulated. The tensor core takes two FP8 input matrices and produces a higher-precision accumulator:

**FP8 × FP8 → FP16 (or FP32) accumulation**

This is analogous to int8's int8 × int8 → int32 accumulation from Chapter 6. The inputs are 8-bit, but the intermediate results accumulate in a wider format to avoid overflow. After the matmul completes, the FP16/FP32 result can be cast back to FP8 for the next layer — a conversion that introduces rounding, just as int32 → int8 requantization does in integer pipelines.

On supported hardware, FP8 typically halves memory bandwidth vs FP16/BF16 and can deliver substantial throughput gains (realized speedups are workload- and kernel-dependent). For transformers — where outliers are the dominant quantization challenge — FP8 is often a better trade-off than int8.

---

## FP8 for Training

### Why Gradients Are Different

In inference, we care about the *current value* of each activation — how precisely 0.352 is represented determines the model's output accuracy.

In training, we care about *how values should change* — the gradients. Gradients tell the optimizer "increase this weight by 0.00037" or "decrease it by 0.00000012." Their magnitudes routinely span 6+ orders of magnitude within a single layer, from near-zero corrections on well-learned features to larger corrections on features the model is still learning.

This is why FP8 training uses two different formats:

- **E4M3** for weights and activations in the forward pass — more precision per zone (8 positions), moderate range (±448). The forward pass needs to distinguish similar values accurately.
- **E5M2** for gradients in the backward pass — less precision per zone (4 positions), but vastly wider range (±57,344). Gradients need to be *representable at all* across a huge dynamic range; whether a gradient is 0.00312 or 0.00375 matters less than whether it underflows to zero.
- **FP32** for master weights and optimizer states — these accumulate tiny gradient updates over many steps and need full precision.

This halves the memory and bandwidth for forward and backward passes compared to FP16 training. On H100, FP8 training achieves near-FP16 accuracy with substantial throughput improvement for large transformer models.

### Loss Scaling: Still Required

Gradients in FP8 E5M2 can underflow — values too small to represent become zero. When a gradient underflows, the corresponding weight stops updating, and learning stalls for that parameter.

*Loss scaling* multiplies the loss by a large factor before the backward pass, which scales all gradients up into the representable range. After the gradient computation, the scale is divided back out. Automatic loss scaling adjusts this factor dynamically — increasing it when underflow is rare, decreasing it when overflow occurs. This adds a feedback loop to the training process, but it is well-automated in modern frameworks.

### What FP8 Cannot Do

FP8 is not a drop-in replacement for FP16. Some operations exceed what 8-bit floating-point precision can handle:

- **LayerNorm, softmax, and loss computation** remain in FP16 or FP32. These operations involve reductions and exponentials where a 3-bit mantissa produces unacceptable numerical error.
- **Optimizer states** (momentum, variance in Adam) stay in FP32. They accumulate tiny increments over thousands of steps — FP8 would round away the updates.
- **Scaling infrastructure** is required — frameworks like NVIDIA Transformer Engine and Intel Gaudi's FP8 flow automate scale insertion, amax tracking, and delayed scaling, but they add engineering surface area.

*If you remember one thing:* FP8 is powerful, not magical. It dramatically reduces memory and bandwidth for the bulk of computation (matmuls), but a layer of FP16/FP32 "scaffolding" around sensitive operations is structurally necessary.

---

## The Broader Shift: From Integer to Floating-Point Quantization

The trajectory of quantization hardware tells a story:

- **2018–2020:** Int8 acceleration (GPU tensor cores, NPU MACs). Quantization means "integer quantization."
- **2020–2023:** Int4 weight-only quantization for LLMs. Integer quantization remains dominant.
- **2023–present:** FP8 hardware arrives. Quantization splits into two paradigms — integer for edge/CNN workloads, floating-point for datacenter/transformer workloads.

This book's framework — scale, zero-point, boundaries, calibration — still applies to FP8. The scale contract (Chapter 3) still exists: FP8 values are scaled to fit the representable range. Calibration (Chapter 9) still determines the scale. The difference is that FP8's non-uniform grid makes calibration less critical — a suboptimal scale degrades FP8 less than it degrades int8, because the exponential spacing provides a form of built-in range adaptation. But FP8 does not eliminate the need for scaling; it reduces the sensitivity.

The field is not converging on a single format. It is diverging into format-per-workload: int8 for edge CNNs, int4 for LLM weight storage, FP8 for datacenter inference and training, and combinations of all three within a single deployment.

---

## Conceptual Consolidation

FP8 replaces the uniform integer grid with a non-uniform floating-point grid that provides more resolution near zero and less near the extremes. This naturally handles the outlier distributions that break int8 in transformers. With native hardware support on current datacenter GPUs, FP8 delivers throughput comparable to int8 with better precision characteristics for heavy-tailed distributions.

**Hardware availability.** FP8 tensor core support is available on NVIDIA H100, B200, and later GPUs, as well as recent AMD Instinct (MI300) accelerators. An older GPU with int8 tensor cores (e.g., A100, T4) cannot execute FP8 operations — the instruction set simply does not exist on that silicon. If your deployment target is pre-H100, FP8 is not an option regardless of its theoretical advantages.

---

## The Quantization Landscape: A Unifying View

Having covered the full arc — from the grid (Chapter 2) through integer pipelines (Chapters 6–8) to transformers (Chapters 14–18) and now FP8 — the picture that emerges is not one neat "quantization story" but a landscape of trade-offs that depends on three independent choices:

| Dimension | Options | Key Trade-off |
|---|---|---|
| **Format** | int8, int4, FP8 (E4M3/E5M2), mixed | Precision vs. range vs. hardware support |
| **Granularity** | Per-tensor, per-channel, per-group, per-token | Accuracy vs. metadata overhead vs. kernel complexity |
| **Strategy** | PTQ, QAT, Dynamic, Weight-only, GPTQ, AWQ, SmoothQuant | Cost (compute/data/expertise) vs. accuracy recovery |

> **📊 INSERT DIAGRAM: The Quantization Decision Space (3-Axis)**
>
> A 3D conceptual diagram (or three side-by-side 2D plots):
>
> ```
> Axis 1 (Format):     int8 ─── int4 ─── FP8 ─── FP4
>                       (safe, broad HW)  (aggressive)  (new HW only)
>
> Axis 2 (Granularity): per-tensor ─── per-channel ─── per-group(128) ─── per-group(32)
>                        (cheapest)                                    (most accurate, most overhead)
>
> Axis 3 (Strategy):   PTQ ─── SmoothQuant ─── GPTQ/AWQ ─── QAT
>                       (free)    (minutes)      (hours)      (days)
> ```
>
> Plot typical deployment points:
> - CNN on edge device: int8 × per-channel × PTQ (bottom-left corner: cheap and effective)
> - 7B LLM serving: int4 × per-group(128) × GPTQ (middle: moderate cost, good compression)
> - 70B LLM serving: int4 × per-group(128) × AWQ (similar, faster to produce)
> - LLM training: FP8 × per-tensor × dynamic (top-right: new hardware, dynamic scaling)
> - Accuracy-critical model: int8 × per-channel × QAT (high cost, best accuracy)
>
> Annotate: "The era of 'one quantization recipe' is over. The choice is now workload-specific."

The key insight from this book: **quantization is not a single technique — it is a design space.** The right point in this space depends on your model architecture, your deployment hardware, your latency budget, and how much engineering effort you can invest. The chapters you have read give you the tools to navigate this space rather than guess.

The question is no longer "int8 or float16?" It is "int8, int4, FP8, or float16 — and at which granularity, for which tensors, on which hardware?" The quantization landscape is now a matrix of formats, each optimal for a specific workload-hardware combination.
