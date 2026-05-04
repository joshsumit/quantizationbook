# Chapter 17: GPTQ, AWQ, and Offline Calibration

## Better Than Rounding

Weight-only quantization with group-wise scales determines the ranges. But a question remains: once the scales are determined, how is each weight value mapped to its integer?

The naive answer is rounding — each weight independently rounds to the nearest grid point. This is what PTQ does (Chapter 10). For int8, naive rounding works reasonably well because 256 levels provide sufficient resolution for most weight distributions. For int4 — 16 levels — the rounding error per weight is much larger, and the cumulative effect across millions of weights can be severe.

GPTQ and AWQ are algorithms that do better than naive rounding. They choose integer values for each weight in a way that minimizes the effect of quantization error on the layer's output — taking into account how weights interact with each other and with the data.

---

## Why Naive Rounding Is Suboptimal

When a single weight is rounded to the nearest grid point, the rounding error for that weight is minimized. But weights do not operate in isolation. They participate in a matrix multiplication:

$$y = Wx$$

Each output value is a dot product of a weight row with the input \\(x\\). The quantization error in the output depends not on the error of any individual weight, but on the sum of errors weighted by the input values. A small rounding error on a weight that multiplies a large activation contributes more to the output error than a large rounding error on a weight that multiplies a near-zero activation.

Naive rounding ignores this. It minimizes per-weight error, not per-output error. GPTQ and AWQ both minimize per-output error, but through different mechanisms.

---

## GPTQ: Error Redistribution

GPTQ quantizes weights sequentially, using second-order information from calibration activations to redistribute quantization error. When a weight is rounded to its nearest grid point, the rounding error is computed. That error is then *redistributed* to the not-yet-quantized weights in the same row — adjusting them so that the total row output stays as close as possible to the float original.

### The Optimization Problem

GPTQ solves a specific objective. For a single layer with weight matrix \\(W\\) and calibration inputs \\(X\\), find quantized weights \\(\hat{W}\\) that minimize the *layer reconstruction error*:

$$\min_{\hat{W}} \| WX - \hat{W}X \|_2^2$$

This is not per-weight error (\\(\|W - \hat{W}\|\\)) — it is per-*output* error, weighted by the actual inputs the layer sees. A weight that multiplies large activations matters more than one that multiplies near-zero activations.

The Hessian of this objective with respect to the weights is \\(H = 2XX^T\\) — a matrix that captures the second-order sensitivity of the output to each weight. In plain terms: the Hessian tells us how much a small change in each weight affects the output error. A weight that multiplies large activations consistently has a large Hessian value — it is sensitive. A weight that multiplies near-zero activations has a small Hessian value — it is insensitive and can absorb more rounding error. GPTQ uses this sensitivity information to decide where to redistribute the rounding error.

The key approximation: GPTQ processes weights sequentially (often in blocks) rather than jointly optimizing all weights simultaneously. Joint optimization over millions of weights is intractable. The sequential approach, derived from the Optimal Brain Surgeon framework, is an approximation that trades global optimality for tractability — and empirically, the loss is small.

The redistribution uses second-order information from the Hessian, approximated from a small calibration dataset (typically 128–256 samples, though the exact count is model- and calibration-coverage-dependent). The Hessian captures how sensitive the layer's output is to changes in each weight — allowing the algorithm to distribute error preferentially to weights that affect the output least.

### Concrete example

A row of four weights being quantized to int4 with step size \\(S = 0.1\\):

| Weight | Float value | Naive int4 | Naive error |
|---|---|---|---|
| \\(w_1\\) | 0.37 | 0.4 | +0.03 |
| \\(w_2\\) | 1.82 | 1.8 | -0.02 |
| \\(w_3\\) | 0.54 | 0.5 | -0.04 |
| \\(w_4\\) | -0.28 | -0.3 | -0.02 |

**Naive rounding:** each weight rounds independently. Total output error = sum of (weight error × input), uncontrolled.

**GPTQ:** \\(w_1\\) rounds to 0.4 (error = +0.03). The Hessian indicates \\(w_2\\) is the least sensitive to adjustment. GPTQ adjusts \\(w_2\\)'s float value from 1.82 to \\(1.82 - 0.03 \times h_{12}/h_{22} = 1.84\\) (where \\(h_{ij}\\) are Hessian entries) before rounding it. \\(w_2\\) now rounds to 1.8 from 1.84 instead of 1.82 — a slightly different rounding, but one that compensates for \\(w_1\\)'s error in the output.

**Detailed walkthrough with calibration inputs.** Suppose calibration produces average input \\(x = [2.5, 1.0, 0.3, 1.8]\\). The target output for this row:

$$y_{\text{float}} = 0.37 \times 2.5 + 1.82 \times 1.0 + 0.54 \times 0.3 + (-0.28) \times 1.8 = 0.925 + 1.82 + 0.162 - 0.504 = 2.403$$

Naive rounding: \\(y_{\text{naive}} = 0.4 \times 2.5 + 1.8 \times 1.0 + 0.5 \times 0.3 + (-0.3) \times 1.8 = 1.0 + 1.8 + 0.15 - 0.54 = 2.41\\). Error: \\(|2.403 - 2.41| = 0.007\\).

GPTQ processes \\(w_1\\) first. Error = +0.03. The Hessian \\(H = 2XX^T\\) tells us \\(w_2\\) has the lowest sensitivity. GPTQ adjusts \\(w_2\\): \\(1.82 - 0.03 \times (2.5 \times 1.0) / (1.0^2) = 1.82 - 0.075 = 1.745\\). Rounds to 1.7. Then \\(w_3\\): adjusted from 0.54 by the accumulated residual, rounds to 0.5. Then \\(w_4\\): adjusted from -0.28, rounds to -0.3.

GPTQ output: \\(0.4 \times 2.5 + 1.7 \times 1.0 + 0.5 \times 0.3 + (-0.3) \times 1.8 = 1.0 + 1.7 + 0.15 - 0.54 = 2.31\\). Error: \\(|2.403 - 2.31| = 0.093\\).

In this toy example, GPTQ is worse — the redistribution pushed \\(w_2\\) too far. But across thousands of output rows and realistic 4096-wide dot products, the Hessian-guided redistribution statistically produces lower total reconstruction error than naive rounding. The benefit emerges in aggregate, not in any single row.

The process continues left to right. Each weight's rounding error is distributed to subsequent weights. By the end of the row, the total output error is significantly smaller than naive rounding — even though individual weight errors may be larger.

GPTQ processes the entire model layer by layer, each time using the calibration dataset to compute the Hessian. The total runtime is minutes to hours, depending on model size — far cheaper than QAT.

---

## AWQ: Protecting Salient Weights

AWQ takes a different approach. Instead of redistributing error after rounding, it identifies which weight channels are most *salient* — most important for the layer's output — and protects them during quantization.

### The Optimization Problem

AWQ solves a different objective from GPTQ. Instead of optimizing the quantized values directly, it optimizes *per-channel scaling factors* \\(s\\) applied before quantization:

$$\min_{s} \| WX - Q(s \cdot W) \cdot (X / s) \|_2^2$$

where \\(Q(\cdot)\\) is the quantization function (round to nearest grid point) and the scaling/division is per-channel. The key insight: scaling a weight channel by \\(s > 1\\) before quantization gives it more grid resolution in its important range, at the cost of the corresponding activation channel being divided by \\(s\\).

AWQ approximates this by defining channel saliency as the average activation magnitude: \\(\text{saliency}_k = \mathbb{E}[|x_k|]\\), measured over the calibration dataset. High-saliency channels get larger scaling factors. Unlike GPTQ's sequential Hessian-based approach, AWQ's scaling is largely saliency-driven with lightweight tuning — no iterative weight adjustment is needed, making it substantially cheaper to run.

The key observation: not all weight channels contribute equally to the output. Channels that are consistently multiplied by large activations have a disproportionate effect. A small quantization error in a salient channel causes more output degradation than a large error in a non-salient channel.

AWQ uses a calibration dataset to measure activation magnitudes and identify salient channels. It then applies a per-channel scaling: salient channels are scaled up before quantization (so they receive more grid points in their important range) and the corresponding activations are scaled down to compensate.

**Worked example: AWQ saliency-driven scaling.** A layer with 4 input channels. Calibration dataset (100 samples) produces mean activation magnitudes:

| Channel | Mean \\(|x|\\) | Saliency rank | AWQ scale \\(\beta\\) |
|---|---|---|---|
| 0 | 0.5 | 3rd | 0.8 |
| 1 | 3.2 | 1st (most salient) | 2.0 |
| 2 | 0.1 | 4th | 0.3 |
| 3 | 0.8 | 2nd | 1.2 |

Channel 1 (saliency rank 1) gets \\(\beta = 2.0\\): its weights are multiplied by 2.0 before quantization. With int4 step size 0.067, channel 1’s effective step size becomes \\(0.067 / 2.0 = 0.034\\) — 2× finer precision for the most important channel. Channel 2 (\\(\beta = 0.3\\)): effective step size \\(= 0.067 / 0.3 = 0.223\\) — 3× coarser, but channel 2 contributes little to the output (mean activation 0.1). The net output error decreases because precision is allocated proportionally to each channel’s contribution.

This is conceptually related to SmoothQuant (Chapter 15), but the goal is different. SmoothQuant smooths activation outliers to make activations quantizable. AWQ protects salient weight channels to make weight quantization more accurate. Note that AWQ's scaling is per weight *input* channel and compensation is on the corresponding activation channel — distinct from per-output-channel quantization scales.

### How it works

Suppose channel \\(k\\) has high saliency (large activation magnitudes). AWQ scales \\(W_k\\) by a factor \\(\beta > 1\\) before quantization:

$$\tilde{W}_k = \beta \cdot W_k$$

After quantization, the dequantized weight \\(\hat{W}_k\\) has smaller relative error because the scaling placed the weights in a region of the grid with better resolution relative to their original magnitude. The output is compensated by dividing the input:

$$y = \hat{W} \cdot (x / \beta_{\text{per-channel}})$$

The net effect: salient channels receive higher quantization precision at the cost of slightly more error in non-salient channels. Since salient channels dominate the output, the total error decreases.

---

## Algorithm ≠ Format ≠ Runtime

A source of confusion in the quantization ecosystem is conflating three independent concerns:

**Algorithm** (GPTQ, AWQ, naive rounding): determines *how* each weight value is mapped to an integer. This runs once, offline, and produces a set of quantized weights.

**Format** (GGUF, GPTQ-format, AWQ-format, safetensors): determines *how* the quantized model is stored on disk — the layout of weight data, scale metadata, group size information, and any other parameters.

**Runtime** (llama.cpp, vLLM, TensorRT-LLM, ExLlamaV2): determines *what executes* the quantized model at inference time — how weights are loaded, dequantized, and multiplied.

These three are independent:
- GPTQ (algorithm) can produce weights stored in GGUF (format) and executed by llama.cpp (runtime).
- The same GPTQ weights can be stored in a different format and executed by a different runtime.
- AWQ (algorithm) and GPTQ (algorithm) can both target the same format and runtime.

When someone says "I'm using GPTQ," they mean the algorithm was used to choose the integer weight values. They are not describing the format or the runtime. Some communities use "GPTQ" to refer to a storage format; in this book we use it strictly as the offline quantization algorithm. Keeping these three concerns separate prevents a class of confusion that is common in practice.

---

## Both Use Calibration Data

Both GPTQ and AWQ require a small calibration dataset — typically 128–256 samples from a representative corpus. But they use calibration data *differently* from the standard observers in Chapter 9:

- Chapter 9's observers use calibration data to estimate *ranges* — where activations fall, how wide the scale should be. This is *range estimation*.
- GPTQ uses calibration data to compute the *Hessian* — which weights interact most with actual inputs. This is *sensitivity estimation*.
- AWQ uses calibration data to measure *activation magnitudes* — which channels carry the most signal. This is *saliency estimation*.

All three can suffer from calibration drift (Chapter 13, Pattern 4): if the calibration data is unrepresentative of deployment data, the algorithm's decisions will be suboptimal. The same calibration insufficiency tests from Chapter 9 apply — check scale stability across subsets, check range exceedance on held-out data.

**Canonical categories addressed:** GPTQ and AWQ primarily mitigate Resolution Collapse (weights) at int4 by choosing better integer assignments than naive rounding. Secondarily, they address Distribution Mismatch / Budget Waste (weights) when some channels dominate error and need protection. They remain vulnerable to Calibration Mismatch if calibration data is unrepresentative, and do not address activation-side patterns (Tail Clipping, Cumulative Rounding Noise) because activations are not quantized.

**Choosing between GPTQ and AWQ:** GPTQ uses more offline compute (Hessian computation, sequential redistribution) and often achieves higher fidelity. AWQ is cheaper to run offline and provides a strong practical default. Both depend on calibration representativeness. The choice is typically driven by offline compute budget and empirical accuracy on a validation set for the target model.

---

## Conceptual Consolidation

GPTQ and AWQ are offline algorithms that improve upon naive rounding for weight quantization. GPTQ redistributes rounding error across weights using second-order information. AWQ protects salient channels by scaling them before quantization. Both use small calibration datasets and run once, producing quantized weights that are then stored in a format and executed by a runtime — three independent concerns.

**Choosing between GPTQ and AWQ.** GPTQ generally produces slightly more accurate results because it uses full Hessian-guided error redistribution, but it takes longer to run (hours for large models). AWQ is faster (minutes to an hour) and often achieves similar practical quality because it focuses on protecting the channels that matter most. For a first attempt, AWQ is the pragmatic starting point. Switch to GPTQ when AWQ leaves a measurable accuracy gap.

### Deployment Reality: What Breaks in Practice

The theory above assumes a smooth path from algorithm → format → runtime. In practice, each runtime has constraints:

| Runtime | Supports GPTQ? | Supports AWQ? | Common Pitfalls |
|---|---|---|---|
| **vLLM** | Yes (via Marlin/GPTQ kernels) | Yes (via AWQ kernels) | Older GPTQ checkpoints may use incompatible group sizes; verify `quantize_config.json` |
| **TensorRT-LLM** | Yes (via weight-only plugin) | Limited | Custom calibration pipeline; GPTQ checkpoint conversion required |
| **llama.cpp / GGUF** | GPTQ → GGUF conversion available | AWQ → GGUF conversion available | Quality depends on quantization type (Q4_K_M vs Q4_0); not all group sizes supported |
| **ExLlamaV2** | Yes (native GPTQ) | No | Optimized for GPTQ specifically; fastest for GPTQ inference |
| **HuggingFace Transformers** | Yes (via `auto-gptq`) | Yes (via `autoawq`) | Slower than specialized runtimes; useful for validation, not production throughput |

The algorithm–format–runtime separation means you can quantize with GPTQ but deploy on any runtime that reads the resulting format. However, not all runtimes support all group sizes, bit widths, or packing schemes. Always verify that your target runtime can load and execute the specific checkpoint you produced.

These algorithms operate within the framework established throughout this book. They do not change the grid (Chapter 2), the scale contract (Chapter 3), or the hardware constraints (Chapter 4). They improve how weight values are assigned to grid points — making better use of the finite representational budget that int4 provides.

---

## The Decompression Cost

Weight-only quantization requires a *dequantization kernel* that converts int4 weights to float16 before the matrix multiply. This kernel is not free.

For each group of \\(g\\) weights, the kernel must: load the packed int4 data, unpack each 4-bit value from its packed byte, load the group's scale (and zero-point if asymmetric), multiply each value by the scale and add the offset, and write the float16 result to registers or shared memory.

On modern GPUs, this dequantization runs concurrently with compute — the memory bandwidth for loading int4 weights is the bottleneck, and the dequantization arithmetic fits in the gaps. But on less capable hardware — mobile GPUs, NPUs, older data center GPUs — the dequantization kernel can become a compute bottleneck, partially or fully canceling the bandwidth savings.

Smaller group sizes make this worse. Group size 32 requires loading a scale every 32 weights. Group size 128 requires one scale per 128 weights. The metadata loads interleave with weight loads, and at small group sizes, the scale loading overhead becomes measurable.

A practical rule of thumb (must be profiled per target): if the dequantization kernel takes more than ~20% of the total matmul time, the bandwidth savings from int4 are being eroded by decompression cost. This is hardware-dependent.

**Worked example: dequantization overhead.** For a [4096 × 4096] matmul with int4 weights and group size 128:

- Weight data: \\(4096 \times 4096 \times 0.5 = 8.35\\) MB. At 2 TB/s bandwidth: load time \\(\approx 4.2\\) µs.
- Scale metadata: 131,072 scales × 2 bytes = 0.26 MB. Load time \\(\approx 0.13\\) µs.
- Dequant kernel (unpack 4-bit, multiply by scale, store float16): ~8–12 µs (hardware-dependent, includes unpacking and scale lookups every 128 weights).
- Float16 matmul: ~25 µs.
- Total: ~37–41 µs. Dequant fraction: \\(10 / 39 \approx 26\%\\).

At 26%, dequantization is above the 20% threshold — the bandwidth savings from int4 are being partially eaten by decompression. On a faster GPU with better dequant fusion, this drops to ~15%. On a weaker GPU, it can reach 40%. Profile before committing to a group size.

If dequantization dominates, the symptom is effectively a Dequant Launch Overhead pattern — the int4 bandwidth savings are real, but the compute overhead to unpack and convert cancels the gain.

---

## Beyond Integers: NF4 and the Non-Uniform Grid

Everything in this book so far assumes a *uniform* quantization grid — equal spacing between adjacent levels. Int4 divides the range into 16 equal steps. But neural network weight distributions are not uniform. They are often roughly bell-shaped — concentrated near zero with fewer values at the extremes.

NormalFloat4 (NF4), introduced in QLoRA, replaces the uniform grid with a *non-uniform* grid whose levels are placed at the quantiles of a standard normal distribution. The 16 levels are:

$$\{-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0\}$$

These levels are denser near zero (where most weights cluster) and sparser at the tails (where few weights exist). The result: NF4 has lower average quantization error than uniform int4 for bell-shaped weight distributions, because the grid allocates more of its representational budget where the data actually lives.

The trade-off: NF4 requires a lookup table for dequantization instead of a simple multiply-add. This is slightly more expensive on hardware that lacks LUT support. And the non-uniform spacing means NF4 values cannot participate in integer multiply-accumulate — they must be dequantized to float16 before any computation, making NF4 strictly a weight-only format. NF4 cannot be used for activations because activations must remain in a compute-friendly format.

NF4's relationship to this book's framework: it changes the grid (Chapter 2) from uniform to distribution-matched. The scale contract (Chapter 3) still applies — each group has a scale that maps the normalized NF4 values to the actual weight range. The error taxonomy (Chapter 5) still applies — but the rounding error for typical values is smaller because the grid is denser where data is dense.

FP8 formats (E4M3 and E5M2) extend this idea further and are covered in Chapter 19.
