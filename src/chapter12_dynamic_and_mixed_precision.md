# Chapter 12: Dynamic Quantization and Mixed Precision

## Beyond Static Scales

Static quantization fixes scales before inference — either from calibration (PTQ) or from training (QAT) — and every input is quantized using the same parameters. This works when activation distributions are stable across inputs.

But what if they are not? What if the range of activations varies significantly from one input to the next? A static scale may be too narrow for some inputs (causing clipping) and too wide for others (wasting resolution). Two strategies address this: dynamic quantization, which computes scales at runtime, and mixed precision, which selectively keeps sensitive layers in higher precision.

---

## Dynamic Quantization

Dynamic quantization quantizes weights statically (they do not change between inputs) but determines activation scales at inference time — per-batch or per-token.

Recall from Chapter 9 that in static quantization, observers are *temporary*: they collect activation statistics during a calibration phase, produce fixed scales and zero-points, and are then removed from the model. The deployed model contains only the constants they computed — no observer logic runs at inference time.

Dynamic quantization flips this. The observer's job — scanning a tensor to find its range — becomes a *permanent runtime operation*. For each input, before the quantized computation runs, a lightweight runtime statistic (often absmax or min/max, sometimes per-row or per-token) is computed to derive the activation scale and zero-point. Implementations may use approximations to avoid a full-tensor reduction. The quantized computation then proceeds as usual.

The contrast is sharp:

| | Static quantization | Dynamic quantization |
|---|---|---|
| **Weight parameters** | Fixed before deployment | Fixed before deployment |
| **Activation parameters** | Fixed during calibration, stored as constants | Computed at runtime for each input |
| **Observer logic at inference** | Absent — removed after calibration | Present — runs every layer, every input |
| **Calibration dataset required?** | Yes | No |

The benefit is clear: the scale adapts to each input's actual activation range. An input with activations peaking at 3.0 gets a scale optimized for [-3.0, 3.0]. A different input with activations peaking at 8.0 gets a wider scale. Neither clips. Neither wastes resolution on a range set by a different input's statistics.

The cost is equally clear. Computing the activation scale requires an extra reduction pass over the activation tensor — often implemented as a separate kernel — which must complete before the matmul can run. For each layer in the model, the sequence is:

1. Compute activation range (extra reduction pass)
2. Derive scale and zero-point
3. Quantize activations to int8
4. Execute the int8 matmul
5. Convert output — depending on the regime, results may remain in int8 for subsequent integer ops or be dequantized back to floating-point at the next graph boundary

Steps 1-3 add latency that does not exist in static quantization. For models with many layers, this overhead accumulates. Whether the improved range accuracy outweighs the overhead depends on the model and the hardware.

### When Dynamic Makes Sense

Dynamic quantization is most useful when:

- **Calibration data is unavailable or unrepresentative.** If you cannot obtain inputs that reflect the production distribution, static scales will be wrong. Dynamic scales are always correct for the current input.
- **Input distributions vary widely.** NLP models processing text of varying lengths, streaming models processing shifting data distributions, or any workload where the activation range is not stable across inputs.
- **Weight quantization is the primary goal.** In many LLM serving scenarios, the dominant cost is loading weights from memory (Chapter 1). Weights are quantized statically. Activations may remain in float16 entirely, or be dynamically quantized per-token. In autoregressive decoding, per-token or per-row activation scaling is often the practical dynamic granularity.

**Concrete example — memory-bound vs. compute-bound.** A 70B model with int8 weights (35 GB). At 900 GB/s bandwidth:

- *Batch 1 decode (memory-bound):* weight load = $35 / 900 = 38.9$ ms. Compute for 1 token: ~70 billion MACs at ~500 TOPS = 0.14 ms. The GPU is idle 99.6% of the time — pure bandwidth-bound. Weight-only quantization directly improves tokens/second.
- *Batch 32 prefill:* weight load still 38.9 ms (loaded once, shared). Compute: $32 \times 0.14 = 4.5$ ms. Still bandwidth-bound, but compute is catching up.
- *Batch 256 prefill:* compute = $256 \times 0.14 = 35.8$ ms — roughly equal to memory load. Beyond this batch size, compute dominates, and weight-only quantization no longer improves throughput. Full int8 quantization (weights *and* activations) cuts compute by ~2× via native int8 matmul.

Dynamic quantization changes quantization *parameters* per input; it does not necessarily change the graph's operator set or precision assignments at runtime. This distinguishes it from dynamic *precision selection*, which would switch between int8 and fp16 paths based on runtime conditions.

**Why weight quantization is the primary goal in LLMs.** During autoregressive LLM generation (producing one token at a time), the activation tensor at each layer is shaped [1, hidden_dim] — one row per layer, a few kilobytes. Even in float16, the cost to load activations per inference is tiny. The weights, however, are the same 35–40 GB every token. Reducing weight size directly reduces the memory bandwidth cost per token.

**Dynamic quantization vs failure patterns:**

- *Calibration Mismatch / Calibration Drift*: dynamic scales avoid baking in fixed activation ranges from unrepresentative or shifting calibration data.
- *Tail Clipping*: per-input scale reduces saturation on inputs with larger-than-calibrated ranges.
- *Budget Waste / Distribution Mismatch*: per-input scale avoids over-wide ranges caused by other inputs' statistics.
- *Fusion Loss* (runtime): the extra reduction and conversion kernels can prevent operator fusion — a cost that must be weighed against the accuracy benefit.

---

## Mixed Precision

Not all layers tolerate quantization equally. In a typical model, some layers are robust — quantizing them to int8 costs negligible accuracy — while others are sensitive — quantizing them causes disproportionate accuracy loss.

Mixed precision exploits this variation by assigning different precisions to different layers. Sensitive layers stay in float16 (or float32). Robust layers run in int8. The model becomes a mix of precisions, with each layer operating at the precision its sensitivity demands.

### Identifying Sensitive Layers

The standard approach is *sensitivity analysis*: quantize one layer at a time while keeping all others in float, and measure the accuracy impact of each layer's quantization independently.

Consider a 12-layer model. Quantizing each layer individually and measuring accuracy on a validation set might produce:

| Layer | Accuracy drop when quantized alone |
|-------|------|
| 1 | 2.3% |
| 2 | 0.05% |
| 3 | 0.08% |
| 4 | 0.04% |
| 5 | 0.07% |
| 6 | 1.8% |
| 7 | 0.03% |
| 8 | 0.06% |
| 9 | 0.09% |
| 10 | 0.04% |
| 11 | 0.07% |
| 12 | 2.1% |

Layers 1, 6, and 12 are clearly sensitive. The rest are robust. A mixed-precision policy might keep layers 1, 6, and 12 in float16 and quantize the remaining 9 layers to int8.

The result (illustrative numbers):
- **Uniform int8:** all 12 layers quantized → 4.5% total accuracy drop
- **Mixed precision:** 3 layers float16, 9 layers int8 → 0.3% total accuracy drop
- **Model size:** ~1.3× the fully-quantized version, ~3× smaller than the floating-point baseline

Actual accuracy gains and size ratios depend on which tensors remain floating-point, whether weights dominate the model footprint, and the specific task. The pattern is typical: a small number of sensitive layers accounts for most of the quantization error.

### How to Run Sensitivity Analysis

The procedure is mechanical:

1. **Establish a baseline.** Run the full floating-point model on a validation set and record the accuracy metric (top-1 accuracy, perplexity, F1 — whatever the task requires).

2. **Quantize one layer at a time.** For each layer $i$ in the model, quantize only layer $i$ to int8 while keeping all other layers in floating-point. Run the validation set. Record the accuracy drop $\Delta_i$.

3. **Rank by sensitivity.** Sort layers by $\Delta_i$ descending. Layers with the highest $\Delta_i$ are the most sensitive — they contribute the most error when quantized.

4. **Set a threshold.** Choose an acceptable per-layer accuracy drop — typically 0.1% for classification tasks, 0.5 perplexity points for language models. Layers above this threshold stay in float16. Layers below are quantized to int8.

5. **Validate the combination.** Quantize all robust layers simultaneously and measure the total accuracy drop. The sum of individual drops is an upper bound — the actual combined drop may be lower (errors can partially cancel) or slightly higher (errors can interact across layers).

For the 12-layer example above, steps 2–3 take 12 forward passes over the validation set — roughly 12× the cost of a single evaluation. For a 100-layer model, it takes 100×. This is expensive but mechanical, and it runs only once per model-hardware combination.

### Mixed Precision Is Not a Fallback

Keeping sensitive layers in float introduces precision boundaries in the graph. At every transition between an int8 layer and a float16 layer, values must be converted between formats — an operation that costs bandwidth, adds format-conversion work, and reduces fusion opportunities (Chapter 4). Each such transition is a boundary with its own cost.

Mixed precision is an architectural decision: allocate the precision budget where sensitivity demands it, accept the boundary costs at transitions, and quantize everything else aggressively. It is not a fallback for "quantization didn't work." It is a deliberate design that balances accuracy, size, and throughput.

Mixed precision is the standard remedy for Resolution Collapse concentrated in a small subset of layers. If mixed precision underperforms expectations, suspect Fusion Loss at precision transitions and Silent Fallback where the backend lacks efficient mixed-precision kernels for certain ops.

---

## Conceptual Consolidation

Static quantization fixes scales before inference — simple but rigid. Dynamic quantization adapts scales per-input — flexible but adds per-inference overhead. Mixed precision assigns different precisions to different layers — precise but introduces inter-precision boundaries.

None of these is universally best. The choice depends on the deployment constraints:

**Decision checklist:**

- If the issue is **calibration representativeness** (Calibration Mismatch / Drift) → dynamic quantization.
- If the issue is **a few sensitive layers** (Resolution Collapse in a subset) → mixed precision.
- If the issue is **widespread distribution hostility** (Tail Clipping / Distribution Mismatch across many layers) → QAT.
- If latency overhead from dynamic scale computation is unacceptable → static quantization with better calibration.

Each strategy is a trade-off with quantifiable costs, not a solution to be applied by default.

### Strategy Comparison: What Each Approach Can and Cannot Fix

| Strategy | What It Fixes | What It Cannot Fix | Cost |
|---|---|---|---|
| **PTQ** (Ch.10) | Compact, well-behaved distributions | Hostile distributions, outlier channels | Minutes of calibration |
| **Dynamic Quant** (Ch.12) | Calibration drift, input-dependent ranges | Structural weight issues, throughput limits | Per-inference overhead |
| **Mixed Precision** (Ch.12) | A few sensitive layers with high error | Widespread distribution hostility across many layers | Precision boundaries, larger model size for FP16 layers |
| **QAT** (Ch.11) | Structurally hostile distributions (the model learns to survive quantization) | Models without retraining access; extreme outlier architectures | Days of retraining; needs training data and infrastructure |
| **SmoothQuant** (Ch.15) | Activation outliers (migrates range difficulty to weights) | Weight-side outliers; non-linear ops | Offline calibration; minimal |
| **GPTQ / AWQ** (Ch.17) | Suboptimal weight rounding (redistributes error intelligently) | Activation quantization; runtime overhead | Hours of offline optimization; no retraining |
| **Weight-Only** (Ch.16) | Memory-bandwidth bottleneck for large models at low batch sizes | Compute-bound workloads; activation errors | Dequantization overhead at runtime |
| **FP8** (Ch.19) | Outlier distributions that break int8 grids; training + inference | Legacy hardware; not available on pre-H100 GPUs | Requires H100/B200 or equivalent hardware |
