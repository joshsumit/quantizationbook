# Chapter 4: Hardware Dictates the Rules

## The Core Fact

An int8 × int8 multiply-accumulate — the operation at the heart of every linear layer and convolution — executes on specialized low-precision units built into modern accelerators. Two 8-bit operands are multiplied and accumulated into a wider accumulator (commonly int32) at very high throughput. These units are compact: they skip the exponent handling, normalization, and rounding that floating-point arithmetic requires.

A float32 × float32 multiply-accumulate is a fundamentally more expensive operation. The floating-point multiplier requires circuitry for exponent addition, mantissa multiplication, normalization, and rounding — each step consuming silicon area and energy. Even compared to FP16/BF16, integer paths are simpler and can be packed more densely in silicon, especially when paired with dedicated matrix units (called tensor cores on NVIDIA GPUs, matrix engines on Intel Gaudi, and systolic arrays on Google TPUs — all different names for the same idea: a grid of multiply-accumulate circuits that operate in lock-step to process many values simultaneously).

Inference is memory-bandwidth-bound. But quantization is not just about smaller data. It is about matching data to hardware operations that are physically cheaper to execute. The int8 datapath exists in silicon specifically because the economics demand it.

One detail that will matter in later chapters: int8 × int8 multiplication typically accumulates into an int32 register. Once a tile of dot products is complete, an *epilogue* — scale, round, clamp, and optionally a fused activation — converts the int32 accumulator back to int8 for output. This accumulate-then-requantize cycle is the heartbeat of every quantized inference engine.

---

## What the Hardware Supports

A quantized model does not run on abstract "lower precision." It runs on specific hardware that supports specific operations at specific precisions. Each hardware target — CPU, GPU, NPU, DSP — exposes a fixed set of integer operations it can execute. This set is called its *capability envelope*: the supported ops, data types, layouts, and fusions.

A typical capability envelope for an int8 accelerator might include: convolution, matrix multiplication, element-wise addition, ReLU, max pooling, and concatenation. These operations run in int8 at full speed. Operations outside the envelope — say, a GELU activation, a layer normalization, or a specific attention variant — may not have int8 implementations on that hardware.

When the model hits an operation that is outside the capability envelope, the runtime has two choices: refuse to execute the model, or *fall back* — dequantize the inputs to floating-point (often fp16 or fp32), run the operation in float, and re-quantize the outputs back to int8.

Fallback is not graceful degradation. It is a failure mode.

---

## The Cost of Falling Back

When a single operation falls back to floating-point, the cost is not limited to that operation being slower. The full sequence is:

1. Dequantize inputs from int8 to float (memory write + read in the larger format)
2. Execute the operation in floating-point
3. Quantize outputs back to int8 (another conversion + memory write)

The dequantize-compute-requantize round trip adds memory transactions that would not exist if the operation ran in int8. On a memory-bandwidth-bound system, these additional transactions can dominate the cost.

Consider a model where 90% of operations run in int8 and 10% fall back to float. Naively, you might expect a 10% slowdown. In practice, the fallback operations can make the quantized model *slower than the original floating-point model* — because the float model does not pay dequantization and requantization costs at those boundaries. The quantized model pays for the conversions *and* the float compute.

This is why hardware support is not a nice-to-have. An operation that falls back to float does not merely lose the quantization benefit — it actively makes things worse.

---

## Backend Capability Envelopes Are Not Portable

A model quantized for one backend may behave entirely differently on another. The same model with the same quantized weights can produce different latencies, different accuracy, and different execution patterns on different hardware — because the set of operations each backend supports in int8 is different.

A backend that supports fused Conv-BatchNorm-ReLU as a single int8 operation will execute three layers in one pass: one memory read for the input, one memory write for the output, no intermediate values materialized. A backend that does not support this fusion — and where batch normalization has not been folded into the convolution weights during model preparation — will execute three separate operations: the convolution produces an int32 accumulator that must be converted to int8 and written to memory, then batch normalization reads it back, computes, writes again, and ReLU reads once more. Same model. Same weights. Three additional memory round trips on the second backend.

**Concrete cost.** For an activation tensor of shape [1, 256, 56, 56] in int8 — roughly 0.8 MB — each memory round trip (write + read) costs $2 \times 0.8 = 1.6$ MB of bandwidth. Three extra round trips = 4.8 MB of additional traffic per layer. At 900 GB/s bandwidth, that is ~5.3 µs per layer. Across 50 convolutional layers in a ResNet: $50 \times 5.3 = 265$ µs of pure memory overhead that the fused backend does not pay. If the total inference time is 2 ms, the unfused backend spends 13% of inference just shuttling intermediate values to and from memory.

The practical consequence: a quantized model is not purely a mathematical object. It is a compiled artifact that targets specific silicon. Deploying it on different hardware without verifying the capability envelope is not "testing on a different device." It is running a different execution graph.

---

## Mixed Precision Is Not Free

When some layers run in int8 and others in float16, the model graph contains precision boundaries — points where data must be converted between integer and floating-point formats. Mixed precision introduces format conversions, extra memory traffic, and reduced fusion opportunities at every such boundary.

Even when the hardware can execute both datatypes efficiently on independent compute units, the data must still be converted between formats — int8 values widened to float16, or float16 values quantized to int8 — adding memory transactions at every transition. The cost is often dominated by this memory movement and graph fragmentation rather than the arithmetic of conversion itself.

Mixed precision is a legitimate tool for managing accuracy-sensitive layers, but each precision boundary in the graph has a concrete cost. Treating mixed precision as a free fallback leads to quantized models that are slower than uniform-precision alternatives.

---

## Packing and Memory Layout

Int8 values are four times denser than float32 values in memory. A tensor that occupies 4 MB in float32 occupies 1 MB in int8. This density is not just a storage benefit — it changes how hardware loads data into compute units.

Modern accelerators load data in fixed-width chunks — 128 bits, 256 bits, or wider. A single 256-bit load fetches 32 int8 values or 8 float32 values. The int8 path delivers 4× more useful values per memory transaction. This translates directly to higher utilization of the compute units, because more operands are available per cycle.

But this advantage depends on alignment. Hardware tiled matrix multiplication units expect tensors to be laid out in specific patterns — rows packed into tile-sized blocks that match the hardware's native processing dimensions. If a tensor's dimensions do not align with the tile size, the hardware either pads the data (wasting memory and bandwidth) or falls back to a less efficient execution path.

Quantization changes not just the values in a tensor but the physical layout of that tensor in memory. A quantized model's tensors must be packed and aligned to match the target hardware's expectations. This coupling between quantization and memory layout is invisible to the user — the compiler handles it — but it is a concrete constraint that determines whether the theoretical throughput advantage of int8 is actually realized.

Some accelerators require specific packing formats — for example, tensors laid out in tiled patterns where four int8 channels are interleaved into a single 32-bit word (these layouts go by names like NHWC4 or NC/4HW4 on mobile NPUs and DSPs — the exact names differ by vendor, but the idea is the same: pack multiple small values into one wide word). If the tensor's channel count is not a multiple of 4, the remaining slots must be padded — wasting both memory and bandwidth. Similarly, many tiled matrix units require tensor dimensions aligned to 32 or 64 bytes. A [513 × 4096] weight matrix padded to [544 × 4096] adds 6% overhead in memory and bandwidth before a single computation begins.

The alignment constraint is not academic. On edge devices running NPUs or DSPs, misalignment can cause the runtime to fall back to a less vectorized code path — losing the throughput advantage of packed int8 execution entirely. The compiler may not warn about this; profiling is often the only way to detect it.

---

## Hardware Constraints Predict Failure Patterns

Each hardware limitation maps to a specific quantization failure mode. The failure pattern names in the table below are explained in full in Chapter 13 — this table is a preview to show that each hardware gap has a predictable symptom.

| Hardware Constraint | What It Causes | Failure Pattern |
|---|---|---|
| Operation not in int8 capability envelope | Silent dequantize→float→requantize | **Silent Fallback** — model runs slower than float32 |
| Lack of fused kernel support | Extra requantization boundaries | **Residual Ghost** — error accumulates at merge points |
| Accumulator bit-width too narrow | Overflow during dot products | **Numerical Explosion** — garbage outputs on long sequences |
| Misaligned tensor dimensions | Padding overhead or scalar fallback | **Throughput Collapse** — int8 model is no faster than float16 |
| No per-channel scale support | Per-tensor scale forced by hardware | **Outlier Explosion** — one channel destroys resolution for all |

**How to detect each failure:**

- **Silent Fallback:** Profile with the backend's op-level tracer (e.g., TensorRT `trtexec --verbose`, ONNX Runtime profiling). Look for ops executing in fp16/fp32 that you expected in int8.
- **Residual Ghost:** Compare per-layer output between float and quantized models. Error spikes at merge/add nodes signal missing fusion.
- **Numerical Explosion:** Monitor accumulator outputs or final logits for NaN/Inf. Reduce sequence length or accumulation depth to confirm.
- **Throughput Collapse:** Compare int8 vs fp16 latency. If int8 is not faster, check tensor dimension alignment and padding in the profiler.
- **Outlier Explosion:** Inspect per-channel activation ranges. A ratio above 10:1 between the largest and smallest channel range signals the problem.

When a quantized model misbehaves, check the hardware first. The failure is often not in the quantization parameters — it is in the mismatch between what the model needs and what the silicon provides.

---

## Conceptual Consolidation

A quantized model does not run "in lower precision." It runs on specific hardware that provides specific integer operations at specific throughputs. Every operation outside the hardware's capability envelope falls back to float at a cost that can negate the entire quantization benefit.

Whenever you consider quantizing a model, the first question is not "what precision should I use?" It is: what does the target hardware actually support, and does every operation in my model fall within that envelope?

**Envelope checklist for a new target:**

1. Which ops have int8 kernels? (Convolution, MatMul, element-wise, pooling, normalization?)
2. Which datatypes are supported? (int8, int4, fp16, bf16, fp8?)
3. Is per-channel quantization supported, or only per-tensor?
4. Which fusions are available? (Conv-ReLU, Conv-BN-ReLU, MatMul-Add?)
5. What are the layout and packing requirements? (NHWC, tiled, channel-multiple constraints?)
6. What is the accumulator width? (int16, int32?) And what epilogue operations run before the result is written back?
