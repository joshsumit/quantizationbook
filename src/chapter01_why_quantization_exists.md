# Chapter 1: Why Quantization Exists

## What Quantization Is

*Quantization*, in one sentence, is: **mapping floating-point values to a finite set of integer values using a learned scale factor, so the model can run faster and smaller on real hardware.**

In standard model development, we operate in the high-precision environment of 32-bit floating-point (FP32), which offers over four billion representable levels. This vast resolution is ideal for the sensitivity of gradient-based learning, but it creates a massive memory and compute burden during deployment.

Quantization replaces dense 32-bit values with coarser 8-bit or 4-bit representations, collapsing the available numerical levels from billions down to 256 or 16. This transition effectively shrinks the model's footprint and accelerates arithmetic by utilizing specialized hardware paths, though it introduces a measurable loss in fidelity. The objective of this book is to analyze that loss — identifying where it occurs and how to ensure the model maintains its functional integrity under these constraints.

---

## The Bottleneck Is Not Compute

A modern GPU or accelerator can execute trillions of arithmetic operations per second — we measure this in TFLOP/s (tera-floating-point operations per second, i.e. \\(10^{12}\\) operations per second). Raw multiplication throughput is rarely the limiting factor during inference. The bottleneck is getting data to the compute units fast enough to keep them busy.

When the processor spends most of its time *waiting for data* rather than computing, we call this *memory-bandwidth-bound*. The opposite — when data arrives fast enough and compute is the bottleneck — is called *compute-bound*. For large deployed models, especially at low batch sizes, the dominant limiter is memory traffic. In some regimes — small models, large batches, strong cache reuse — inference can be compute-bound, but these are the exception not the rule.

Consider a model with 7 billion parameters stored in float32. That model occupies 28 GB. In practice, large models operate in a weight-streaming regime where a significant fraction of parameters must be fetched from memory during each forward pass. If the hardware's memory bandwidth is 900 GB/s — a realistic figure for a high-end GPU — loading the full model takes roughly 31 milliseconds. That is 31 milliseconds before a single multiply has been performed. The compute itself, once data arrives, takes a fraction of that time.

This means inference speed is dominated by how fast parameters move from memory to silicon. The term for this is *memory-bandwidth-bound*: the processor spends most of its time waiting for data, not computing.

Quantizing the same 7 billion parameters to int8 reduces the model to 7 GB. Loading time drops to roughly 7.8 milliseconds — a 4× improvement — because each weight now occupies 1 byte instead of 4, so the same bandwidth delivers 4× as many weights per second. Quantizing to int4 cuts it to roughly 3.5 GB (ignoring metadata like scales and packing overhead) and roughly 3.9 milliseconds.

For large models, the primary speedup comes from reduced memory traffic. Specialized low-precision units (tensor cores, dedicated int8/int4 datapaths) can add further arithmetic gains, but bandwidth savings usually dominate.

---

## Energy Scales with Data Movement

Arithmetic is cheap. An 8-bit integer multiply costs roughly 0.2 picojoules on modern silicon. A 32-bit floating-point multiply costs roughly 3.7 picojoules — about 18× more energy per operation. (These figures are order-of-magnitude estimates for 7nm-class process nodes. Exact values vary by architecture, voltage, and fabrication process, but the ratios are consistent across published measurements from hardware vendors.)

But neither of these is the dominant cost. Moving a single 32-bit value from DRAM to a compute unit costs roughly 640 picojoules — over 170× the cost of the float32 multiply itself. This ratio — data movement costing two orders of magnitude more than computation — has been consistently measured across process generations and is a consequence of the physical distance and capacitance between memory and compute. The energy budget of inference is dominated by data movement, not computation.

This matters at scale. A datacenter serving billions of inference requests per day pays primarily for the energy to shuttle parameters from memory to processors and back. Cutting the representation from 32 bits to 8 bits reduces the total memory traffic per inference, which typically reduces energy.

Quantization is, at its economic core, an energy reduction strategy. The arithmetic savings are real but often secondary.

---

## Hardware Will Not Solve This

A natural question: if the bottleneck is memory bandwidth, why not build chips with more bandwidth?

Bandwidth has improved. But not at the rate compute has improved. Depending on the chosen endpoints, peak arithmetic throughput has grown orders of magnitude faster than off-chip memory bandwidth — roughly 60,000× versus roughly 100× over the past two decades.

To see the divergence concretely: in the early 2000s, a GPU might deliver ~1 TFLOP of peak compute at ~50 GB/s of memory bandwidth — a compute-to-bandwidth ratio of roughly 20 FLOP per byte. By 2024, a datacenter GPU delivers ~2,000 TFLOP at ~3,000 GB/s — a ratio of roughly 670 FLOP per byte. The compute side grew ~2,000×. The bandwidth side grew ~60×. The ratio of compute-to-bandwidth widened by ~33×, meaning that for every byte of data delivered, the chip can now do 33× more arithmetic than it could two decades ago. The hardware has become dramatically more compute-rich and, relatively, more bandwidth-starved.

This divergence is not a temporary engineering lag — it reflects a physical asymmetry. Building wider memory buses and faster memory cells hits power, area, and signal-integrity limits that do not yield to the same scaling laws that drive transistor density.

The result is a structural gap: every hardware generation can compute faster than it can feed data to the compute units. This gap widens with each generation. Quantization does not close the gap, but it is the most direct way to operate within it — by ensuring that every byte transferred carries as much useful information as possible.

Waiting for better hardware is not a strategy. The physics runs in the wrong direction.

---

## The Trade-Off Is Real

Nothing in the preceding sections should suggest that quantization is free. Representing a value with 256 levels instead of 4 billion levels discards information. That information loss is permanent — the original precision cannot be recovered from the quantized representation.

Whether that loss matters depends on the model, the task, and the specific values being quantized. Some models tolerate aggressive quantization with negligible accuracy loss. Others collapse. Understanding why — and predicting which outcome to expect — is the subject of the remaining chapters.

One final nuance: weights are not the only traffic. Activations and, in autoregressive models, the KV cache also consume bandwidth and memory. Later chapters will show when these become the dominant bottleneck. (The KV cache is the memory that stores intermediate computation results during text generation — explained in detail in Chapter 18.)

---

## The Quantization Pipeline at a Glance

Before diving into the details in the chapters ahead, here is the end-to-end picture of what happens when a model gets quantized. Every concept introduced below maps to a specific chapter.

> **📊 INSERT DIAGRAM: Quantization Mental Model — End-to-End Pipeline**
>
> A horizontal flow diagram showing the full quantization pipeline for a single layer:
>
> ```
> Float32 weights ──→ [Scale & Zero-Point (Ch.3)] ──→ Int8 grid (Ch.2)
>                                                         │
> Float32 activations ──→ [Observer / Calibration (Ch.9)] ──→ Int8 activations
>                                                         │
>                                              ┌──────────┴──────────┐
>                                              │  Int8 × Int8 Matmul │
>                                              │  (Hardware, Ch.4)   │
>                                              └──────────┬──────────┘
>                                                         │
>                                              Int32 Accumulator (Ch.6)
>                                                         │
>                                              [Requantization (Ch.7)]
>                                                         │
>                                              ┌──────────┴──────────┐
>                                              │  Fused Ops (Ch.8)   │
>                                              │  (ReLU, Add, BN)    │
>                                              └──────────┬──────────┘
>                                                         │
>                                              Int8 output ──→ Next Layer (Boundary)
> ```
>
> Annotate each arrow with the type of error it introduces: rounding error at the grid mapping, clipping error at the observer range, and accumulated error at the requantization step. Mark the "boundary" between layers where scale transitions happen.

You do not need to understand every box yet. By the time you finish Chapter 9, every element in this diagram will be fully explained. By the time you finish Appendix B, you will be able to trace a real number through this entire pipeline and predict the output error before running the model.

---

## Conceptual Consolidation

Quantization exists because inference is memory-bandwidth-bound, not compute-bound. Smaller representations move faster and consume less energy. Hardware trends will not resolve this — the gap between compute throughput and memory bandwidth widens every generation.

For large models deployed at scale, the economics often make quantization inevitable. The question is: what can the model afford to lose?
