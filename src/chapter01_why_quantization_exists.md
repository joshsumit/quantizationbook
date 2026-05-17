# Chapter 1: Why Quantization Exists

## What Quantization Is

Quantization is the process of mapping continuous floating-point values to a finite, discrete set of integer values using a mathematically derived scale factor. This mapping allows deep learning models to execute significantly faster and consume far fewer resources when deployed on physical hardware. In standard model development, engineers operate in the high-precision environment of 32-bit floating-point ($\text{FP32}$), which offers over four billion representable numerical levels. While this vast resolution is ideal for capturing the subtle gradients necessary during backpropagation, it creates an unsustainable memory and computational burden during inference.

By replacing dense 32-bit values with coarser 8-bit ($\text{INT8}$) or 4-bit ($\text{INT4}$) representations, quantization collapses the available numerical levels from billions down to just 256 or 16. This transition effectively shrinks the model's memory footprint and accelerates arithmetic execution by routing workloads through specialized hardware datapaths. However, compressing the numerical space introduces a measurable loss in fidelity. The objective of this book is to analyze that loss at a systems level—identifying exactly where precision degrades, how it propagates through silicon, and how to preserve a model's functional integrity under these hardware constraints.

---

## The Bottleneck Is Not Compute

A modern hardware accelerator can execute trillions of arithmetic operations per second, a metric typically measured in $\text{TFLOP/s}$ ($\text{tera-floating-point}$ operations per second, or $10^{12}$ operations per second). Because raw multiplication throughput is rarely the limiting factor during inference, the primary bottleneck becomes the physical challenge of moving data to the compute units fast enough to keep the execution pipelines saturated. When an execution unit spends the majority of its clock cycles idling while waiting for data to arrive from memory, the workload is classified as *memory-bandwidth-bound*. Conversely, when data arrives quickly enough to keep the execution units fully utilized, the workload becomes *compute-bound*. For large language models deployed in production—especially at low batch sizes—inference is heavily dominated by memory traffic rather than computation. 

To visualize this constraint, consider a 7-billion-parameter model stored in unquantized $\text{FP32}$ precision. This single model occupies 28 GB of memory and operates in a weight-streaming regime, meaning a significant fraction of its parameters must be fetched from off-chip memory during every single forward pass. If the underlying hardware features a memory bandwidth of 900 GB/s, simply loading the full model into the processor takes roughly 31 milliseconds before a single matrix multiplication can even begin. Because the actual arithmetic computation takes only a fraction of that time once the data arrives on-chip, inference speed is dictated almost entirely by the speed of data transit across the memory bus.

Quantizing those same 7 billion parameters to $\text{INT8}$ reduces the model's footprint to 7 GB, causing the loading time to drop to roughly 7.8 milliseconds. This $4\times$ improvement occurs because each weight now occupies 1 byte instead of 4, allowing the same physical memory bandwidth to deliver four times as many parameters per second. Moving further down to an $\text{INT4}$ representation cuts the footprint to approximately 3.5 GB and drops the transfer time to roughly 3.9 milliseconds, though this ignores metadata overhead like scale factors and bit-packing layouts. Ultimately, while specialized low-precision units like Tensor Cores or dedicated vector datapaths offer massive arithmetic speedups, the primary performance dividend for large models stems from the drastic reduction in memory bandwidth pressure.

---

## Energy Scales with Data Movement

Arithmetic operations themselves are remarkably inexpensive in terms of energy consumption. On a modern 7nm-class process node, an 8-bit integer multiplication consumes roughly 0.2 picojoules ($\text{pJ}$), whereas a 32-bit floating-point multiplication costs approximately 3.7 pJ. While this represents an $18\times$ energy disparity per operation across identical silicon, neither figure represents the dominant cost of running a model. Moving a single 32-bit value from off-chip Dynamic Random-Access Memory ($\text{DRAM}$) to a compute unit costs roughly 640 pJ, which is over 170 times the energy cost of the $\text{FP32}$ multiplication itself. 

This stark imbalance—where data movement costs two orders of magnitude more than physical computation—is a persistent characteristic of modern silicon architectures driven by the physical realities of line capacitance and the distance separating memory pools from logic gates. When scaled to an enterprise datacenter serving billions of inference requests daily, the operational budget is paid primarily in the electricity required to shuttle parameters across these buses. By compressing representations from 32 bits down to 8 or 4 bits, quantization significantly curtails total memory traffic, making it fundamentally an energy-reduction strategy where arithmetic savings are a welcome, but secondary, benefit.

---

## Hardware Will Not Solve This

It is natural to wonder why hardware vendors do not simply solve this bottleneck by engineering chips with vastly wider memory buses. While memory bandwidth has certainly improved over time, its scaling trajectory has failed to keep pace with the exponential growth of compute performance. Over the past two decades, peak arithmetic throughput across major architectures has surged by roughly 60,000$\times$, whereas off-chip memory bandwidth has crawled forward by a factor of only 100$\times$.

To see this divergence concretely, a typical GPU in the early 2000s might deliver 1 $\text{TFLOP}$ of peak compute alongside 50 GB/s of memory bandwidth, yielding a compute-to-bandwidth ratio of roughly 20 $\text{FLOPs}$ per byte of transferred data. By 2024, a standard datacenter GPU delivers approximately 2,000 $\text{TFLOPs}$ against 3,000 GB/s of bandwidth, shifting that ratio to roughly 670 $\text{FLOPs}$ per byte. Because the compute capacity grew $2,000\times$ while bandwidth only grew $60\times$, modern chips are $33\times$ more compute-rich and relatively more bandwidth-starved than their predecessors. 

This structural gap represents a hard physical asymmetry rather than a temporary engineering delay. Building wider memory buses and scaling faster memory cells runs directly into severe thermal, area, and signal-integrity boundaries that do not scale with the same clean mechanics as transistor density. The resulting divergence ensures that every successive hardware generation can compute significantly faster than it can feed itself. Quantization accepts this reality by optimizing the information density of every single byte transferred across the bus, ensuring that waiting for a hardware-driven salvation remains an unviable architectural strategy.

---

## The Trade-Off Is Real

Despite the immense performance and economic benefits detailed above, quantization is never a free lunch. Constraining a value to a discrete 256-level or 16-level grid inherently discards information, creating a permanent precision loss that cannot be mathematically recovered from the final quantized representation. 

Whether this degradation materially impacts a model's capabilities depends heavily on the underlying architecture, the specific task domain, and the distribution of the values being clamped. While certain models tolerate aggressive quantization with negligible deviations in accuracy, others experience catastrophic degradation. 

Demystifying this variance—and predicting how specific network structures behave under quantization—forms the core technical narrative of the chapters ahead. Furthermore, because weights represent only part of the inference data footprint, subsequent sections will explore how activations and the autoregressive Key-Value ($\text{KV}$) cache interact with memory subsystems during runtime execution.

---

## The Quantization Pipeline at a Glance

Before diving into the details in the chapters ahead, here is the end-to-end picture of what happens when a model gets quantized. Every concept introduced below maps to a specific chapter.

Before diving into the low-level mechanics of discrete mappings, it is valuable to establish an architectural baseline of the end-to-end quantization pipeline. The diagram below illustrates how a single tensor layer transforms during execution, with each stage corresponding directly to deep dives found later in this book.

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


While you do not need to master every stage of this pipeline immediately, by the time you complete Chapter 9, each element of this execution flow will be thoroughly decoupled. By the conclusion of Appendix B, you will possess the tools to trace an arbitrary real number through this entire silicon pipeline and analytically predict its structural error.

---

## Conceptual Consolidation

Quantization exists because modern deep learning inference is inherently constrained by memory bandwidth rather than raw computational speed. Reducing the bit-width of model parameters allows data to move across physical buses faster and with significantly lower energy draw. Because physical hardware limitations ensure that the gap between compute capabilities and memory performance will continue to widen with each generation, quantization has transitioned from an optional optimization to an economic inevitability for large-scale production deployments.
