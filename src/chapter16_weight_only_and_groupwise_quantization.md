# Chapter 16: Weight-Only and Group-Wise Quantization

In this chapter, we quantize model weights while keeping activations in higher precision.

## The Memory Problem Returns

In large language model (LLM) generation, inference is primarily memory-bandwidth-bound: the primary bottleneck is loading model parameters from High Bandwidth Memory (HBM) to the processor, rather than arithmetic computation. While techniques like SmoothQuant address activation quantization by mathematically smoothing out outlier channels to enable full $\text{int8}$ execution, a dominant paradigm for LLM serving is **weight-only quantization**. Under this approach, only the model weights are compressed, while activations remain entirely in higher floating-point precision during execution.

This approach addresses the core hardware and memory bottlenecks of LLM serving:

* **Weight Footprint:** A 70-billion-parameter model stored in $\text{float16}$ requires $140\text{ GB}$ of memory. On a hardware platform with a memory bandwidth of $900\text{ GB/s}$, streaming the entire model to generate a single token requires approximately $156\text{ ms}$. Quantizing the weights to $\text{int4}$ reduces the memory footprint to $35\text{ GB}$, dropping the single-token transfer time to approximately $39\text{ ms}$—a linear $4\times$ throughput improvement.
* **Activation Scale during Decode:** During the autoregressive decode phase with a batch size of 1, the activation tensor at any given layer is small, typically shaped $[1, \text{hidden\_dim}]$—typically a few kilobytes. Even in native $\text{float16}$, these tensors consume only a few kilobytes of memory traffic. Consequently, quantizing activations during decode yields negligible bandwidth savings.

While activation data transfer and compute density scale up significantly during the prefill phase or under massive batch sizes (shifting the workload into a compute-bound regime), the weight streaming bottleneck dominates low-batch generation. Therefore, memory bandwidth optimization is maximally realized by targeting the weights alone.

---

## Compute-Bound vs. Memory-Bound: When Weight-Only Applies

Not all workloads benefit from weight-only quantization. The real-world payoff depends entirely on whether inference is *memory-bound* or *compute-bound*.

**Memory-bound workloads** (e.g., LLM token generation at batch size 1): The GPU spends most of its execution cycles loading weights from High Bandwidth Memory (HBM). The arithmetic compute units remain idle while waiting for data. Weight-only quantization reduces the data transfer payload by $2\times$ to $4\times$, directly improving tokens-per-second throughput. This represents the dominant regime in LLM serving scenarios.

**Compute-bound workloads** (e.g., CNNs, large-batch LLM prefill, image models): The arithmetic compute units are fully utilized. The primary bottleneck is processing throughput, not data loading latency. Weight-only quantization still reduces the memory footprint—enabling massive models to fit onto fewer acceleration devices—but it does *not* improve execution throughput. This limitation occurs because the quantized weights must be dequantized back to floating-point precision on-the-fly, running at the speed of a native $\text{float16}$ matrix multiplication. Furthermore, if the dequantization logic introduces significant instruction overhead, it can act as a limiter and degrade overall performance.

| Workload | Bottleneck | Weight-Only Benefit | Full $\text{int8}$ Benefit |
| :--- | :--- | :--- | :--- |
| **LLM Decode** ($B=1$) | Memory bandwidth | High ($2\times$ to $4\times$ throughput) | Limited (activations are negligible) |
| **LLM Prefill** ($B=32$) | Compute | Low (memory footprint savings only) | High (up to $2\times$ compute throughput, hardware-dependent) |
| **CNN Inference** | Compute | Low | High |
| **Batch Embedding** | Compute | Low | High |

This architectural distinction explains why weight-only quantization dominates LLM serving infrastructure but is rarely deployed for vision networks. A ResNet-50 inference pipeline at batch size 32 is entirely compute-bound; switching to full $\text{int8}$ quantization doubles operational throughput, whereas weight-only quantization yields no performance gains.

---

## Weight-Only Quantization

In weight-only quantization, model weights are stored in a low-precision format—typically $\text{int4}$ or $\text{int8}$—while activations remain entirely unquantized. At inference time, the weights are dequantized to $\text{float16}$ on-the-fly immediately before matrix multiplication, allowing the General Matrix Multiply (GEMM) execution to run in native $\text{float16}$ precision.

The structural data flow for a single linear layer proceeds as follows:

1. **Load:** Fetch low-precision (e.g., $\text{int4}$) weights from memory, minimizing memory bandwidth consumption.
2. **Dequantize:** Unpack and scale the weights back to $\text{float16}$ on-the-fly, an operation typically fused directly into the GEMM kernel to avoid intermediate memory round-trips.
3. **Compute:** Perform standard floating-point matrix multiplication ($\text{float16}$ activations $\times$ $\text{float16}$ weights).
4. **Output:** Pass the resulting $\text{float16}$ tensor directly to the next layer without downstream quantization steps.

Under this paradigm, the only source of quantization noise resides within the static weight representation. Because weights are immutable and do not vary based on runtime inputs, this error is fixed and deterministic. This design entirely bypasses input-dependent variance, dynamic range tracking issues, and calibration drift. The quantization error is completely baked into the parameters at model load time.

Weight-only configurations commonly employ symmetric $\text{int4}$ quantization (where the zero-point is fixed at 0) combined with per-group scales. This structural choice simplifies runtime metadata overhead by eliminating the storage and integer alignment logic required for explicit zero-points (asymmetric-scaling overhead).

Consequently, the underlying arithmetic remains exact within the limits of $\text{float16}$ precision. Because the output layer does not require a domain conversion step back to integer precision, the system circumvents the requantization boundaries discussed in Chapter 7. For this reason, the complex cumulative rounding noise budgets typically associated with full integer pipelines are non-factors in weight-only execution.

---

## The Resolution Problem at 4-Bit

Standard $\text{int4}$ quantization provides exactly 16 discrete representation levels. Representing a weight distribution that spans a symmetric range of, say, $[-0.5, 0.5]$ across these 16 levels yields a quantization step size ($S$) of:

$$S = \frac{1.0}{15} \approx 0.067$$

Mathematically, any two distinct weight values that differ by less than $0.067$ collapse into the same integer code. For deep architectures where fine-grained parametric precision dictates model performance—such as large language models—this discrete grid is highly destructive. 

> **Canonical Category: Resolution Collapse**  
> The $\text{int4}$ quantization grid is fundamentally too coarse to preserve the subtle weight distinctions required to maintain model accuracy.

Coarse quantization granularities like per-tensor quantization (where a single scale factor is applied to an entire weight matrix) severely compound this issue. If one sub-region of a weight matrix contains tight values clustered within $[-0.01, 0.01]$ while another sub-region spans a wider range of $[-0.5, 0.5]$, the global per-tensor scale is dictated entirely by the maximum absolute boundary. As a result, the narrow distribution region is allocated perhaps a single quantization grid point, effectively destroying the underlying weight information and architectural signal in that area.

Transitioning to per-channel quantization (assigning an independent scale factor to each output channel, corresponding to a row in a standard `[out_features, in_features]` weight matrix) partially mitigates this destruction by isolating the dynamic range of individual rows. However, intra-row variance within a single channel can still remain high enough to trigger precision dropouts. To solve this, the quantization granularity must be driven down even finer.

---

## Group-Wise Quantization

Group-wise quantization assigns an independent scale factor (and optionally a zero-point) to a fixed sub-segment or *group* of consecutive weights along a single dimension. Instead of using one scale for the entire tensor or one scale per channel, the dimension is partitioned into distinct blocks of size $g$.

Consider a weight matrix of shape $[4096 \times 4096]$, totaling approximately 16.7 million weights.

* **Per-tensor (group size = all):** 1 scale factor. The entire matrix shares a single dynamic range. If any specific region contains extreme outliers, every other region pays the price via coarser resolution.
* **Per-channel (group size = 4096, one per row):** 4,096 scale factors. Each output channel (row) receives an independent range. While this accounts for variance between rows, it fails to capture high variance nested within an individual row.
* **Group-wise (group size = 128):** Each 4096-element row is subdivided into 32 distinct groups of 128 weights. Each individual group is quantized using its own scale factor. This yields a total metadata budget of: 
  $$4096 \times \left(\frac{4096}{128}\right) = 131,072 \text{ scales}$$

### A Detailed Look at Intra-Row Variance

Suppose we evaluate Row 1 of our $[4096 \times 4096]$ matrix across its 32 sub-groups using symmetric 4-bit quantization (spanning 15 quantization steps from center):

* **Group 1 (columns 0–127):** Local weights fall within $[-0.12, 0.15]$, giving a local range of $0.27$. 
  The calculated scale factor is:
  $$S_1 = \frac{0.27}{15} = 0.018$$
  An arbitrary weight value of $0.05$ within this group maps to the integer code:
  $$\text{round}\left(\frac{0.05 - (-0.12)}{0.018}\right) = \text{round}(9.44) = 9$$
  This dequantizes back to:
  $$-0.12 + (9 \times 0.018) = 0.042 \quad (\text{Absolute Error} = 0.008)$$

* **Group 17 (columns 2176–2303):** Local weights fall within a wider range of $[-0.48, 0.47]$, giving a local range of $0.95$. 
  The calculated scale factor is:
  $$S_{17} = \frac{0.95}{15} = 0.063$$
  The same weight value of $0.05$ within this group maps to the integer code:
  $$\text{round}\left(\frac{0.05 - (-0.48)}{0.063}\right) = \text{round}(8.41) = 8$$
  This dequantizes back to:
  $$-0.48 + (8 \times 0.063) = 0.024 \quad (\text{Absolute Error} = 0.026)$$

Because Group 1 isolates its narrow range from Group 17, it achieves $3.3\times$ better precision for the same underlying value. Under global per-tensor quantization, both sub-groups would share a massive matrix-wide range (e.g., $[-0.52, 0.50]$, forcing a scale of $S = \frac{1.02}{15} = 0.068$). At that global resolution, Group 1 would only utilize roughly 4 grid levels instead of its full 15. Group-wise quantization effectively preserves this local precision.

For an ultra-tight group of 128 weights falling completely within $[-0.02, 0.02]$, the step size becomes:

$$S = \frac{0.04}{15} \approx 0.00267$$

Compared to a global per-tensor resolution where $S = 0.067$, this local block's resolution is $25\times$ finer. The parameters are represented with an exactitude matched directly to their local behavior.

> **Canonical Category: Resolution Collapse & Distribution Mismatch** > Group-wise quantization directly addresses Resolution Collapse and Budget Waste by matching each group's scale to its local range, eliminating the phenomenon where quantization codes are wasted on an inflated, globally-set range.

### The Metadata Overhead

These local scales must be packaged and loaded alongside the quantized weights. Storing 131,072 scales in standard $\text{float16}$ precision (2 bytes each) introduces a structural metadata footprint:

$$131,072 \times 2 \text{ bytes} = 262 \text{ KB}$$

Meanwhile, the core weight matrix stored in packed $\text{int4}$ elements occupies:

$$16.7 \times 10^6 \times 0.5 \text{ bytes} = 8.35 \text{ MB}$$

The resulting metadata overhead is approximately $3\%$ of the total weight footprint—a highly acceptable engineering tax given the substantial gains in perplexity and representation accuracy.

### Group Size as an Optimization Knob

Modulating the group size ($g$) acts as an explicit architectural trade-off between mathematical accuracy and hardware memory overhead:

| Group Size ($g$) | Scales per Row | Representation Accuracy | Metadata Overhead |
| :--- | :--- | :--- | :--- |
| **4096** (Per-Channel) | 1 | Lowest | Negligible |
| **128** | 32 | Good | $\sim 3\%$ |
| **32** | 128 | Better | $\sim 12\%$ |
| **1** (Per-Element) | 4096 | Best (Theoretical Maximum) | $> 100\%$ (Defeats the purpose) |

In production environments, group sizes of 128 and 32 represent the sweet spots for hardware accelerators. Today, **group size 128** serves as the dominant structural default for modern $\text{int4}$ LLM deployment configurations.

---
## Error Propagation: Weight-Only vs. Full Integer Chains

Systems architects who have internalized the precision constraints of full integer quantization pipelines (Chapters 6–8)—where explicit quantization boundaries, requantization noise, and error-mitigating operator fusion dominate the design—might expect those same engineering challenges to surface here. They do not.

To understand why these concerns do not carry over, recall the standard integer arithmetic pipeline: 
$$\text{int8} \times \text{int8} \rightarrow \text{int32 accumulator} \rightarrow \text{requantize to int8} \rightarrow \text{downstream layer}$$

Every stage in that conventional integer pipeline introduces a discrete rounding boundary. In contrast, weight-only quantization maintains activations in floating-point precision throughout the entire operational lifecycle. Because there is no integer accumulation step and no low-precision integer output, the entire downstream requantization workflow is completely bypassed.

The hardware and mathematical implications of this architectural bypass include:

* **Elimination of the Integer Accumulator Stage:** The matrix multiplication executes via floating-point units. Accumulation occurs natively in floating-point precision ($\text{fp16}$, or $\text{fp32}$ internally depending on the GPU architecture's tensor core configuration). This removes the $\text{int32} \rightarrow \text{int8}$ downscaling step entirely.
* **Absence of Requantization Boundaries:** Because the GEMM output remains a native $\text{float16}$ tensor, no domain conversion is required. The strict requantization error budgets established in Chapter 7 are mathematically irrelevant.
* **Altered Purpose of Operator Fusion:** In weight-only models, kernel fusion is still heavily utilized, but its role shifts exclusively to systems-level optimization. It minimizes hardware kernel launch overheads and reduces memory-bandwidth traffic by fusing the dequantization logic directly into the GEMM loader; it is not deployed to truncate downstream rounding errors, as no such errors exist.

Consequently, the singular source of quantization noise in this paradigm is the static representation error of the weights themselves (the delta between the uncompressed $\text{float16}$ parameter and its discrete $\text{int4}$ or $\text{int8}$ code). Because this error is immutable and completely decoupled from execution-time inputs, there is no activation calibration step, no data-dependent variance, and zero risk of calibration drift. The complex *Cumulative Rounding Noise* pattern introduced in Chapter 13 simply does not manifest in a boundary-accumulation form during weight-only serving.

> **Canonical Category: Error Insulation** > Weight-only serving architectures remain completely insulated from Cumulative Rounding Noise (boundary form) and Calibration Mismatch risks. The remaining mathematical vulnerability is bounded entirely by Resolution Collapse within the weight grid itself, which is controlled strictly via group size selection.

---

## The KV-Cache: The Secondary Memory Bottleneck

While weight-only quantization systematically mitigates the weight-loading bottleneck during generation, long-context LLM execution introduces a second critical memory bottleneck: the **Key-Value (KV) cache**.

During autoregressive generation, each transformer layer caches the Key and Value tensor projections for all historical tokens in the sequence to avoid redundant recomputation. For a 70B model configuration featuring 80 layers, 8 Grouped-Query Attention (GQA) KV heads, and a head dimension of 128, the KV-cache allocation required per single token is calculated as follows:

$$80 \text{ layers} \times 2 \ (\text{K and V}) \times 8 \text{ heads} \times 128 \text{ dim} \times 2 \text{ bytes (float16)} = 327,680 \text{ bytes} \approx 320 \text{ KB per token}$$

Scaling this context footprint reveals how rapidly memory consumption shifts at runtime:
* **At a context of 8,192 tokens:** $320 \text{ KB} \times 8,192 = 2.56 \text{ GB}$
* **At a context of 128,000 tokens:** $320 \text{ KB} \times 128,000 = 40 \text{ GB}$

The KV-cache scales linearly with both sequence length and batch size, requiring a complete round-trip read from High Bandwidth Memory (HBM) at every sequential decoding step. For deep, long-context context windows, the memory bandwidth traffic required to load the KV-cache can easily rival or surpass the bandwidth cost of streaming the static model weights themselves.


KV-cache quantization directly addresses this runtime expansion. Compressing keys and values into $\text{int8}$ format immediately reduces the memory footprint by $50\%$. Native $\text{fp8}$ formats (explored in Chapter 19) offer identical capacity savings while minimizing precision dropouts, leveraging a non-uniform floating-point grid that natively maps to activation distributions.

However, from a hardware compilation and numerical perspective, KV-cache quantization is fundamentally decoupled from weight-only quantization mechanics:

* **Static vs. Dynamic Lifecycles:** Model weights are static parameters, allowing them to be quantized once offline or at initial model load time. Conversely, KV-cache entries are generated *dynamically* at runtime; each newly appended token appends fresh tensor slices that must be quantized on-the-fly inside the inference execution path.
* **Offline Calibration vs. Online Extraction:** Weight distributions can be fully profiled and calibrated prior to deployment. KV-cache activation distributions are dictated entirely by the active runtime input sequence and cannot be predicted ahead of time. Scale factors must be extracted online, per-head and per-layer, and adjusted continuously as the context length increases.
* **Deterministic vs. Stochastic Noise Budgets:** Weight quantization error remains completely fixed and deterministic throughout production execution. KV-cache quantization error exhibits high variance across different generation sequences, producing distinct activation distributions and shifting quantization noise patterns depending on the content of the prompt.

> **Canonical Category: Runtime State Degradation** > Quantizing the KV-cache reintroduces *Calibration Mismatch* risks (where scale factors must be derived online from an incomplete or highly local context window) and *Tail Clipping* risks (causing saturation clipping if the scaling factor bounds are not aggressively updated as token sequences evolve). This places KV-cache quantization squarely within the dynamic activation quantization paradigm analyzed in Chapter 12.

KV-cache quantization is explored in depth in Chapter 18.

---

## Conceptual Consolidation

Weight-only quantization directly targets the binding constraint of contemporary large language model deployment: memory bandwidth consumption during parameter streaming. Activations are preserved in native floating-point precision ($\text{float16}$ or $\text{bfloat16}$) during the autoregressive decode phase because their tensor footprints are remarkably small, rendering the bandwidth savings of activation compression numerically negligible. 

To overcome the discrete resolution limitations inherent to low bit-widths like $\text{int4}$, group-wise quantization introduces fine-grained precision management. By partitioning rows into isolated sub-segments and assigning localized scale factors, it prevents wide parameter distributions from truncating subtle signal variations. This configuration establishes group size ($g$) as an explicit architectural knob, enabling systems engineers to trade off representation accuracy against metadata memory overhead.

Consequently, the core engineering question for weight-only architecture shifts away from a broad debate over precision format, focusing instead on defining the optimal group size. Resolving this choice requires a careful balancing of the underlying model's architectural sensitivity to quantization noise against the hardware platform's tolerance for metadata storage and memory tracking overhead.



