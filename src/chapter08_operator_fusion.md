# Chapter 8: Operator Fusion

In this chapter, we quantize intermediate activations and study how fusion changes where quantization happens.

## The Direct Remedy

Requantization at every boundary compounds error â€” and the number of boundaries is fixed by the graph topology. The primary mechanism for reducing that count is *fusion* â€” merging consecutive operators into a single fused operator so that intermediate boundaries disappear entirely.

Fusion is not an optimization in the performance-tuning sense. It is a structural transformation that changes the number of times values are rounded and clamped. Fused and unfused versions of the same model produce numerically different results â€” fusion changes where rounding and clamping occur (and how many times), so the numerics differ even when the underlying real-valued computation is logically equivalent.

*Accuracy pattern: Cumulative Rounding Noise â€” every removed boundary removes one rounding+clamp event. Runtime pattern: Fusion Loss â€” when fusion fails, boundaries remain.*

---

## What Fusion Does

Consider three consecutive operations: Conv â†’ BatchNorm â†’ ReLU.

> **ðŸ“Š INSERT DIAGRAM: Fusion Before and After â€” Conv-BN-ReLU**
>
> Two side-by-side execution flow diagrams:
>
> ```
> UNFUSED (2 boundaries):                    FUSED (1 boundary):
>
> int8 input                                 int8 input
>   â”‚                                          â”‚
>   v                                          v
> [Conv] â†’ int32 acc                         â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
>   â”‚                                        â”‚ Conv+BN+ReLU    â”‚
>   v  â† Boundary 1 (requant)               â”‚ (single kernel) â”‚
> int8                                       â”‚ int32 acc       â”‚
>   â”‚                                        â”‚ â†’ BN (folded)   â”‚
>   v                                        â”‚ â†’ ReLU (fused)  â”‚
> [BN] â†’ int32 acc                           â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜
>   â”‚                                                â”‚
>   v  â† Boundary 2 (requant)                       v  â† Only 1 boundary
> int8                                             int8 output
>   â”‚
>   v
> [ReLU] (clamp, no requant)
>   â”‚
> int8 output
>
> Error: 2 rounding events                   Error: 1 rounding event
> Memory: 3 read/write round-trips           Memory: 1 read/write round-trip
> ```
>
> Annotate: "Fusion eliminates 1 of 2 rounding events and 2 of 3 memory round-trips. The numerics of the unfused and fused paths are different â€” fused is more accurate."

**Unfused execution:**

1. Conv: int8 inputs â†’ int32 accumulator
2. **Boundary 1**: int32 â†’ int8 (requantize)
3. BatchNorm: int8 â†’ int32 accumulator (scale and shift, or BN folded into Conv parameters where applicable)
4. **Boundary 2**: int32 â†’ int8 (requantize)
5. ReLU: int8 â†’ int8 (clamp negatives to zero)

Two requantization boundaries. Two rounds of rounding and clamping. Two sites where information is lost.

**Fused execution:**

1. Conv-BN-ReLU (single fused operator): int8 inputs â†’ int32 accumulator â†’ batch norm scaling applied in int32 â†’ ReLU applied in int32
2. **Boundary 1**: int32 â†’ int8 (requantize)

One boundary instead of two. The int32 accumulator passes through convolution, batch normalization, and ReLU without ever being compressed to int8 in between. The intermediate values maintain int32 precision throughout.

The fused operator produces the same real-valued result as the unfused sequence in exact arithmetic. But the intermediate requantization steps are eliminated, and that changes the numerical result â€” because every requantization step that is removed is a rounding operation that no longer corrupts the signal.

---

## The Numerical Difference

To see why this matters, trace a concrete value through both paths.

Suppose the Conv layer produces an int32 accumulator value of 47,382 for a particular output position. Under unfused execution:

**Boundary 1** (after Conv): \\(47{,}382\\) rescales and rounds to int8 value \\(126\\).

BatchNorm then operates on this int8 value \\(126\\). Suppose BN's scale and shift produce an int32 accumulator of \\(8{,}241\\).

**Boundary 2** (after BN): \\(8{,}241\\) rescales and rounds to int8 value \\(93\\).

ReLU passes 93 through unchanged (positive value).

Final output: int8 value \\(93\\).

Under fused execution, the same Conv produces int32 value \\(47{,}382\\). BN's scale and shift are applied directly to this int32 value, producing int32 value \\(8{,}247\\) (different from 8,241 because the input was not rounded to int8 first). ReLU passes it through.

**Single boundary**: \\(8{,}247\\) rescales and rounds to int8 value \\(93\\).

In this case the final outputs happen to match. But change the numbers slightly â€” as happens across thousands of output positions â€” and the unfused path produces values that differ from the fused path by one or two int8 steps. Across an entire layer, these differences accumulate. Across an entire model, they compound.

The fused result is closer to the float32 reference because it rounds once instead of three times. Every eliminated boundary is a rounding operation that no longer distorts the signal.

---

## Scale Alignment at Elementwise Operations

Fusion eliminates boundaries between operators that form a chain. But not all operator combinations can be fused. Elementwise operations â€” addition, concatenation, residual connections â€” require all inputs to share the same scale and zero-point.

Consider a residual addition: the main path and the skip connection meet at an elementwise add. If the main path has output scale \\(S_1 = 0.042\\) and the skip connection has output scale \\(S_2 = 0.037\\), the integers from the two paths represent different real values per step. Adding them directly would be meaningless â€” it would be like adding a measurement in inches to a measurement in centimeters.

To make the addition valid, one or both paths must be *rescaled* to a common scale before the add. This rescaling is itself a requantization step â€” it introduces rounding and clamping. The boundary that fusion was supposed to eliminate reappears at the merge point.

This is the scale alignment invariant: **an integer-only elementwise add requires all inputs to have identical \\((S, Z)\\) parameters.** When scales mismatch, the compiler must insert a rescale (requantization) or fall back to a higher-precision path â€” either way adding a boundary that increases error.

*Accuracy pattern: Distribution Mismatch â€” forcing a common scale may waste budget on one branch's unused range. Runtime pattern: Silent Fallback â€” if the quantized add is unsupported, the runtime falls back to float.*

In architectures with many residual connections â€” ResNets, transformers â€” every skip connection is a potential site for scale mismatch. If the scales of the main and residual paths diverge, each residual addition injects a rescale boundary. A 50-layer ResNet with 25 residual blocks can gain up to 25 additional boundaries from scale mismatch alone (worst case, pre-fusion).

---

## Fusion Determines What to Calibrate

The decision of which operators to fuse must be made *before* calibration (Chapter 9), because fusion changes the graph that will actually execute.

In the unfused graph, there is an intermediate int8 tensor between Conv and BatchNorm. An observer placed there would collect statistics about that tensor's range. In the fused graph, that tensor does not exist â€” the int32 accumulator flows directly through BN without materializing an int8 intermediate.

If calibration is performed on the unfused graph but execution uses the fused graph, the collected scales correspond to boundaries that no longer exist. The calibration data is for a graph that is not the one being executed. This mismatch can produce incorrect scales at the boundaries that do exist â€” scales that were never directly observed.

The rule is straightforward: calibrate the fused graph, not the original. The observer placement must match the execution graph. Observer placement is part of the compilation artifact â€” changing fusion changes which tensors exist and therefore what can be observed.

*Accuracy pattern: Calibration Mismatch â€” calibrating an unfused graph but executing a fused graph produces scales for boundaries that no longer exist.*

---

## What Hardware Provides

Fusion is not a compiler trick that works on any operator sequence. Hardware backends provide a fixed set of *fusion recipes* â€” specific operator patterns that they can execute as a single fused kernel.

Common supported fusions include:
- Conv â†’ BatchNorm â†’ ReLU (as seen above)
- Conv â†’ ReLU
- Linear â†’ ReLU
- Conv â†’ Add (for residual connections, when scales are already aligned)

Operator sequences that do not match a supported recipe remain unfused. The intermediate boundaries remain. The rounding error remains.

A fused kernel executes as a single kernel launch â€” one memory read for the inputs, one memory write for the outputs, no intermediate materialization. An unfused sequence often forces writing the intermediate int8 tensor to memory and reading it back for each subsequent operation. The cost is not just computational â€” each memory round trip consumes bandwidth, which is the bottleneck established in Chapter 1.

**Concrete memory cost of unfused execution.** For a Conv layer producing a [1, 256, 56, 56] activation tensor in int8 (0.8 MB), unfused Conv â†’ BN â†’ ReLU requires: write after Conv (0.8 MB), read for BN (0.8 MB), write after BN (0.8 MB), read for ReLU (0.8 MB), write after ReLU (0.8 MB), plus one final read by the next layer (0.8 MB). Total: 6 memory transactions Ã— 0.8 MB = 4.8 MB. Fused Conv-BN-ReLU: write once (0.8 MB), read once by next layer (0.8 MB). Total: 1.6 MB. The unfused path moves 3Ã— more data for the same computation. At 900 GB/s, the unfused overhead is \\((4.8 - 1.6) / 900{,}000 \approx 3.6\\) Âµs per layer. Across 50 layers in a ResNet: 180 Âµs of pure bandwidth waste.

---

## Fusion in the Weight-Only Regime

Everything above assumes full integer quantization: int8 inputs, int8 weights, int32 accumulators, and int8 outputs. The boundary rules and fusion benefits follow from this pipeline. But in weight-only quantization (Chapter 16), the picture changes fundamentally.
(Weight-only means only weights are quantized, while activations stay in floating-point.)

In weight-only mode, weights are stored in int4 or int8 but are dequantized to float16 *inside the kernel* before the matrix multiply. The matmul itself runs in float16. Activations are never quantized. The output is float16.

This means:

- There is no int32 accumulator domain. Accumulation happens in float16 (or float32 on some hardware).
- There are no activation-domain requantization boundaries. The output is already in float16 â€” no int32â†’int8 conversion is needed because activations remain floating-point throughout.
- The boundary count from this chapter is irrelevant. A 50-layer model with weight-only quantization has zero requantization boundaries in the sense defined by Chapters 6â€“7.

Fusion still matters for weight-only models, but for a different reason: it reduces the number of dequantization kernel launches. A fused Linearâ†’ReLU kernel dequantizes the weights once and applies ReLU in the same kernel. An unfused version dequantizes, writes float16 output, reads it back for ReLU, and writes again.

The critical distinction: Conv-BN-ReLU fusion in full integer mode eliminates *requantization error*. The same fusion in weight-only mode eliminates *memory traffic*. The computational benefit is different, and treating them as equivalent leads to incorrect performance predictions.

---

## Conceptual Consolidation

Fusion is the primary mechanism for reducing requantization boundaries. Every boundary it eliminates is a rounding step that no longer corrupts the signal. But fusion does not solve all boundary problems: elementwise operations require scale alignment, and mismatched scales reintroduce the boundaries fusion was meant to remove.

When evaluating a quantized model's accuracy, two structural questions matter: how many boundaries does the fused graph have, and how many of those are rescale insertions forced by scale mismatch? These counts â€” not the precision of individual weights â€” determine the error surface.

**Fusion checklist for practitioners:**

1. Which fusion recipes does the target backend support? (Conv-BN-ReLU, Linear-ReLU, Conv-Add?)
2. Are scales aligned at all elementwise operations, or are rescale insertions being generated?
3. Are observers placed on the fused graph, not the original unfused graph?
4. Profiling symptom: many small kernels with conversions between them â†’ Fusion Loss. Unexpected floating-point ops in a quantized graph â†’ Silent Fallback.

