# Chapter 7: Requantization

In this chapter, we quantize accumulator outputs back into activation domains at graph boundaries.

## What Happens at the Boundary

Every linear layer and convolution in a quantized graph produces int32 accumulators that must be converted to int8 before the next operator. This conversion â€” *requantization* â€” introduces error, and in many deep quantized graphs, the cumulative effect across many boundaries is a leading source of accuracy loss, especially once tail clipping is under control.

Requantization (or an equivalent domain conversion) is structurally required whenever an int32 accumulator must feed an operator that consumes int8 in a different domain. It is often fused into the producer's epilogue, but the conversion itself is unavoidable unless domains are explicitly aligned or operators are fused.

---

## The Requantization Operation

The int32 accumulator holds a value \\(x_{\text{acc}}\\) that represents a real number under the accumulator's scale \\(S_{\text{acc}}\\). The next operator expects int8 inputs under a different scale \\(S_{\text{out}}\\) with zero-point \\(z_{\text{out}}\\). Requantization converts from one domain to the other:

$$q_{\text{out}} = \text{clamp}\!\left(\left\lfloor \frac{S_{\text{acc}}}{S_{\text{out}}} \cdot x_{\text{acc}} \right\rceil + z_{\text{out}},\; q_{\min},\; q_{\max}\right)$$

Three things happen in sequence:

1. **Rescale**: multiply the int32 value by the ratio \\(S_{\text{acc}} / S_{\text{out}}\\) to convert from one scale to another.
2. **Round**: the rescaled value is generally not an integer. Round to the nearest integer.
3. **Clamp**: if the rounded value falls outside the int8 range \\([q_{\min}, q_{\max}]\\), force it to the boundary.

After this operation, the int32 accumulator is discarded. The int8 result is all that remains.

---

## A Concrete Example

Continue the three-layer model from Chapter 6: Linearâ‚ â†’ ReLU â†’ Linearâ‚‚.

At Boundary 1 (after Linearâ‚, before ReLU), suppose:
- The int32 accumulator holds the value \\(x_{\text{acc}} = 47{,}382\\)
- The accumulator scale is \\(S_{\text{acc}} = 0.000237\\) (product of input scale and weight scale)
- This accumulator value represents the real number: \\(47{,}382 \times 0.000237 = 11.23\\)
- The output domain expects signed int8 [-128, 127] with scale \\(S_{\text{out}} = 0.0891\\) and \\(z_{\text{out}} = 0\\)

The scale ratio is:

$$\frac{S_{\text{acc}}}{S_{\text{out}}} = \frac{0.000237}{0.0891} = 0.002661$$

Applying the rescale:

$$0.002661 \times 47{,}382 = 126.09$$

Rounding: \\(\lfloor 126.09 \rceil = 126\\)

Clamping: 126 is within [-128, 127]. No clamping needed.

The int8 output is 126, representing real value: \\(0.0891 \times 126 = 11.23\\).

*Accuracy pattern: Cumulative Rounding Noise â€” the core mechanism of this chapter.*

In this case, the rounding error is small: the original real value was 11.23, and the dequantized result is also 11.23. But change the accumulator to \\(x_{\text{acc}} = 47{,}490\\):

$$0.002661 \times 47{,}490 = 126.37$$

Rounds to 126 â€” the same output as before. The original real value was \\(47{,}490 \times 0.000237 = 11.26\\), but the dequantized output is still 11.23. The difference of 0.03 is lost. Two distinct accumulator values â€” 47,382 and 47,490 â€” have collapsed to the same int8 output.

Now consider a value that exceeds the range. If \\(x_{\text{acc}} = 50{,}200\\):

$$0.002661 \times 50{,}200 = 133.58$$

Rounds to 134, but the int8 range maxes out at 127. The value is clamped to 127, representing \\(0.0891 \times 127 = 11.32\\). The true value was \\(50{,}200 \times 0.000237 = 11.90\\). The clipping error is \\(11.90 - 11.32 = 0.58\\) â€” far larger than the rounding error.

*Accuracy pattern: Tail Clipping â€” when the rescaled value exceeds the int8 range, the clamp destroys information.*

---

## The Scale Ratio Is Approximated in Hardware

The rescale step uses the ratio \\(S_{\text{acc}} / S_{\text{out}}\\). In the mathematical description, this is a floating-point division. Hardware does not implement it that way â€” hardware lacks fast floating-point division in the integer datapath.

The ratio is precomputed at compile time and decomposed into two integers: a multiplier \\(M\\) and a right-shift \\(n\\). The rescale becomes:

$$q_{\text{out}} = \text{clamp}\!\left(\left\lfloor \frac{M \cdot x_{\text{acc}}}{2^n} \right\rceil + z_{\text{out}}\right)$$

This replaces a floating-point division with an integer multiply and a bit-shift â€” operations that map directly to fixed-function silicon (Chapter 4).

\\(M\\) is typically a 32-bit integer. \\(n\\) is chosen so that \\(M / 2^n\\) approximates \\(S_{\text{acc}} / S_{\text{out}}\\) as closely as possible. But the approximation is never exact. The realized scale is the nearest value representable as \\(M / 2^n\\), not the true ratio.

For the example above: the true ratio is 0.002661. With \\(n = 31\\) and \\(M = 5{,}718{,}573\\):

$$\frac{5{,}718{,}573}{2^{31}} = \frac{5{,}718{,}573}{2{,}147{,}483{,}648} = 0.0026628...$$

The approximation error is \\(|0.002661 - 0.002663| = 0.000002\\) â€” seemingly negligible. But this error applies to every value passing through this boundary. In static integer-only inference, the multiplier/shift approximation is fixed at compile time and applies to every value crossing that boundary for every input the model will ever see.

**Verifying the approximation.** The true ratio is 0.002661. The hardware computes \\(M / 2^n = 5{,}718{,}573 / 2{,}147{,}483{,}648 = 0.002663\\). The relative error is \\(0.000002 / 0.002661 = 0.075\%\\). Applied to the accumulator value 47,382: the true rescaled value is \\(0.002661 \times 47{,}382 = 126.09\\). The hardware computes \\(0.002663 \times 47{,}382 = 126.18\\). Both round to 126 â€” identical. But at larger accumulator values, the error can push a value across a rounding boundary. At \\(x_{\text{acc}} = 188{,}000\\): true result \\(= 500.07\\), hardware result \\(= 500.44\\) â€” a 0.37 difference that could round differently. Across a 4096-wide dot product with many such boundaries, the cumulative effect of the \\(M/2^n\\) approximation is typically small but not zero.

---

## Compounding Across Depth

A single requantization step introduces a small rounding error. A 50-layer model has roughly 50 requantization boundaries, each independently introducing rounding error. The error at each boundary becomes part of the input to every subsequent layer.

Consider the three-layer example. At Boundary 1, the accumulator value 47,490 rounds to int8 value 126, introducing an error of 0.03 in real-valued terms. This int8 value of 126 becomes one of the inputs to Linearâ‚‚. Linearâ‚‚ multiplies it by weights, accumulates the products, and produces its own int32 accumulator â€” which now includes the error from Boundary 1 multiplied and mixed with the errors from all other inputs.

At Boundary 2, the process repeats: the int32 accumulator is rescaled, rounded, and clamped to int8. New rounding error is introduced on top of the error inherited from Boundary 1.

The model has no mechanism to detect that its input was rounded at a previous boundary. It processes whatever values arrive. Error introduced at layer \\(k\\) propagates through all layers \\(k+1, k+2, \ldots, N\\). In a deep network, the earliest requantization errors influence every subsequent computation.

### Why Early Layers Hurt More

Not all noise is equal. Error introduced at layer 1 passes through every subsequent layer â€” it is multiplied by weights, mixed with other errors, and amplified or attenuated by each layerâ€™s Jacobian. Error introduced at the last layer affects only the final output.

Think of it concretely: an error of 0.03 at layer 1 gets multiplied by layer-2 weights (\\(\approx \pm0.1\\) to \\(\pm0.5\\)), producing a modified error. That error is then mixed with all other layer-2 errors, summed in a dot product of 4096 terms, and rounded again at the next boundary. By layer 50, the original layer-1 error has been transformed, amplified, and mixed so many times that its effect on the final output can be large or small â€” but it is present in every subsequent computation. An error at layer 50 never had the chance to propagate.

Residual connections partially mitigate this: the skip path carries a less-corrupted copy of the signal past the noisy main path. But the mitigation is partial â€” the merge point (elementwise add) still combines the corrupted and clean signals, and the resulting sum carries error from both branches.

---

## Residual Connections Merge Independent Errors

In architectures with skip connections, a residual branch and a main branch are processed independently. Each branch passes through its own operators and its own requantization boundaries, accumulating its own rounding errors under its own scales.

When the two branches merge â€” typically through an elementwise addition â€” their independently quantized values are summed. The rounding errors from both paths combine. If the main branch rounded up at its last boundary and the residual branch also rounded up, the combined error is additive. If they rounded in opposite directions, the errors partially cancel â€” but there is no mechanism that guarantees cancellation.

Every residual addition is a site where independently accumulated quantization noise merges. In architectures like ResNets, where residual additions occur at every block, these merge points are dense. A 50-layer ResNet has roughly 25 residual additions, each combining errors from two independently requantized paths.

### Scale Alignment as a Preventive Fix

The residual merging problem has a direct mitigation: *scale alignment*. Instead of letting each branch compute its own optimal scale independently, force both the main branch and the residual branch to share a single \\((S, Z)\\) pair at the addition node.

**Worked example: scale mismatch at a residual addition.** Main path output: scale \\(S_1 = 0.042\\), \\(Z_1 = 5\\). Skip path output: scale \\(S_2 = 0.037\\), \\(Z_2 = -3\\). A skip path int8 value of 50 represents real value \\(0.037 \times (50 - (-3)) = 0.037 \times 53 = 1.961\\). To add this in the main path's domain, we need \\(q_{\text{rescaled}} = \text{round}(1.961 / 0.042) + 5 = \text{round}(46.69) + 5 = 47 + 5 = 52\\). The rescaled int8 value 52 represents \\(0.042 \times (52 - 5) = 1.974\\). The true value was 1.961 â€” the rescaling introduced an error of 0.013. This error is added on top of whatever rounding error the skip path already carried. Scale alignment (forcing \\(S_1 = S_2\\) and \\(Z_1 = Z_2\\)) eliminates this rescale step and its error entirely.

This means one branch â€” typically the one whose distribution is narrower â€” uses a scale that is suboptimal for its own range. That branch suffers slightly worse resolution than it would under its own independently computed scale. But the payoff is structural: because both branches share the same scale, the elementwise addition is valid without any rescale insertion. No extra boundary is created. No additional rounding step occurs at the merge point.

The trade-off is explicit: slightly worse per-branch resolution in exchange for eliminating an entire requantization boundary. In many practical graphs, eliminating a rescale boundary at a residual merge is worth a modest per-branch resolution loss â€” especially when the merge would otherwise require an extra requantization step. There are cases where the shared scale forces too wide a range and hurts resolution significantly; profiling both configurations is the only reliable way to decide.

Scale alignment must be decided before calibration. The observer at the addition node must observe both branches simultaneously and compute a single scale that covers both ranges. If each branch is calibrated independently and the mismatch is discovered at compile time, the only fix is a rescale insertion â€” exactly the boundary this technique avoids.

*Accuracy pattern: Distribution Mismatch â€” a shared scale may waste budget on one branch's unused range. Runtime pattern: Fusion Loss â€” if the quantized add cannot be fused, a float fallback at the merge point adds both latency and an extra conversion boundary.*

---

## Conceptual Consolidation

Quantized models do not fail because integer arithmetic is inaccurate. Integer multiply-accumulate in int32 is exact. They fail because the conversion from int32 back to int8 â€” requantization â€” rounds and clamps at every boundary, and the cumulative effect of dozens or hundreds of these projections reshapes value distributions beyond what the model's learned representations can tolerate.

When accuracy degrades in a quantized model, the first question is: how many requantization boundaries exist, and can any of them be eliminated?

**Diagnostic checklist for requantization-driven accuracy loss:**

1. Count boundaries in the quantized graph (before and after fusion). Each is a noise source.
2. Check clamp frequency: if many values hit \\(q_{\min}\\) or \\(q_{\max}\\) at any boundary, Tail Clipping dominates â€” fix calibration or widen the range.
3. At residual additions, check whether branches share a scale or require a rescale insertion. Rescale insertions are extra noise sources.
4. If the accuracy drop is mostly from rounding noise (gradual, not sharp), QAT (Chapter 11) can shift weight distributions to be robust to these projections.


