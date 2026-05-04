# Chapter 2: Quantization as a Representational Constraint

## What Float32 Gives You

A 32-bit floating-point number can represent approximately 4.3 billion distinct bit patterns. These values are not evenly distributed — they are packed densely near zero and spread apart as magnitude increases. Near zero, float32 is extremely dense: within [0, 1] you get on the order of a billion distinct representable values. At large magnitudes the spacing grows rapidly — between 1,000 and 1,001, roughly 8,000 values remain.

For neural network purposes, this density is effectively infinite. Two activations that differ by 0.00001 are stored as genuinely different numbers. The network can learn to depend on differences that small, and the representation will preserve them faithfully.

This is the representational budget that float32 provides: a near-continuous space where any value can be stored and distinguished from its neighbors.

---

## What Int8 Gives You

An 8-bit integer has exactly 256 distinct values. That is the entire budget — not 4 billion, not a million, but 256. These values are evenly spaced across whatever range they are assigned to cover.

Suppose the range is [-1.0, 1.0]. Those 256 levels must span 2.0 units of real-valued range. The spacing between adjacent representable values — the *step size* — is given by:

$$\Delta = \frac{r_{\max} - r_{\min}}{L - 1}$$

where $L$ is the number of levels and we assume the representable endpoints align with the range boundaries (levels include both endpoints — this is the standard convention in most quantization implementations). We use $L - 1$ in the denominator, not $L$, because we are counting the *gaps between* levels rather than the levels themselves: with 256 levels, there are 255 gaps. For [-1.0, 1.0] with $L = 256$:

$$\Delta = \frac{2.0}{255} \approx 0.00784$$

Every real value that falls within this range gets mapped to the nearest grid point. The value 0.50000 maps to one grid point. The value 0.50700 maps to the same grid point. In float32, these were different numbers. In int8, they are identical. The distinction between them no longer exists.

---

> **Key Terms — Quick Reference**
>
> These terms appear throughout the book. Brief definitions are here for orientation; each is explained fully in its own chapter.
>
> | Term | Meaning | Chapter |
> |---|---|---|
> | **Grid** | The fixed set of representable integer values (e.g., 256 for int8) | Ch. 2 |
> | **Scale (S)** | The step size between adjacent grid points; converts integers back to real values | Ch. 3 |
> | **Zero-point (Z)** | An integer offset that aligns the grid's zero with the real-valued zero | Ch. 3 |
> | **Boundary** | A point in the computation graph where the quantization domain (scale, zero-point, precision) changes | Ch. 6 |
> | **Domain** | The (scale, zero-point, precision) triple that governs how integers are interpreted | Ch. 6 |
> | **Requantization** | Rescaling an int32 accumulator result back to int8 for the next layer | Ch. 7 |
> | **Fusion** | Merging consecutive operations (e.g., Conv + BN + ReLU) to eliminate intermediate boundaries | Ch. 8 |
> | **Observer** | A temporary module that records activation ranges during calibration | Ch. 9 |
> | **Accumulator** | The wider-precision register (typically int32) that holds matmul partial sums | Ch. 6 |
> | **Calibration** | Running representative data through the model to determine activation scales | Ch. 9 |

---

## The Grid

Quantization places a uniform grid over a range. The grid has a fixed number of points — 256 for int8, 16 for int4 — and the spacing between them is constant. Every real value snaps to the closest grid point, the way a ruler with millimeter marks cannot distinguish between 3.2 mm and 3.3 mm if it only has marks at each millimeter.

Consider a layer whose activation values in float32 fall in the range [-1.0, 1.0]. Under int8 quantization with 256 levels:

- The value 0.3021 maps to grid point $0.2980$ (integer index 166).
- The value 0.3058 also maps to grid point $0.2980$ (integer index 166).
- These two values — distinct in float32 — are now the same number. Whatever functional difference the network relied on between 0.3021 and 0.3058 is gone.

The mapping rule for this simplified case (unsigned codes 0 to $L-1$):

$$q = \text{round}\!\left(\frac{r - r_{\min}}{\Delta}\right), \qquad r' = r_{\min} + q \cdot \Delta$$

For 0.3021 with $r_{\min} = -1.0$ and $\Delta \approx 0.00784$: $q \approx \text{round}(1.3021\,/\,0.00784) = 166$, recovering $r' = -1.0 + 166 \times 0.00784 = 0.2980$.

### Worked Example: One Range, Three Outcomes

To anchor the intuition for the rest of the book, here is a single complete example showing scale computation, clean rounding, and clipping — the three things that happen to every value entering a quantized layer.

**Setup.** A layer's activation range is $[-1.2, 1.0]$. We quantize to int8 asymmetric (unsigned codes $0$ to $255$).

**Step 1 — Scale and zero-point.**

$$S = \frac{r_{\max} - r_{\min}}{255} = \frac{1.0 - (-1.2)}{255} = \frac{2.2}{255} \approx 0.008627$$

$$Z = \text{round}\!\left(\frac{0 - r_{\min}}{S}\right) = \text{round}\!\left(\frac{1.2}{0.008627}\right) = \text{round}(139.1) = 139$$

So integer code 139 corresponds to real value 0.0 (approximately). The grid spans from code 0 ($= -1.2$) to code 255 ($= 1.0$).

**Step 2 — A value that rounds cleanly.** Take $r = 0.37$.

$$q = \text{round}\!\left(\frac{0.37}{0.008627}\right) + 139 = \text{round}(42.89) + 139 = 43 + 139 = 182$$

$$r' = (182 - 139) \times 0.008627 = 43 \times 0.008627 = 0.3710$$

Error: $|0.37 - 0.3710| = 0.0010$. Well within the maximum rounding error of $S/2 \approx 0.0043$. The grid handled this value without trouble.

**Step 3 — A value that clips.** Take $r = 1.5$ (outside the range).

$$q = \text{round}\!\left(\frac{1.5}{0.008627}\right) + 139 = \text{round}(173.8) + 139 = 174 + 139 = 313$$

But int8 unsigned can only hold codes $0$ to $255$. Code 313 is clamped to 255.

$$r' = (255 - 139) \times 0.008627 = 116 \times 0.008627 = 1.0007 \approx 1.0$$

Error: $|1.5 - 1.0| = 0.5$. That is ~$500\times$ larger than the rounding error above. The value was outside the representable range and was crushed to the boundary. This is clipping error (Chapter 5).

**Takeaway.** Same grid, same scale — two completely different error magnitudes. The value inside the range lost 0.001. The value outside lost 0.5. This is why range selection (Chapter 9) determines whether quantization succeeds or fails.

The grid does not care which values are important. It is uniform — and uniform for a reason: constant step size maps directly to integer arithmetic, which is what hardware can execute cheaply and in bulk. The grid does not allocate more resolution where the network needs it and less where it does not. Every region of the range gets the same step size, regardless of how many values actually live there. Later chapters will exploit distribution structure through per-channel scaling and groupwise quantization, but the underlying grid remains uniform.

---

## The Range–Precision Trade-Off

The step size depends on two things: the range being covered and the number of grid points available. With a fixed number of grid points (256 for int8), a wider range forces a larger step size.

Take the same 256 levels and expand the range from [-1.0, 1.0] to [-10.0, 10.0]:

$$\Delta = \frac{20.0}{255} \approx 0.0784$$

The step size increased 10×. Values that were distinguishable under the narrow range are no longer distinguishable under the wide range. The value 0.3021 and the value 0.3500 — clearly different numbers — now map to the same grid point.

This is the fundamental trade-off: *wider range means coarser resolution*. If the range must accommodate a few extreme values, the resolution for all other values suffers. The grid has a fixed budget, and it must be spread across whatever range is claimed. Rule of thumb: if outlier tails force the range wide, effective resolution on the dense bulk collapses.

Now consider int4 — 16 levels instead of 256. Under the [-1.0, 1.0] range:

$$\Delta = \frac{2.0}{15} \approx 0.1333$$

Sixteen grid points to cover 2.0 units of range. Enumerating them: 0.0, 0.1333, 0.2667, 0.4, 0.5333, 0.6667, 0.8, 0.9333, 1.0667, 1.2, 1.3333, 1.4667, 1.6, 1.7333, 1.8667, 2.0 — all 16 values spaced at exactly 0.1333. The entire interval from 0.0 to 0.1333 collapses to a single representable value. Values of 0.05, 0.08, and 0.12 — all distinct in float32, all distinguishable in int8 — become the same number in int4.

The coarseness at 4-bit is not subtle. It is a qualitative change in what the representation can express.

---

## What Is Actually Lost

The loss is not "precision" in the colloquial sense, as if the numbers become slightly noisy. The loss is the ability to distinguish values that were previously distinct.

When two different activation values collapse to the same grid point, the network downstream sees identical inputs where it previously saw different ones. If the network's learned computation depended on that difference — if one path should fire and the other should not — the quantized representation cannot support that distinction. The decision boundary that existed in float32 has been erased from the representable set.

This is not degradation. It is a constraint. The model must now express everything it needs to express using only the values that survive the grid. If its learned representations happen to require distinctions finer than the step size, those representations break. If they do not, quantization costs nothing.

Whether a given model survives this constraint depends on the relationship between its learned value distributions and the grid imposed by quantization. That relationship is the subject of the chapters that follow.

> **📊 INSERT DIAGRAM: The Range–Precision Trade-Off**
>
> Three side-by-side number-line diagrams, all using int8 (256 levels), showing the same distribution of activation values (a bell curve centered near 0) under three different range choices:
>
> ```
> Panel A: Range [-1, 1] — "Tight range"
>   - Grid points densely packed (step = 0.0078)
>   - Bell curve fits perfectly inside the range
>   - No clipping, high precision
>   - Label: "✓ Best case: range matches distribution"
>
> Panel B: Range [-10, 10] — "Too wide"
>   - Grid points spread apart (step = 0.078)
>   - Bell curve occupies only ~10% of the grid
>   - No clipping, but ~230 of 256 codes are wasted in empty tails
>   - Label: "✗ Wasted resolution: most codes represent values that never appear"
>
> Panel C: Range [-0.5, 0.5] — "Too narrow"
>   - Grid points very dense inside range (step = 0.0039)
>   - Tails of the bell curve extend past ±0.5 and are clipped
>   - High precision inside, but values outside are crushed to boundary
>   - Label: "✗ Clipping: tail values destroyed"
> ```
>
> Annotate: "This is the fundamental trade-off (Ch.5, Ch.9): widen the range and lose precision, or narrow it and clip outliers."

---

## Conceptual Consolidation

Quantization is not "using lower precision." It is operating within a finite grid of representable values, uniformly spaced, with a step size determined by the range and the number of available levels.

Whenever you evaluate a quantized model, the question is not "how much precision did we lose?" The question is: does the step size preserve the distinctions the model depends on? If the answer is yes for every layer, quantization is free. If not, the model's behavior changes — and understanding where and why is the purpose of this book.

This chapter treated the simplest uniform grid. The next chapter makes it practical by introducing scale and zero-point.
