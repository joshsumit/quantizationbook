## What Float32 Gives You

To understand quantization, we first need to understand the default number system used in deep learning: Float32.

### The Problem with Fixed Decimals

In a hardware register restricted to 8 decimal digits (this example uses decimal digits for intuition; real hardware operates in binary bits), implementing a fixed-point system requires permanently locking the radix point (decimal point in base-10) at a predetermined position— for example, directly in the center. This design enforces a rigid structural split:

```
[ integer ][ integer ][ integer ][ integer ] . [ fraction ][ fraction ][ fraction ][ fraction ]
```

This layout allocates exactly four digits for whole numbers and four digits for fractional precision. While it can accurately store a value like 0012.5000 or 0000.0075, this rigid structure imposes two fundamental limitations:

**Inability to represent large magnitudes:** A value like the speed of light (299,792,458) cannot be stored because only four digits are available before the radix point.

**Inability to represent high precision:** A minute subatomic measurement like 0.000000000053 cannot be stored because the fractional capacity cuts off after just four digits.

To overcome this limitation, we need a way to reuse the same digits across vastly different scales. This is exactly what scientific notation does.

Instead of storing the entire number with the radix point fixed at a predetermined position (as in fixed-point representation), the representation is split into two parts: a **mantissa** (which stores the significant digits) and an **exponent** (which controls the scale by effectively shifting the radix point).

\\[
2.99792458 \\times 10^8 \\quad (\\text{mantissa} = 2.99792458,\\; \\text{exponent} = 8)
\\]

\\[
5.3 \\times 10^{-11} \\quad (\\text{mantissa} = 5.3,\\; \\text{exponent} = -11)
\\]

Here, the mantissa stores the digits, while the exponent controls the scale of the number.

By separating the number into a mantissa and an exponent, we store the digits and the required scale independently. The exponent specifies how many places the radix point is effectively shifted, allowing the same set of digits to represent values across a vast range.

In binary, the same idea looks like: \\(1.011 \\times 2^8\\) (where the base is 2 instead of 10). Modern computing hardware implements this exact concept using base-2 (binary) logic, standardized universally under the IEEE-754 specification.

### The Floating Dynamic Range

### The Exponent
The Exponent dictates the scale or boundaries of the grid, but it does not define the range by itself—it scales a *normalized* mantissa. 

A normalized binary number is written in a form where there is exactly one non-zero digit before the radix point. In base-2 (binary), this means every number is written as `1.xxxxx`, where everything after the point is the fractional part. Because of this rule, the mantissa is always constrained to the range:

\\[
1.0 \\leq \\text{mantissa} < 2.0
\\]

For example:
- `1.0101` is normalized  
- `0.1010` is **not** normalized  (it would be rewritten as `1.010 \\times 2^{-1}`)

This is why the exponent alone does not define the range. Instead, it scales this fixed base interval. The actual values representable for a given exponent are obtained by multiplying the entire normalized range by \\(2^e\\), producing:

\\[
[2^e,\\; 2^{e+1})
\\]

To make this concrete, you can visualize the same normalized interval being scaled across different exponent values:

**Normalized Mantissa Interval (before scaling):**
```
1.0                                              2.0
|----|----|----|----|----|----|----|----|----|----|
```

**Exponent = 0 (scale = \\(2^0\\)) (no scaling):**
```
1.0                                              2.0
|----|----|----|----|----|----|----|----|----|----|
     very fine spacing (tiny gaps between numbers)
```

**Exponent = 16 (scale = \\(2^{16}\\)) (scaled up):**
```
65536                                            131072
|----|----|----|----|----|----|----|----|----|----|
     same number of steps, but much wider gaps
```

Even though all three views use the *exact same number of representable values*, the exponent stretches or compresses the interval across the number line.

For example:
- When the exponent is set to \\(0\\), the grid spans the window between `1.0` and `2.0`  
- When the exponent increases to \\(16\\), the exact same normalized interval is scaled up to span `65,536.0` to `131,072.0`

### The Mantissa 
The Mantissa provides a fixed precision budget to divide the window established by the exponent. In a standard \\(\\text{Float32}\\) architecture, the mantissa utilizes 23 physical bits (plus one implicit leading bit—a leading `1` that is not stored explicitly but assumed for normalized numbers) to provide 24 bits of resolution. 

This means that regardless of the scale chosen by the exponent, the grid within each exponent window is always divided into exactly \\(2^{24}\\) (\\(16,777,216\\)) uniformly spaced steps.

Because the number of internal grid lines remains constant while the boundaries scale geometrically, the density of representable numbers is fundamentally non-uniform. The gap between adjacent numbers roughly scales as \\(2^{(\text{exponent} - 23)}\\). The table below maps how this fixed budget of \\(2^{24}\\) steps alters the resolution across different numerical ranges:

| Exponent Scale (Window) | Total Window Width | Number of Grid Steps (Resolution) | Step Size (Gap Between Numbers) | Density Context |
| :--- | :--- | :--- | :--- | :--- |
| **\\(2^0\\)** (`1.0` to `2.0`) | `1.0` | \\(2^{24}\\) (\\(16,777,216\\)) | \\(\sim 5.96 \\times 10^{-8}\\) | High Density (Precise tracking near 1.0) |
| **\\(2^{16}\\)** (`65,536.0` to `131,072.0`) | `65,536.0` | \\(2^{24}\\) (\\(16,777,216\\)) | `0.00390625` | Coarse Density (Stretched capacity for large activations) |

### Key Trade-Offs of Non-Uniformity

* **Near-Zero Precision:** When values reside in smaller exponent windows, the fixed allocation of \\(16,777,216\\) steps is squeezed into a tight interval, resulting in highly dense, microscopic step sizes.
* **Large Magnitude Tolerance:** When a value scales up to a window like \\(131,072.0\\), those same \\(16,777,216\\) steps must stretch across a massive numerical span. The step size balloons to a coarser gap, sacrificing sub-decimal precision to prevent register overflow.

For machine learning contexts (e.g., backpropagation), this uneven spacing is a massive advantage:
1. **In backpropagation**, tiny gradients near zero get hyper-precise updates without getting rounded to zero.
2. **For large weights or activations**, the system can handle massive values without hitting an overflow error, even if it sacrifices a little bit of precision to do so.


*(For a granular, bit-level walkthrough of how a floating-point value is encoded into binary registers under the IEEE-754 specification, refer to **Appendix C: Floating-Point Bit Architecture**).*
---
## What Int8 Gives You

An 8-bit integer (\\(\text{INT8}\\)) provides exactly \\(2^8 = 256\\) distinct representable levels. This integer budget represents the entire numerical pool available to a neural network layer once it undergoes quantization. Unlike floating-point systems, an integer format contains no hidden decimal components, no shifting scaling factors, and no variable density. The 256 unique points are spread completely uniformly across whatever real-world range they are assigned to cover.

To trace how this constraint manifests physically, consider a real-valued activation range bound tightly between \\([-1.0, 1.0]\\). These 256 discrete integer levels must stretch across a total continuous distance of exactly 2.0 units on the real number line. The uniform distance between any two adjacent points on this newly established integer grid is defined as the *step size* (or scale factor, denoted by \\(\Delta\\) or \\(S\\)). This value is derived using the uniform grid blueprint:

\\[\Delta = \frac{r_{\max} - r_{\min}}{L - 1}\\]

Within this mathematical framework, \\(L\\) represents the total number of available discrete levels (256 for \\(\text{INT8}\\)), while \\(r_{\max}\\) and \\(r_{\min}\\) represent the boundary limits of the real-world floating-point distribution. The denominator uses \\(L - 1\\) (255) rather than \\(L\\) because the objective is to calculate the precise distance of the gaps *between* the points, rather than counting the points themselves. 

Evaluating this formula for the \\([-1.0, 1.0]\\) range yields the following step size calculation:

\\[\Delta = \frac{1.0 - (-1.0)}{256 - 1} = \frac{2.0}{255} \approx 0.007843\\]

Every continuous floating-point activation or weight that passes into this quantized layer is forced to snap onto the nearest uniform point on this 256-level grid. 

If an incoming floating-point value evaluates to exactly `0.50000`, it lands precisely on a representable grid coordinate. If a subsequent value evaluates to `0.50400`, the uniform step size of `0.007843` is mathematically too wide to differentiate the two inputs. Consequently, the second value snaps to the exact same grid coordinate as the first. 

In the native \\(\text{Float32}\\) domain, these two values represent distinct signals capable of triggering downstream variance in the network. In the quantized \\(\text{INT8}\\) domain, they collapse into identical bit configurations, and the subtle difference between them is permanently erased.