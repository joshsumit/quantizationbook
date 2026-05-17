# Chapter 2: Quantization as a Representational Constraint

## What Float32 Gives You

To understand quantization, we first need to understand the default number system used in deep learning: Float32.

### The Problem with Fixed Decimals

Imagine you are designing a computer register that only has 8 slots to store a regular base-10 decimal number. If you decide to permanently lock the decimal point right in the middle, you create a "fixed-point" system:
[ integer ][ integer ][ integer ][ integer ] . [ fraction ][ fraction ][ fraction ][ fraction ]

This layout gives you exactly four slots for whole numbers and four slots for fractional details. It can safely store a number like `0012.5000` or `0000.0075`. However, this rigid structure has two fundamental limitations:
1. You cannot store a very large number like the speed of light (`299,792,458`), because you only have four slots before the decimal point.
2. You cannot store a very tiny subatomic measurement like `0.000000000053`, because your fractional details cut off after just four slots.

To solve this limitation, scientists and engineers don't use fixed decimal points. Instead, they use **Scientific Notation**. Instead of writing out endless zeros, they write:

\\[ 2.99792458 \times 10^8 \\]
\\[ 5.3 \times 10^{-11} \\]

By letting the decimal point "float" left or right based on a multiplier (the power of 10), a tiny budget of digits can represent an incredibly vast range of values. Modern computing hardware builds on this exact principle using base-2 (binary) logic under the IEEE-754 standard.

---

## The "Sliding Window" Mental Model

In a standard \\(\text{Float32}\\) variable, the computer allocates a budget of 32 bits divided into three distinct functional components: a **Sign bit**, an **Exponent** (the multiplier), and a **Mantissa** (the fractional precision budget). 

Instead of treating the 32-bit register as a flat number line, it is much more effective to visualize it as a **sliding window hardware engine**:
* The **Exponent** acts as a coarse control loop. It slides a physical window up and down across an astronomical scale of magnitude (from roughly \\(10^{-38}\\) to \\(10^{38}\\)) by shifting powers of 2.
* The **Mantissa** acts as a fine control loop. Once the exponent anchors the window to a specific power-of-two bucket, the mantissa uses its fixed allocation of bits to carve up *only that specific window* into uniformly spaced, high-resolution steps.

### The Non-Uniform Number Line

Consider how this looks on a real number line. The density of representable \\(\text{Float32}\\) numbers is fundamentally non-uniform:

* **Scenario A (Near Zero):** When the exponent is small, the sliding window is focused tightly around zero. The mantissa splits this tiny window into billions of microscopic increments. The step size (the distance between one representable float and the absolute next) is ultra-dense, shrinking down to roughly \\(10^{-9}\\) or smaller.
* **Scenario B (Deep Outliers):** When a model activation explodes out to a large value like \\(131,072.0\\), the exponent slides the window far up the number line. Because the window is now massive, the mantissa must stretch its fixed budget across a huge span. The step size between adjacent representable numbers balloons up to a coarse gap of \\(0.0078125\\).

For training deep neural networks, this non-uniformity is an absolute superpower. During backpropagation, microscopic gradients near zero can be updated with pristine mathematical precision, while massive weight outliers can coexist in the same tensor without breaking the system register. 

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

---

## The Grid

Quantization operates by laying a rigid, uniform grid directly over a continuous distribution of real numbers. Because the total number of points on this grid is physically bounded by the bit-width configuration, the distances separating the representable markers are invariant across the entire range. Every real value is subjected to an unyielding rounding operation. 

This behavioral mechanism maps perfectly to a physical millimeter ruler. A standard ruler marked exclusively in full millimeter increments is structurally incapable of preserving the variance between a measurement of 3.2 mm and 3.3 mm. Because it lacks intermediate lines to represent that micro-space, both measurements are forced to round to the 3 mm marker. The grid functions as a digital millimeter ruler for hardware execution.


To see how these equations process numerical signals, assume a neural network layer features an operational range bound between \\([-1.0, 1.0]\\). Applying uniform 8-bit quantization produces a grid of 256 lines numbered sequentially from code 0 to code \\(L-1\\) (255). 

The mathematical blueprint to map a continuous real value (\\(r\\)) to its corresponding integer code (\\(q\\)) requires a scaling and rounding function:

\\[q = \text{round}\!\left(\frac{r - r_{\min}}{\Delta}\right)\\]

Conversely, the blueprint to reconstruct that integer code back into an approximated real-world value (\\(r'\\)) during subsequent execution steps evaluates as:

\\[r' = r_{\min} + q \cdot \Delta\\]

Tracing two distinct real values through these mapping equations demonstrates the behavior of the grid, utilizing the established minimum boundary of \\(r_{\min} = -1.0\\) and the uniform step size of \\(\Delta \approx 0.007843\\).

Executing the mapping sequence for an initial input value of `0.3021` yields:

\\[q = \text{round}\!\left(\frac{0.3021 - (-1.0)}{0.007843}\right) = \text{round}\!\left(\frac{1.3021}{0.007843}\right) = \text{round}(166.02) = 166\\]

\\[r' = -1.0 + (166 \times 0.007843) = -1.0 + 1.301938 = -0.298062 \approx -0.2981\\]

Executing the identical mapping sequence for a slightly larger adjacent input value of `0.3058` yields:

\\[q = \text{round}\!\left(\frac{0.3058 - (-1.0)}{0.007843}\right) = \text{round}\!\left(\frac{1.3058}{0.007843}\right) = \text{round}(166.49) = 166\\]

\\[r' = -1.0 + (166 \times 0.007843) = -0.2981\\]

The technical implications of this walkthrough are critical. In the first instance, the true value of `0.3021` shifted to `-0.2981` upon reconstruction, injecting an immediate approximation error. In the second instance, because the gap between `0.3021` and `0.3058` was smaller than the grid's structural resolution, both unique inputs collapsed into integer code 166. The downstream mathematical layers now receive the exact same value for both tokens, obliterating the original functional divergence.

---

### Worked Example: One Range, Three Outcomes

To evaluate how a single uniform grid handles varied input distributions, consider a setup where an identical grid structure yields three distinct numerical outcomes: accurate rounding, maximum boundary error, and destructive out-of-bounds clipping.

**The Parameters:** A target activation layer exhibits a real-world data distribution bound between \\([-1.2, 1.0]\\). The pipeline implements asymmetric 8-bit quantization, mapping the values to an unsigned integer grid spanning from code 0 to code 255.

**Stage 1: Calculation of Grid Parameters**

The calculation of the uniform grid scale (\\(S\\)) and integer zero-point (\\(Z\\)) follows the standard asymmetric derivation:

\\[S = \frac{r_{\max} - r_{\min}}{255} = \frac{1.0 - (-1.2)}{255} = \frac{2.2}{255} \approx 0.008627\\]

\\[Z = \text{round}\!\left(\frac{0 - r_{\min}}{S}\right) = \text{round}\!\left(\frac{1.2}{0.008627}\right) = \text{round}(139.09.1) = 139\\]

The resulting configuration dictates that integer code 139 represents a true real value of exactly `0.0`. Integer code 0 maps to the lower real bound of `-1.2`, and integer code 255 maps to the upper real bound of `1.0`.

**Stage 2: Execution of an In-Range Value**

An incoming real-world activation value evaluates to \\(r = 0.37\\). The transformation to the integer domain follows the scale and offset mapping equation:

\\[q = \text{round}\!\left(\frac{r}{S}\right) + Z = \text{round}\!\left(\frac{0.37}{0.008627}\right) + 139 = \text{round}(42.88) + 139 = 43 + 139 = 182\\]

Reconstructing this integer code back into the floating-point domain to isolate the representation error yields:

\\[r' = (q - Z) \times S = (182 - 139) \times 0.008627 = 43 \times 0.008627 = 0.370961 \approx 0.3710\\]

The absolute structural error introduced by the grid evaluates as \\(|0.37 - 0.3710| = 0.0001\\). This error falls safely within the worst-case rounding tolerance of a uniform grid, which is bounded at exactly half a step size:

\\[\text{Maximum Rounding Error} = \frac{S}{2} = \frac{0.008627}{2} \approx 0.004313\\]

The grid has successfully preserved the signal within standard operational limits.

**Stage 3: Execution of an Out-of-Range Outlier Value**

Anomalous model behavior or unnormalized attention mechanics drive a sudden outlier activation value to evaluate well past the grid boundaries at \\(r = 1.5\\). Tracing this value through the mapping equation yields:

\\[q = \text{round}\!\left(\frac{1.5}{0.008627}\right) + 139 = \text{round}(173.87) + 139 = 174 + 139 = 313\\]

Because a standard unsigned 8-bit integer register is physically capped at an absolute maximum storage capacity of 255, the value `313` cannot be represented. The hardware forcefully clamps the value to the maximum boundary code of 255. 

Converting this clamped integer code back to its real-world floating-point approximation yields:

\\[r' = (255 - 139) \times 0.008627 = 116 \times 0.008627 = 1.000732 \approx 1.0\\]

The resulting error for this outlier value evaluates as \\(|1.5 - 1.0| = 0.5\\). This structural distortion is 5,000 times larger than the standard rounding error calculated in Stage 2. Because the true value escaped the analytical bounds of the grid, its magnitude was sheared off at the perimeter. This phenomenon is defined as *clipping error*.

The core lesson of this system breakdown is that the grid possesses no native awareness of information importance. It spreads its 256 representable points completely linearly across the predefined range, oblivious to where the dense majority of activations reside. This rigid design is sustained because uniform spacing enables hardware logic gates to run high-speed, low-cost integer matrix multiplications. Balancing the tension between rounding errors inside the grid and clipping errors at the boundaries constitutes the primary engineering challenge of post-training quantization.

---

## The Range–Precision Trade-Off

The step size of a quantized layer is dictated by a hard architectural compromise between two competing parameters: the width of the real-world range requiring coverage, and the finite bit budget of the processor registers. Because the number of grid points is locked at 256 for \\(\text{INT8}\\) execution, expanding the boundary limits to catch extreme outliers automatically forces the step size to grow wider.

If the baseline range of \\([-1.0, 1.0]\\) is stretched to an expanded tracking range of \\([-10.0, 10.0]\\) to insulate the layer against clipping errors, the uniform step size recalculates as:

\\[\Delta = \frac{10.0 - (-10.0)}{255} = \frac{20.0}{255} \approx 0.078431\\]

Widening the boundary ten-fold has forced the step size to scale up by an identical factor of ten. Real values like `0.3021` and `0.3500`—which mapped to completely independent positions on the tight grid—now collapse into the identical integer position on the wide grid. This defines the fundamental range-precision trade-off: *maximizing range coverage uniformly minimizes local signal resolution*.

This representational crisis intensifies when dropping down to 4-bit integer processing (\\(\text{INT4}\\)), where the total discrete budget collapses from 256 levels down to a mere \\(2^4 = 16\\) unique points. Attempting to map the baseline \\([-1.0, 1.0]\\) range across 16 uniform levels forces the step size to open drastically:

\\[\Delta = \frac{1.0 - (-1.0)}{16 - 1} = \frac{2.0}{15} \approx 0.133333\\]

Mapping the complete sequence of all 16 representable points on this digital ruler reveals the following available real values: 

`-1.0`, `-0.8667`, `-0.7333`, `-0.6000`, `-0.4667`, `-0.3333`, `-0.2000`, `-0.0667`, `0.0667`, `0.2000`, `0.3333`, `0.4667`, `0.6000`, `0.7333`, `0.8667`, `1.0`.

The structural gap between `0.0667` and `0.2000` spans a massive `0.1333` units. Any floating-point activation drifting inside this gap loses its mathematical identity entirely. Values such as `0.08`, `0.12`, and `0.15`—which easily maintained separation under \\(\text{Float32}\\) and \\(\text{INT8}\\) configurations—are crushed into the exact same point. Moving an execution pipeline down to 4 bits represents a profound architectural shift that fundamentally degrades the expressive capacity of the network layers.

---

## What Is Actually Lost

The distortion injected by quantization equations cannot be accurately modeled as standard, random Gaussian noise. It represents a systematic, permanent loss of numerical differentiability. 

When distinct floating-point values collapse onto an identical uniform grid coordinate, the subsequent layers in the execution graph receive uniform inputs where they historically processed detailed features. If the learned parameters of a pre-trained transformer model rely on those fine differences to route attention scores or compute word token probabilities, the execution pathway breaks down. The fine decision boundaries established in high-precision space are fundamentally erased from the silicon.

This degradation is an unyielding physical constraint of low-precision compute architectures. The neural network must execute its entire analytical workload using exclusively the numbers that survive the grid mapping. If the pre-trained weights require finer structural alignment than the computed step size permits, model accuracy drops precipitously. If the underlying mathematical paths are resilient to uniform grouping, the system unlocks massive hardware execution throughput for zero functional cost. 

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

Quantization is not a casual software compression flag; it is the forceful execution of neural network logic inside a rigid, finite grid of uniformly spaced numbers. The physical size of the gaps between those numbers is determined entirely by the boundary selections and the available register bit-width.

When evaluating a model for hardware deployment, the primary objective is to resolve a single engineering question: does the chosen uniform step size preserve the core differentiability required for the network to compute accurate predictions? If the step size maintains those differences across every layer block, quantization delivers massive compute speedups and drastic memory reductions with zero loss. If the step size violates those boundaries, the internal representation corrupts. Tracing, predicting, and mitigating those specific points of structural breakdown is the foundational focus of systems engineering.

---

## Key Terms — Quick Reference

The following architectural definitions form the core technical vocabulary used throughout this book. While subsequent chapters provide dedicated systems analyses of each mechanism, this reference map illustrates how the components interface within an execution pipeline.

| Term | Meaning | Chapter Reference |
|---|---|---|
| **Grid** | The fixed, finite set of representable integer levels determined by bit-width (e.g., 256 levels for 8-bit quantization). | Ch. 2 |
| **Scale (S / \\(\Delta\\))** | The uniform step size between adjacent grid points used to map real floating-point values to integers and vice versa. | Ch. 3 |
| **Zero-point (Z)** | An integer offset value that aligns the physical integer grid's zero marker with the true real-valued zero point. | Ch. 3 |
| **Boundary** | A precise structural intersection in the model graph where the quantization parameters (scale, zero-point) change. | Ch. 6 |
| **Domain** | The mathematical triple (Scale, Zero-point, Bit-width) that dictates how an integer tensor is interpreted. | Ch. 6 |
| **Requantization** | The process of downscaling a wide \\(\text{INT32}\\) intermediate accumulator sum back into a standard \\(\text{INT8}\\) layout. | Ch. 7 |
| **Fusion** | The compilation technique of merging consecutive operators (e.g., MatMul + Bias + ReLU) into a single execution kernel. | Ch. 8 |
| **Observer** | A diagnostic module injected during calibration to track and record the statistical dynamic range of activations. | Ch. 9 |
| **Accumulator** | A high-precision hardware register (typically \\(\text{INT32}\\)) used to safely sum partial matrix products without overflow. | Ch. 6 |
| **Calibration** | The execution of a representative validation dataset through the network to determine optimal activation scale factors. | Ch. 9 |
