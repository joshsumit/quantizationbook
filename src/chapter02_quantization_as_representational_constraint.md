# Chapter 2: Quantization as a Representational Constraint

## What Float32 Gives You

A 32-bit floating-point number (\\(\text{Float32}\\)) has \\(2^{32}\\) (approx. 4.3 billion) distinct bit patterns, which do not correspond to evenly spaced real numbers. Instead, these representable points are packed very closely together near zero and get wider apart as the values grow larger. Because of this layout, \\(\text{Float32}\\) offers very dense representable values for tiny numbers, but provides far fewer available values as you move up the number line. 

### The Sliding Window Mental Model

You can think of \\(\text{Float32}\\) as a sliding window over the number line. The exponent decides where the window is placed, and the mantissa decides how finely we divide that window. The key constraint is that each window contains \\(2^{23}\\) evenly spaced representable values. If the window gets wider to cover larger values, the gaps between those points must increase. Each window corresponds to the range between two consecutive powers of 2.

### Inside a 32-Bit Number

To see how this sliding window works on real hardware, we have to look at how a computer constructs a floating-point value from 32 bits of memory. The storage is split into three parts: 1 bit for the sign (positive or negative), 8 bits for the exponent, and 23 bits for the mantissa (which stores the fractional part).

The final real-world value is calculated using this formula:

\\[\text{Value} = (-1)^{\text{sign}} \times 2^{\text{exponent} - 127} \times (1 + \text{fraction})\\]

The exponent bits control the placement of our sliding window by scaling powers of 2. Inside that window, the 23 bits of the mantissa act as a fixed number of evenly spaced points (\\(2^{23}\\) points). Because of the way the formula is written, the mantissa encodes only the fractional part after the implicit leading 1.

### Walkthrough: Building a Value

Let us look at a concrete example to see how the mantissa and exponent work together. Suppose we want to represent the value 2.5. In binary floating-point, this is written as:

\\[2.5 = 2^1 \times 1.25\\]

To encode this into our formula:
- The **sign bit** is `0` (positive).
- The **exponent** bits must evaluate to a scale factor of \\(2^1\\). Using the formula's offset, we set the exponent bits to `128`, because \\(128 - 127 = 1\\).
- The **mantissa (fraction)** needs to handle the remaining `0.25`, because the formula automatically includes the implicit leading `1`. In binary fractional bits, `0.25` is exactly \\(1/4\\), which sets the very first mantissa bit to `1` and leaves the remaining 22 bits as `0`.

When the hardware decodes this bit pattern, it executes the formula:

\\[\text{Value} = (-1)^0 \times 2^{128 - 127} \times (1 + 0.25) = 1 \times 2^1 \times 1.25 = 2.5\\]

### How Exponents Impact Packing Density

Within any interval between two consecutive powers of 2 (such as 1 to 2, or 1024 to 2048), \\(\text{Float32}\\) represents exactly \\(2^{23}\\) distinct values within each range. Each exponent range always contains exactly \\(2^{23}\\) representable values; only the size of the range changes. 

Each such interval forms one "window" in our mental model.

**Scenario A: The Exponent Shrinks (High Density near Zero)**
Imagine our activation values are tiny values close to zero, where the exponent bits evaluate to `120`. This places our window between \\(2^{-7}\\) (\\(1/128 \approx 0.0078125\\)) and the next power of 2, \\(2^{-6}\\) (\\(0.015625\\delta\\)). The total distance covered by this range is tiny—only `0.0078125` units wide. Because our window must contain exactly \\(2^{23}\\) points, the distance between any two adjacent values in this region is:

\\[\text{Step Size} = \frac{0.0078125}{2^{23}} \approx 0.00000000093\\]

Because the gaps between values are less than one-billionth of a unit wide, the representation is incredibly dense, allowing the hardware to distinguish extremely small differences between values.

**Scenario B: The Exponent Grows (Sparse Spacing at Large Magnitudes)**
Now imagine the network processes a large outlier value, where the exponent bits evaluate to `137`. This shifts our window to cover the space between \\(2^{10}\\) (`1024`) and \\(2^{11}\\) (`2048`). The total size of this range is now a massive `1,024` units wide. Since the number of points stays constant, increasing the size of the range forces the spacing between points to grow. Our \\(2^{23}\\) points must stretch to cover this entire block, widening the distance between adjacent values:

\\[\text{Step Size} = \frac{1024}{2^{23}} \approx 0.000122\\]

Notice what just happened to the number space. Near zero, our resolution was under a billionth of a unit. At a magnitude of 1,024, the gaps between consecutive values have widened to over a ten-thousandth of a unit. If two values in this large region differ by a tiny fraction like `0.00001`, \\(\text{Float32}\\) cannot separate them anymore; they will round to the exact same bit pattern.

### Summary of the Constraints

\\(\text{Float32}\\) does not provide uniform precision. Near zero, values are densely packed and small differences are preserved. At large magnitudes, spacing widens and small differences disappear. As long as values stay near zero, the representation behaves like a smooth, continuous space. As values grow, that illusion breaks.

---

## What Int8 Gives You

An 8-bit integer (\\(\text{INT8}\\)) has exactly 256 distinct levels. That is your entire budget for the layer. There are no hidden decimals or sliding exponents; you have 256 unique points, and they must be spaced completely evenly across whatever range you assign them to cover.

Let us trace what happens when we use these 256 levels to cover a real-valued range from \\([-1.0, 1.0]\\). Those 256 discrete levels must stretch across a total distance of 2.0 units. The distance between two adjacent points on this grid is called the *step size* (or \\(\Delta\\)), and it is calculated with a simple formula:

\\[\Delta = \frac{r_{\max} - r_{\min}}{L - 1}\\]

In this formula, \\(L\\) is the number of available levels (which is 256 for \\(\text{INT8}\\)), and \\(r_{\max}\\) and \\(r_{\min}\\) are the edges of your real-world range. We use \\(L - 1\\) (255) in the denominator because we are counting the gaps *between* the points, not the points themselves. 

For our \\([-1.0, 1.0]\\) range, the step size calculation looks like this:

\\[\Delta = \frac{1.0 - (-1.0)}{256 - 1} = \frac{2.0}{255} \approx 0.00784\\]

Every real number that enters this layer must snap to the nearest point on this new grid. If a value comes in at 0.50000, it lands on a grid point. I## Key Terms — Quick Reference

These basic definitions will appear throughout the book. While each concept has its own dedicated chapter later on, this list provides a quick map of how the system fits together.

| Term | Meaning | Chapter |
|---|---|---|
| **Grid** | The fixed set of representable integer values (e.g., 256 levels for 8-bit quantization) | Ch. 2 |
| **Scale (S)** | The uniform step size between adjacent grid points used to translate integers back to real numbers | Ch. 3 |
| **Zero-point (Z)** | An integer offset that aligns the physical grid's zero level with the true real-valued zero | Ch. 3 |
| **Boundary** | A precise point in the execution graph where the quantization domain (scale, zero-point, precision) shifts | Ch. 6 |
| **Domain** | The structural triple containing scale, zero-point, and bit-precision that governs integer interpretation | Ch. 6 |
| **Requantization** | The process of rescaling a wide \\(\text{INT32}\\) accumulator result back into an \\(\text{INT8}\\) layout for subsequent layers | Ch. 7 |
| **Fusion** | Merging consecutive mathematical operations (e.g., Conv + BN + ReLU) into a single hardware kernel | Ch. 8 |
| **Observer** | A tracking module deployed during calibration to record the dynamic range of activations | Ch. 9 |
| **Accumulator** | A high-precision register (typically \\(\text{INT32}\\)) used to safely accumulate matrix multiplication partial sums | Ch. 6 |
| **Calibration** | The process of running representative validation data through a model to determine activation scale factors | Ch. 9 |f another value comes in at 0.50400, it will snap to that exact same grid point because the step size is not small enough to separate them. In \\(\text{Float32}\\), these were two different inputs that could trigger different paths in the network. In \\(\text{INT8}\\), they become identical, and the difference between them is completely erased.

---

## The Grid

Quantization works by laying a uniform grid over a specific range of numbers. Because the number of points on the grid is locked in place, the gaps between those points never change. Every single floating-point number is forced to round to the closest available grid marker. This behaves exactly like a standard physical ruler marked only in full millimeters: it cannot tell the difference between 3.2 mm and 3.3 mm because it simply does not have the lines to show that space.

Let us look at a practical example. Imagine a layer whose values run from \\([-1.0, 1.0]\\). We apply 8-bit uniform quantization with 256 levels. 

If we use a basic mapping rule that numbers the grid lines from 0 to \\(L-1\\), the formulas to convert a real number to an integer code (\\(q\\)), and back to a rounded real number (\\(r'\\)), look like this:

\\[q = \text{round}\!\left(\frac{r - r_{\min}}{\Delta}\right)\\]

\\[r' = r_{\min} + q \cdot \Delta\\]

Now let us watch how two values close to each other pass through these equations, using our minimum bound of \\(r_{\min} = -1.0\\) and our step size of \\(\Delta \approx 0.00784\\).

For an input value of 0.3021:

\\[q = \text{round}\!\left(\frac{0.3021 - (-1.0)}{0.00784}\right) = \text{round}(166.08) = 166\\]

\\[r' = -1.0 + 166 \times 0.00784 = 0.2980\\]

Now, let us do the same for a nearby input value of 0.3058:

\\[q = \text{round}\!\left(\frac{0.3058 - (-1.0)}{0.00784}\right) = \text{round}(166.55) = 167\\]

\\[r' = -1.0 + 167 \times 0.00784 = 0.3058\\]

In this case, the two inputs were far enough apart that they landed on separate integers (166 and 167). But notice what happened to 0.3021: it was shifted to 0.2980. The network downstream will now read 0.2980 instead of the original value.

---

### Worked Example: One Range, Three Outcomes

To see exactly how a grid treats different values, let us look at a single setup where three different things happen: clean rounding, maximum error, and out-of-bounds clipping. 

**The Setup:** A layer's activation values fall within the range \\([-1.2, 1.0]\\). We want to use asymmetric 8-bit quantization, which numbers our grid from integer code 0 to 255.

**Step 1: Calculate the grid scale and zero-point.**

\\[S = \frac{r_{\max} - r_{\min}}{255} = \frac{1.0 - (-1.2)}{255} = \frac{2.2}{255} \approx 0.008627\\]

\\[Z = \text{round}\!\left(\frac{0 - r_{\min}}{S}\right) = \text{round}\!\left(\frac{1.2}{0.008627}\right) = \text{round}(139.1) = 139\\]

This means the integer code 139 represents a real value of 0.0. Code 0 represents our lowest value (\\(-1.2\\)), and code 255 represents our highest value (\\(1.0\\)).

**Step 2: Process a normal, in-range value.** Let us take a real activation value of \\(r = 0.37\\). 

\\[q = \text{round}\!\left(\frac{0.37}{0.008627}\right) + 139 = \text{round}(42.89) + 139 = 43 + 139 = 182\\]

Now let us convert integer code 182 back into a real number to see the change:

\\[r' = (182 - 139) \times 0.008627 = 43 \times 0.008627 = 0.3710\\]

The difference between our original value and our quantized value is \\(|0.37 - 0.3710| = 0.0010\\). This is well inside the worst-case rounding error for this grid, which is half a step size (\\(S/2 \approx 0.0043\\)). The grid handled this value safely.

**Step 3: Process an out-of-range outlier value.** Now let us see what happens to a value that falls outside our grid limits, like \\(r = 1.5\\).

\\[q = \text{round}\!\left(\frac{1.5}{0.008627}\right) + 139 = \text{round}(173.8) + 139 = 174 + 139 = 313\\]

Because an unsigned 8-bit integer can only hold codes from 0 to 255, the hardware cannot store 313. It forcefully clamps the number down to the maximum boundary of 255. 

Let us convert that clamped code back to a real value:

\\[r' = (255 - 139) \times 0.008627 = 116 \times 0.008627 = 1.0007 \approx 1.0\\]

The error for this outlier is \\(|1.5 - 1.0| = 0.5\\). This error is 500 times larger than the rounding error we saw in Step 2. Because the value was outside our tracking limits, its true size was completely cut off at the edge. This is called *clipping error*.

**The Lesson:** The exact same grid handles these two numbers with completely different results. The in-range number lost almost nothing, while the outlier was severely distorted. This is why picking the right boundaries for your range determines whether a model works or breaks. 

The grid itself does not know which values are important to the network. It spreads its 256 points evenly across the range, regardless of where most of the numbers actually live. It uses this rigid structure because a constant step size allows hardware engines to run fast, cheap integer math. Later chapters will show how we can isolate channels or group numbers into smaller blocks to get better coverage, but the basic grid points always remain completely uniform.

---

## The Range–Precision Trade-Off

The step size of a layer depends on two competing factors: the width of the range you need to cover, and the number of grid lines you have available. Because your number of grid points is locked at 256 for 8-bit math, widening your boundaries automatically forces your step size to grow.

For instance, if we take those same 256 points and stretch them from our clean \\([-1.0, 1.0]\\) range out to a much wider range of \\([-10.0, 10.0]\\), our step size recalculates to:

\\[\Delta = \frac{20.0}{255} \approx 0.0784\\]

The step size is now ten times larger. Numbers that were easily separated before will now collapse into the same integer. Real values like 0.3021 and 0.3500—which were clearly separate numbers in \\(\text{Float32}\\)—now round to the exact same spot on the grid. This is the fundamental trade-off of quantization: *a wider range always gives you coarser details*. If you open up your boundaries to catch a few extreme outliers, you sacrifice the resolution for the rest of your data.

This problem gets much worse when we drop down to 4-bit math (\\(\text{INT4}\\)), where our total budget drops from 256 levels down to just 16. If we try to cover our baseline range of \\([-1.0, 1.0]\\) with only 16 levels, our step size opens up dramatically:

\\[\Delta = \frac{2.0}{15} \approx 0.1333\\]

If we list out all 16 available values on this ruler, they look like this: 0.0, 0.1333, 0.2667, 0.4, 0.5333, 0.6667, 0.8, 0.9333, 1.0667, 1.2, 1.3333, 1.4667, 1.6, 1.7333, 1.8667, and 2.0. 

Look at the massive gap between 0.0 and 0.1333. Any real number that falls inside that gap completely loses its identity. Numbers like 0.05, 0.08, and 0.12—which were perfectly distinct in \\(\text{Float32}\\) and \\(\text{INT8}\\)—all become the exact same point under an \\(\text{INT4}\\) limit. Moving to 4 bits is a major shift that fundamentally changes how much a model can express.

---

## What Is Actually Lost

The change caused by quantization is not just simple "noise." Instead, it is a permanent loss of the ability to tell numbers apart. 

When separate values collapse onto a single grid point, the layers further down the network receive the exact same input where they used to see distinct features. If the model's learned rules relied on those subtle differences to make a decision, that decision path is now broken. The fine boundary lines that existed in \\(\text{Float32}\\) are wiped away.

This is not a random glitch; it is an absolute physical constraint. The model must now perform its entire task using only the numbers that survive the grid. If the pre-trained rules require finer adjustments than the step size allows, the model's accuracy will drop. If they do not, quantization gives you massive efficiency benefits for free. Whether a model survives this transition depends entirely on how its natural numbers fit into the rigid grid lines we place over them.

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

Quantization is not just "using lower precision." It is forcing software code to run inside a rigid, finite grid of evenly spaced numbers. The size of the gaps between those numbers is determined directly by your boundaries and your available bit budget. 

When we evaluate a quantized model, our main job is to answer a single question: does our chosen step size preserve the core differences that the model needs to work? If the answer is yes for every layer, quantization gives us faster execution and smaller files for zero cost. If the answer is no, the model changes its behavior. Understanding exactly where and why those changes occur is what the rest of this book is about.

