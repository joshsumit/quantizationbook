# Chapter 6: The Quantized Graph

## Beyond Individual Values

Previously, we looked at quantization as an operation on individual, isolated numbers: a real number is mapped to a grid point, a small amount of error is introduced, and that rounded value is passed on to the next step. But a neural network is not a collection of independent values. It is a computation graph — a sequence of operators where the output of one becomes the input to the next.

In a quantized graph, each operator has its own scale and zero-point. Those scales define *domains* — a *domain* is simply the (scale, zero-point, precision) triple that governs how integers are interpreted as real values at a given point in the graph. Crossing domains requires an explicit conversion step. The output domain of one operator is not the input domain of the next. The scale that was appropriate for the first layer's activations has no reason to match what the second layer expects. Something must happen at every junction where one domain ends and another begins.

---

## Quantization Boundaries

A *quantization boundary* is the point in the graph where the numeric domain changes — where the scale and zero-point of one operator's output differ from the scale and zero-point expected by the next operator's input.

Consider a simple two-layer network: Linear₁ followed by Linear₂. In float32, the output of Linear₁ flows directly into Linear₂ — there is no domain mismatch because float32 is a universal representation. In a quantized graph, this is no longer true.

Linear₁ has its own output scale \\(S_1\\) and zero-point \\(Z_1\\), determined by its activation distribution. Linear₂ has its own input scale \\(S_2\\) and zero-point \\(Z_2\\), determined by what its weights were calibrated against. If \\(S_1 \neq S_2\\) or \\(Z_1 \neq Z_2\\), the output of Linear₁ cannot be consumed directly by Linear₂. The integers produced by one layer mean different real values under the mapping expected by the next.
This mismatch point is a boundary. At every boundary, the values must be converted from one domain to another — rescaled, rounded, and clamped. The boundary exists, and it is not optional. Boundaries can only disappear when operators are *fused* — merged into a single combined operation so the conversion happens once at the chain’s output rather than at each intermediate step. (Fusion is covered in Chapter 8.) Boundaries can also disappear when multiple operators intentionally share a single domain.

> **📊 INSERT DIAGRAM: Quantization Boundaries in a Two-Layer Network**
>
> A horizontal graph showing two quantized linear layers with their domains:
>
> ```
> Float input ──→ [Quantize to S₀,Z₀] ──→ INT8
>                                          │
>                                   ┌──────┴──────┐
>                                   │  Linear₁     │  Domain₁: (S₁, Z₁, int8)
>                                   │  int8×int8   │
>                                   │  → int32 acc │
>                                   └──────┬──────┘
>                                          │
>                              ╔══════╩══════╗
>                              ║  BOUNDARY    ║  ← Requantize: int32 → int8
>                              ║  S₁ ≠ S₂     ║     (rounding + clamping here)
>                              ╚══════╦══════╝
>                                          │
>                                   ┌──────┴──────┐
>                                   │  Linear₂     │  Domain₂: (S₂, Z₂, int8)
>                                   │  int8×int8   │
>                                   │  → int32 acc │
>                                   └──────┬──────┘
>                                          │
>                              [Dequantize] ──→ Float output
> ```
>
> Annotations:
> - Highlight the BOUNDARY box: "Every boundary = one requantization = one rounding + one clamping"
> - Show that Domain₁ and Domain₂ have different scales (e.g., S₁=0.020, S₂=0.035)
> - Note: "In float32, this boundary doesn't exist — values flow freely between layers"


### Worked Example: A Boundary in Numbers

To see exactly what happens at a boundary, trace a single value through two layers with different scales.

**Setup.** Layer 1 produces an activation value whose true float32 value is \\(r = 2.13\\). Layer 1's output domain is \\((S_1 = 0.020, Z_1 = 0)\\) (symmetric int8). Layer 2's input domain is \\((S_2 = 0.035, Z_2 = 0)\\).

**Step 1 — Layer 1 quantizes the output.**

$$q_1 = \text{round}\!\left(\frac{2.13}{0.020}\right) = \text{round}(106.5) = 107 \qquad r_1' = 107 \times 0.020 = 2.140$$

Rounding error: \\(|2.13 - 2.14| = 0.010\\).

**Step 2 — At the boundary, convert from Domain 1 to Domain 2.**

The int8 code 107 in Domain 1 represents \\(r = 2.140\\). Domain 2 needs this value re-encoded with its own scale:

$$q_2 = \text{round}\!\left(\frac{2.140}{0.035}\right) = \text{round}(61.14) = 61 \qquad r_2' = 61 \times 0.035 = 2.135$$

Requantization error: \\(|2.140 - 2.135| = 0.005\\).

**Total error from float to Layer 2 input:** \\(|2.13 - 2.135| = 0.005\\). In this case the two rounding errors partially cancelled (by luck). In general they can add or partially cancel — the point is that each boundary *introduces* a rounding event, and across many boundaries the errors accumulate.

**What if the domains matched?** If \\(S_2 = S_1 = 0.020\\), then code 107 would be passed directly to Layer 2 with no conversion. Total error remains at 0.010 from Step 1 — no additional boundary error. This is why *scale alignment* (enforcing matching scales) and *fusion* (eliminating boundaries entirely) are the primary tools for reducing graph-level error.
---

## Accumulator Domains

Within a single operator, quantization creates a second kind of domain transition — one that is less obvious but equally important.

When two int8 values are multiplied, the result requires more than 8 bits. The product of two 8-bit integers can be as large as \\(127 \times 127 = 16{,}129\\), which does not fit in an 8-bit integer (maximum value 127 for signed, 255 for unsigned). When these products are summed across a dot product — say, 512 multiplications accumulated into a single result — the accumulated value can reach:

$$512 \times 127 \times 127 = 8{,}258{,}048$$

This number requires 24 bits to represent.

The exact bit requirement follows a formula. For a dot product of \\(N\\) pairs of \\(b\\)-bit integers, each in the range \\([0, 2^b - 1]\\), the maximum possible accumulator value is \\(N \times (2^b - 1)^2\\). The bits needed to represent this value are:

$$\text{bits}_{\text{acc}} = \lceil \log_2(N \times (2^b - 1)^2) \rceil = 2b + \lceil \log_2(N) \rceil$$

For int8 (\\(b = 8\\)) with a dot product width of \\(N = 1024\\) (a common hidden dimension):

$$\text{bits}_{\text{acc}} = 2 \times 8 + \lceil \log_2(1024) \rceil = 16 + 10 = 26 \text{ bits}$$

An int16 accumulator (16 bits, max value 65,535) overflows at \\(N = 4\\) — after just four multiply-accumulates. An int32 accumulator (32 bits, max value ~2.1 billion) handles dot products up to \\(N = 65{,}536\\) before overflowing — safely above any realistic layer width.

This is not an implementation detail. It is a correctness guarantee. In mainstream int8 kernels on CPUs, GPUs, and NPUs, the int8 × int8 multiply-accumulate produces int32 results. Some edge-device DSPs use int16 accumulators for power savings — these devices can overflow on narrow dot products (even a handful of worst-case signed MACs can exceed int16 range), producing incorrect outputs. *When an accumulator overflows*, the result wraps around: a sum that should be 40,000 becomes \\(40{,}000 - 65{,}536 = -25{,}536\\) in a signed int16 accumulator. The model silently produces garbage. This is the "Numerical Explosion" failure that the bit-growth formula predicts and that Chapter 4's hardware table warns about.

The consequence is that every linear layer and convolution in a quantized model operates in two domains: the inputs arrive as int8, and the accumulation produces int32. The output of the operator — before any conversion — is int32, not int8.

---

## Bias Lives in the Accumulator Domain

Neural network layers typically add a bias term after the matrix multiply: \\(y = Wx + b\\). In the quantized graph, the multiply-accumulate produces an int32 result. The bias must be added to this int32 value.

If bias were quantized to int8, it would be restricted to 256 levels. But the int32 accumulator can hold values in the millions. An int8 bias added to an int32 accumulator is a rounding error at best and a numerical catastrophe at worst — the bias would be quantized so coarsely relative to the accumulator range that it would effectively vanish or snap to a wildly wrong value.

Bias is therefore quantized to int32 — matching the accumulator domain. Its scale is typically set to \\(S_w \times S_x\\) (the product of the weight scale and the input scale), per output channel if weights are per-channel quantized. This ensures the bias occupies the same numeric domain as the accumulated products. This is not a tunable choice. It is a structural requirement: bias must live in the accumulator domain, or the addition is meaningless.

**Worked example: why int8 bias fails.** Suppose \\(S_w = 0.02902\\) (from Chapter 3) and \\(S_x = 0.00523\\) (input scale). The accumulator scale is \\(S_{\text{acc}} = S_w \times S_x = 0.0001517\\). A bias value of 0.5 in float maps to int32 as \\(0.5 / 0.0001517 \approx 3{,}296\\) — well represented in the int32 range.

If bias were int8 with the same scale, it could only represent values at multiples of 0.0001517 up to \\(127 \times 0.0001517 = 0.019\\). A bias of 0.5 would overflow int8 entirely. Even if we gave bias its own wider scale — say \\(S_b = 0.004\\) to cover [-0.5, 0.5] — the bias step size (0.004) would be 26× coarser than the accumulator's step size (0.0001517). Adding a bias quantized at 0.004 steps to an accumulator operating at 0.0001517 steps is like adding a measurement rounded to the nearest meter to one measured in centimeters.

---

## The Output Problem

The accumulator is int32, but the next operator expects int8 inputs. This is the structural problem that defines quantized inference.

After the multiply-accumulate and bias addition, the result sits in an int32 register. It must be converted to int8 before the next operator can consume it — because the next operator's int8 × int8 multiply-accumulate takes int8 inputs, and feeding it int32 values is not a valid operation.

This conversion is not a cast. The int32 value represents a specific real number under the accumulator's scale. The int8 value must represent the same real number (as closely as possible) under a different scale. The conversion involves rescaling, rounding, and clamping — an operation called *requantization*. It introduces error every time it is applied.

The domain ledger for a single linear layer:

```
Linear: int8 (Sx, Zx) × int8 (Sw, Zw) → int32 (Sw·Sx) → requant → int8 (Sy, Zy)
```

Every linear layer and convolution in the quantized graph creates this pattern: int8 inputs → int32 accumulation → conversion back to int8 for the next layer. The conversion back is mandatory. It cannot be skipped, because the next layer's hardware instructions require int8 inputs.

---

## Boundaries Are Structural

The number of these conversion points in a model is not a runtime decision. It is determined entirely by the graph topology.

Trace data through a three-layer model: Linear₁ → ReLU → Linear₂.

- Input \\(x\\) is int8.
- Linear₁: int8 × int8 → int32 accumulator. Bias added in int32. Output is int32.
- **Boundary 1**: int32 must be requantized to int8 for subsequent consumption. If ReLU is fused into the requantization epilogue — applied as a clamp during the int32 → int8 conversion, which is the common case — it does not introduce an additional boundary. If ReLU runs as a separate unfused op, it still preserves the activation domain (same scale and zero-point for positive values, zero output otherwise), so no extra boundary is needed for ReLU itself.
- Linear₂: int8 × int8 → int32 accumulator. Bias added in int32. Output is int32.
- **Boundary 2**: int32 must be requantized to int8 for the model's final output.

Two linear layers produce two boundaries. A 50-layer model produces roughly 50 boundaries (before fusion). A 100-layer model produces roughly 100.

This count is a fixed structural property of the model before optimization. It cannot be reduced by tuning parameters or choosing better calibration data. It can only be reduced by changing the graph itself — fusing operators so that intermediate boundaries disappear (Chapter 8 covers this in detail).

---

## Conceptual Consolidation

A quantized neural network is not a stack of layers with lower precision numbers. It is a graph of operators where each operator internally accumulates in int32, and every transition between operators requires converting from int32 back to int8. These conversion points — boundaries — are fixed by the graph topology.

When you look at a quantized model, count the boundaries. That count determines how many times values will be rounded and clamped during a single inference pass — and the cumulative effect is the dominant source of quantization failure.

Profiling tip: if your quantized model is slower than the floating-point baseline, boundary materialization and fallback are the first suspects. Count the requantization points, then check whether any are failing to fuse.
