# Chapter 01A: Floating-Point Fundamentals for Quantization

## Context for Quantization

This chapter establishes the numerical foundations required for the rest of the book. It explains how Float32 represents values in hardware and then uses that foundation to make later quantization behavior easier to interpret.

---

## 1. Fixed-Point vs Floating-Point

Consider an 8-digit fixed-point representation (decimal point position never moves) in which the radix point is permanently fixed at the midpoint. Here, the radix point (separator between whole and fractional parts) splits the whole-number portion from the fractional portion.

```text
[d][d][d][d].[d][d][d][d]
```

This format can represent values such as `0012.5000`, but it fails at both numerical extremes:

1. Large magnitudes overflow quickly.
2. Very small values lose precision quickly.

Floating-point (scientific-notation style binary encoding) addresses this limitation by storing:

1. Significant information (precision bits).
2. A scale term (exponent).

\\[
2.99792458 \times 10^8,
\quad
5.3 \times 10^{-11}
\\]

Digital hardware implements this in base-2 rather than base-10, but the structural idea remains the same: store a sign, store a scale factor, and store precision bits.

---

## 2. Float32 Bit Layout (IEEE-754)

Float32 is designed to preserve two properties at the same time:

1. Large dynamic range (the ability to represent both very small and very large magnitudes).
2. Useful local precision within each magnitude range.

To achieve this, a 32-bit word is split into three coordinated parts:

1. Sign information,
2. Scale information, and
3. Precision information.

In IEEE-754 (standard binary floating-point format) notation, these are stored as `Sign`, `Exponent`, and `Fraction`.

The bit layout is:

```text
31 30          23 22                                 0
+---+-------------+------------------------------------+
| S |  Exponent   |              Fraction              |
+---+-------------+------------------------------------+
 1 bit   8 bits                 23 bits
```

Meaning of each field:

1. `Sign (S)`: controls polarity (`0` positive, `1` negative).
2. `Exponent (E)`: selects the power-of-two scale window (magnitude region currently being used).
3. `Fraction`: stores local detail within the selected window.


**Normalization** (forcing one canonical binary form) solves representational redundancy. Without normalization, the same value could be written in many equivalent forms, such as `1.01 x 2^3` and `0.101 x 2^4`. If multiple bit patterns can represent the same real number, precision is wasted and comparisons become less clean.

Float32 avoids this by enforcing normalized form.

A normalized binary value is written with exactly one non-zero digit to the left of the binary point. In base-2, that digit is always `1`.

Examples:

1. `1.01 \times 2^3` is normalized.
2. `0.101 \times 2^4` is not normalized (it should be rewritten as `1.01 \times 2^3`).

Because the leading digit is always `1` for normalized values, IEEE-754 does not store that bit explicitly. This is called the **implicit leading 1** (assumed first bit in normalized form).

Why this matters:

1. One bit is saved in storage.
2. Effective precision increases by one bit.

Therefore, in Float32, the stored 23-bit fraction behaves like 24 bits of precision for normalized numbers (implicit `1` plus 23 stored bits). The formal value equation is:

The term $(-1)^S$ in this equation acts as a sign switch:

1. If `S = 0`, then $(-1)^S = +1$.
2. If `S = 1`, then $(-1)^S = -1$.

So this factor applies positive or negative sign to the final value.

[
\text{Value} = (-1)^S \times \text{Significand} \times 2^{(E - 127)}
]

Here, `Significand` means the precision value used after restoring the implicit leading `1`.

Float32 stores exponent using a bias (fixed offset added to true exponent) of `127` so that both positive and negative real exponents can be represented with an unsigned 8-bit field.
Bias allows the exponent to be stored as an unsigned integer while preserving numerical ordering and simplifying hardware comparison.

Interpretation examples:

1. Stored exponent `E = 127` means real exponent `0`.
2. Stored exponent `E = 128` means real exponent `+1`.
3. Stored exponent `E = 126` means real exponent `-1`.

So the stored field acts like a shifted exponent scale centered around 127.

\[
\text{Significand} = 1 + \text{Fraction}
\]

Terminology note:

1. `Fraction` refers to the stored bit field.
2. `Significand` refers to the effective precision value used in computation.
3. `Mantissa` is a legacy informal term; this chapter uses `Fraction` and `Significand` for precision and clarity.

This is why normalized Float32 provides 24 effective bits of precision (1 implicit + 23 stored), even though only 23 fraction bits are physically present in memory.

### Edge Case: Subnormal Numbers

When the exponent field is all zeros (`E = 0`), the representation enters the **subnormal** regime (very small values near zero).

In this regime, the implicit leading `1` is no longer used:

\[
\text{Value}_{\text{subnormal}} = (-1)^S \times (0 + \text{Fraction}) \times 2^{-126}
\]

Practical implication:

1. Values can fade toward zero more gradually instead of dropping to zero abruptly at the smallest normal number.
2. Precision is lower than in normalized numbers.
3. On some accelerators, subnormals are flushed to zero for speed, which can matter for tiny gradients.

---

## 3. Worked Encoding Example: 2.5

Step 1: Convert to binary.

\[
2.5_{10} = 10.1_2
\]

Step 2: Normalize.

\[
10.1_2 = 1.01_2 \times 2^1
\]

Step 3: Build each field.

1. Sign bit: `0` (positive).
2. True exponent: `1`, therefore stored exponent is `1 + 127 = 128 = 10000000_2`.
3. Fraction field: bits after `1.` are `01`, padded to 23 bits:

```text
01000000000000000000000
```

Step 4: Pack fields.

```text
0 | 10000000 | 01000000000000000000000
```

Full 32-bit representation:

```text
01000000001000000000000000000000
```

Hexadecimal representation:

```text
0x40200000
```

Therefore, `2.5` appears in Float32 memory as `0x40200000`.

---

## 4. Why Float32 Precision Is Non-Uniform

For a fixed exponent, normalized values lie within a window:

\[
[2^e,\;2^{e+1})
\]

Within each window, Float32 provides exactly $2^{23}$ distinct normalized values, because the stored fraction field has 23 bits.

The spacing in that window is called ULP (Unit in the Last Place, smallest local representable step), which means the smallest gap between adjacent representable values in that local range:

\[
\Delta_e = 2^{e-23}
\]

Therefore, spacing is deterministic and doubles whenever $e$ increases by 1.

Examples:

1. In $[1,2)$, $e=0$, so spacing is $2^{-23} \approx 1.19 \times 10^{-7}$.
2. In $[1024,2048)$, $e=10$, so spacing is $2^{-13} = 1/8192 \approx 1.22 \times 10^{-4}$.

Both windows contain the same number of representable values ($2^{23}$), but the larger window has wider gaps between adjacent values.

Practical consequence:

1. Near small magnitudes, numerical resolution is fine.
2. At large magnitudes, numerical resolution is coarser.

This trade-off is highly useful in machine learning:

1. Small gradients remain representable.
2. Large activations remain representable without immediate overflow.

### Visual Intuition: Sliding Window vs Fixed Grid

Float32 behaves like a sliding precision window. INT8 behaves like a fixed uniform grid.

```text
Float32 (sliding window):

Near 1.0                  Near 1024
|.|.|.|.|.|.|.|.|         |....|....|....|....|
tiny gaps                 wider gaps

INT8 (fixed grid over chosen range):

|---|---|---|---|---|---|---|---|
same gap everywhere inside that range
```

Interpretation:

1. Float32 adapts spacing with magnitude, giving fine local detail near smaller values and coarser spacing at larger values.
2. INT8 uses one constant step across the selected range, which is simpler and faster but less expressive.

---

## 5. What Changes in INT8

INT8 provides exactly 256 discrete codes.

Quantization (mapping many real values to fewer codes) maps a continuous floating-point range into a finite set of integer codes.

After quantization, those 256 codes must cover a chosen real range `[r_min, r_max]` uniformly.

Here:

1. `r_min` is the smallest real value chosen for coverage.
2. `r_max` is the largest real value chosen for coverage.

The step size (scale, real value per code increment) is:

\[
\Delta = \frac{r_{\max} - r_{\min}}{255}
\]

For example, with range `[-1.0, 1.0]`:

\[
\Delta = \frac{1.0 - (-1.0)}{255} = \frac{2}{255} \approx 0.007843
\]

Values separated by less than approximately `0.007843` may map to the same quantized code.

Example:

1. `0.50000` and `0.50400` may quantize to the same INT8 level.
2. Their difference is then lost.

This loss of distinguishability is a primary source of quantization error (difference from original real value).

---

## 6. Float16 and BFloat16 at a Glance

When reducing precision from Float32, most schemes primarily reduce fraction precision (how finely nearby values can differ).

| Format | Exponent Bits | Fraction Bits | Dynamic Range | Local Precision |
| :-- | :--: | :--: | :-- | :-- |
| Float32 | 8 | 23 | Very wide | High |
| Float16 | 5 | 10 | Narrower | Medium |
| BFloat16 | 8 | 7 | Wide (close to Float32) | Lower |

Interpretation:

1. More exponent bits improve dynamic range.
2. More fraction bits improve local precision.

---

## 7. Takeaway for Quantization

Float32 derives its practical strength from balancing dynamic range and precision through exponent and fraction fields.

Quantization (for example, INT8) replaces that adaptive floating grid with a much smaller uniform grid. This improves memory efficiency and execution speed but introduces irreversible approximation.

Subsequent chapters formalize this mapping through scale and zero-point and then track how the resulting error propagates through real model pipelines.


