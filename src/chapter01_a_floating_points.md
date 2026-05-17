# Chapter 01A: Floating-Point Fundamentals for Quantization

## Why This Chapter Exists

Quantization can be understood correctly only after establishing what is being compressed. In modern deep-learning systems, that baseline representation is typically Float32.

This chapter develops a rigorous yet accessible understanding of Float32 and then contrasts it with INT8 to identify precisely where quantization error is introduced.

---

## 1. Fixed-Point vs Floating-Point

Consider an 8-digit fixed-point representation in which the radix point is permanently fixed at the midpoint:

```text
[d][d][d][d].[d][d][d][d]
```

This format can represent values such as `0012.5000`, but it fails at both numerical extremes:

1. Large magnitudes overflow quickly.
2. Very small values lose precision quickly.

Floating-point addresses this limitation by storing:

1. Significant information (fraction bits plus an implied leading 1 for normalized values).
2. A scale term (exponent).

This is the same principle as scientific notation:

\\[
2.99792458 \times 10^8,
\quad
5.3 \times 10^{-11}
\\]

Digital hardware implements this in base-2 rather than base-10.

---

## 2. Float32 Bit Layout (IEEE-754)

Float32 uses 32 bits partitioned into three fields:

```text
31 30          23 22                                 0
+---+-------------+------------------------------------+
| S |  Exponent   |              Fraction              |
+---+-------------+------------------------------------+
 1 bit   8 bits                 23 bits
```

Its numerical value is defined as:

\\[
\\text{Value} = (-1)^S \times \\text{Significand} \times 2^{(E - 127)}
\\]

For normalized Float32 values:

\\[
\\text{Significand} = 1 + \\text{Fraction}
\\]

Where:

1. `S` is the sign bit (`0` for positive, `1` for negative).
2. `E` is the stored exponent (with bias 127).
3. `Fraction` is the 23-bit stored field to the right of the binary point.

Terminology note:

1. `Fraction` refers to the stored bit field.
2. `Significand` refers to the effective precision value used in computation.
3. `Mantissa` is a legacy informal term; this chapter uses `Fraction` and `Significand` for precision and clarity.

Because normalized binary numbers are of the form `1.xxxxx`, the leading `1` is implicit. Consequently, Float32 provides 24 effective bits of precision (1 implicit + 23 stored).

---

## 3. Worked Encoding Example: 2.5

Step 1: Convert to binary.

\\[
2.5_{10} = 10.1_2
\\]

Step 2: Normalize.

\\[
10.1_2 = 1.01_2 \times 2^1
\\]

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

The spacing (ULP) in that window is:

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

---

## 5. What Changes in INT8

INT8 provides exactly 256 discrete codes.

After quantization, those 256 codes must cover a chosen real range `[r_min, r_max]` uniformly.

The step size (scale) is:

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

This loss of distinguishability is a primary source of quantization error.

---

## 6. Float16 and BFloat16 at a Glance

When reducing precision from Float32, most schemes primarily reduce fraction precision.

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
