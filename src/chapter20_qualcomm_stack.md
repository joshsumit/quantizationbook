# Chapter 20: The Qualcomm Stack — From Training to On-Device Inference

## Why a Dedicated Chapter on Qualcomm?

Qualcomm ships more AI-capable silicon than any other company on the planet. Every Snapdragon SoC in every Android flagship, every Windows-on-Arm laptop, every automotive ADAS module, every XR headset — all of them contain Qualcomm's Hexagon NPU. If you quantize a model and deploy it to an edge device, there is a strong probability that the target hardware is Qualcomm.

But Qualcomm's quantization stack is not just "export to ONNX and hope." It is a vertically integrated pipeline with its own quantization toolkit (AIMET), its own compilation SDK (QNN), its own runtime (QAIRT), and its own model hub (Qualcomm AI Hub). Each layer in this stack makes specific assumptions about how quantization is performed — and if you violate those assumptions, your model either runs slowly (falling back to CPU), produces garbage outputs (silent accuracy loss), or refuses to compile entirely.

This chapter walks through every layer of the stack, from "I have a trained PyTorch model" to "it runs at 30 FPS on a Snapdragon phone," with the quantization details at each stage.

---

## What Is "The Qualcomm Stack"?

The Qualcomm AI stack is a set of tools that form a pipeline. But — and this is a critical nuance — the pipeline is **not** strictly linear. There are **two quantization paths**, and choosing the right one is an expert-level decision:

```
┌─────────────────────────────────────────────────────────┐
│                    YOUR TRAINED MODEL                    │
│              (PyTorch / TensorFlow / ONNX)               │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
  ┌─────────────────────┐   ┌──────────────────────────┐
  │     PATH A: AIMET    │   │   PATH B: QNN Native     │
  │  (Optimization       │   │   (Convenience            │
  │   Engine)            │   │    Quantization)          │
  │                      │   │                           │
  │ • CLE, AdaRound, QAT │   │ • --input_list for calib  │
  │ • Per-layer diagnosis │   │ • MinMax / MSE scaling    │
  │ • Mixed-precision     │   │ • No QAT fallback        │
  │ • Full control        │   │ • Black-box               │
  └──────────┬───────────┘   └────────────┬──────────────┘
             │                            │
             │  ONNX + .encodings         │  Raw ONNX (no encodings)
             │                            │
             └────────────┬───────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│                        QNN SDK                           │
│           Qualcomm Neural Network SDK (Compiler)         │
│   • Model conversion  • Graph optimization                │
│   • Operator mapping  • Custom Op registration            │
│   • Quantization encoding (from AIMET or self-computed)   │
└────────────────────────┬────────────────────────────────┘
                         │  Compiled context binary (.bin)
                         ▼
┌─────────────────────────────────────────────────────────┐
│                       QAIRT                              │
│          Qualcomm AI Runtime (Execution)                 │
│   • Backend selection (CPU/GPU/HTP)                       │
│   • Memory management  • Inference execution              │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   HEXAGON NPU (HTP)                      │
│              Hardware tensor processor                    │
│   • INT8/INT16 native  • VTCM (fast on-chip memory)      │
│   • 256-byte vector width  • HVX instructions             │
└─────────────────────────────────────────────────────────┘
```

**Path A (AIMET → QNN)** is the **recommended path for production.** You get full control over quantization: CLE to rebalance weights, AdaRound to optimize rounding, QAT to fine-tune, and per-layer diagnostics to find which layers are hurting accuracy.

**Path B (QNN Native)** is the **quick-and-dirty path.** The `qnn-onnx-converter` has built-in quantization: pass `--input_list calibration_images.txt` instead of `--quantization_overrides`, and QNN will calibrate and quantize in one shot. This works well for simple CNNs (MobileNet, ResNet) where default min-max quantization is sufficient. But for transformers or any model where INT8 accuracy is not trivially good, Path B is a dead end — if accuracy drops, you have no recovery tools.

Each layer has a specific job:
- **AIMET** — makes your model *quantization-friendly* while you still have access to training data and gradients. The **optimization engine.**
- **QNN** — *compiles* the quantized model into a graph that the hardware can execute. The **execution engine.** Also capable of basic quantization (Path B), but this is a convenience feature, not its primary role.
- **QAIRT** — *runs* the compiled graph on-device, managing memory and dispatching to the right hardware backend.
- **Qualcomm AI Hub** — a cloud service that lets you skip some of these steps for common models, providing pre-optimized binaries and profiling.

The rest of this chapter walks through each layer in detail.

## The Hardware: Hexagon NPU

Before understanding the software, you need to understand what the software is targeting.

Every modern Snapdragon SoC contains a **Hexagon processor** with a dedicated **HTP (Hexagon Tensor Processor)** block. This is Qualcomm's neural processing unit. It is not a GPU. It is not a general-purpose CPU. It is a fixed-function/programmable hybrid designed specifically for tensor operations.

**What HTP supports natively:**
- **INT8** multiply-accumulate with INT32 accumulators — the primary quantized inference datatype.
- **INT16** multiply-accumulate — for layers where INT8 is not accurate enough.
- **FP16** — supported but slower than INT8 on HTP; often used as a fallback.
- **HVX (Hexagon Vector eXtensions)** — 128-byte or 256-byte vector operations (depending on generation). Think of this as Qualcomm's version of AVX-512, but for mobile.
- **VTCM (Vector Tightly Coupled Memory)** — a small, fast on-chip SRAM (typically 256KB–8MB depending on generation) that acts like an L0 cache for tensor data. Keeping activations in VTCM is the single most important performance optimization on Hexagon.

**What HTP does NOT support natively:**
- FP32 inference — runs on the CPU fallback, which is 10–50× slower.
- Arbitrary dynamic shapes — the compiler needs static or bounded shapes.
- Operators outside the supported list — unsupported ops fall back to CPU, serializing the pipeline and killing performance.

**Why this matters for quantization:** If your model produces an INT8-quantized graph where every operator is HTP-compatible, the entire inference runs on the NPU. If even one operator is unsupported, the graph is split: part runs on HTP, data is copied back to CPU, the unsupported op runs on CPU, data is copied back to HTP, and the remaining graph continues. This round-trip typically costs 2–5ms per split — which can be more than the entire inference time of a well-optimized model.

The quantization choices you make (symmetric vs. asymmetric, per-tensor vs. per-channel, which layers to leave in FP16) directly determine whether operators land on HTP or fall back to CPU.

**Generation differences:**

| Snapdragon Gen | Hexagon Version | VTCM Size | INT8 TOPS | Key Quantization Feature |
|----------------|----------------|-----------|-----------|--------------------------|
| 865 | V66 | 256KB | ~7 | Basic INT8 |
| 888 | V69 | 512KB | ~12 | INT16 support |
| 8 Gen 1 | V69+ | 1MB | ~15 | Shared memory architecture |
| 8 Gen 2 | V73 | 2MB | ~18 | INT4 weight support |
| 8 Gen 3 | V75 | 4MB | ~45 | Micro-tile inference |
| X Elite | V73 (multi-core) | 8MB | ~75 | Multi-NPU, LLM support |

The trend: more VTCM, more TOPS, and wider quantization support with each generation. Models that run well on Gen 1 will fly on Gen 3 — but models that were never properly quantized will crawl on all of them.

---

## AIMET — AI Model Efficiency Toolkit

AIMET is Qualcomm's open-source quantization and compression toolkit. It is a Python library that wraps around PyTorch (and optionally TensorFlow/Keras) and provides quantization techniques specifically designed to produce models that run well on Qualcomm hardware.

AIMET is **not** a compiler. It does not produce device-ready binaries. Its job is to take a floating-point model and produce a quantization-friendly version — with quantization parameters (scales and zero-points) baked in — that can then be handed to QNN for compilation.

Think of AIMET as the "training-side" tool and QNN as the "deployment-side" tool. AIMET runs wherever you train (cloud GPU, workstation). QNN runs wherever you compile (workstation, CI pipeline, or the device itself).

**Install:**
```bash
pip install aimet-torch  # For PyTorch
# or
pip install aimet-tensorflow  # For TF
```

AIMET is open source: https://github.com/quic/aimet

---

### What AIMET Actually Does

AIMET provides four core capabilities:

1. **Post-Training Quantization (PTQ)** — quantize a model without any retraining. You provide a small calibration dataset (typically 500–2000 samples), AIMET runs inference to observe activation ranges, and it computes optimal quantization parameters.

2. **Quantization-Aware Training (QAT)** — insert fake-quantization nodes into the training graph and fine-tune the model so it learns to be robust to quantization noise. Requires training data and a few epochs of training.

3. **Cross-Layer Equalization (CLE)** — a mathematical technique that redistributes weight ranges across consecutive layers to make them more uniform and easier to quantize. This is a free improvement — no training data required.

4. **AdaRound (Adaptive Rounding)** — instead of rounding each weight to the nearest integer (which is what naive quantization does), AdaRound learns the optimal rounding direction (up or down) for each weight to minimize the overall layer output error. Requires a calibration dataset but not full retraining.

These techniques can be composed. A typical pipeline is: CLE → AdaRound → QAT. Each step progressively improves quantization quality.

---

### AIMET Quantization Schemes

AIMET supports the quantization schemes that Hexagon HTP expects:

**Symmetric quantization:**
$$q = \text{round}\left(\frac{x}{S}\right), \quad S = \frac{\max(|x|)}{2^{b-1} - 1}$$

No zero-point. The floating-point zero maps exactly to the integer zero. This is what HTP prefers for weights.

**Asymmetric quantization:**
$$q = \text{round}\left(\frac{x}{S}\right) + Z, \quad S = \frac{x_{\max} - x_{\min}}{2^b - 1}, \quad Z = \text{round}\left(\frac{-x_{\min}}{S}\right)$$

Non-zero zero-point. More accurate for activations that are not centered around zero (e.g., post-ReLU activations that are always ≥ 0). HTP supports asymmetric quantization for activations.

**Granularity:**
- **Per-tensor** — one scale/zero-point for the entire tensor. Fastest, least accurate.
- **Per-channel** — one scale/zero-point per output channel (for weights). More accurate, supported on HTP.

**Bit-widths:**
- **INT8** — the primary target. Best performance on HTP.
- **INT16** — for sensitive layers. 2× slower than INT8 on HTP, but sometimes necessary.
- **INT4** — weight-only, on newer Hexagon generations (Gen 2+). Weights stored in INT4, compute in INT8.

**Practical rule:** Start with per-channel symmetric weights + per-tensor asymmetric activations, both INT8. This is the default that QNN expects, and it runs at full speed on HTP.

**Why HTP *specifically* prefers symmetric weights — the MAC hardware reason:**

This is not just a convention — it is a hardware constraint. Consider the full integer-only multiply-accumulate (MAC) expansion for an asymmetric weight:

$$Y = S_w \cdot S_x \cdot \sum_i (W_{q,i} - Z_w)(X_{q,i} - Z_x)$$

Expanding:

$$Y = S_w \cdot S_x \cdot \left[\sum_i W_{q,i} X_{q,i} - Z_w \sum_i X_{q,i} - Z_x \sum_i W_{q,i} + N \cdot Z_w \cdot Z_x \right]$$

That is four terms per MAC. With **symmetric weights** ($Z_w = 0$), the second and fourth terms vanish:

$$Y = S_w \cdot S_x \cdot \left[\sum_i W_{q,i} X_{q,i} - Z_x \sum_i W_{q,i}\right]$$

The $Z_x \sum_i W_{q,i}$ term is a **precomputable bias** — it depends only on weights and the activation zero-point, both of which are known at compile time. QNN precomputes this and folds it into the bias. The HTP hardware then only needs to execute the raw $\sum_i W_{q,i} X_{q,i}$ MAC — which is exactly what its integer-only pipeline is optimized for.

If you use asymmetric weights ($Z_w \neq 0$), the hardware must compute the extra $Z_w \sum_i X_{q,i}$ term at runtime. This cannot be precomputed because $X_{q,i}$ changes every inference. The result: additional instructions per MAC, wasted cycles, and potentially a fallback to a slower kernel path.

**Bottom line:** Symmetric weights are not optional on HTP. They are the only path to the fast MAC pipeline.

---

**Group-wise Quantization (for LLMs):**

Per-channel quantization assigns one scale per output channel. For a weight matrix of shape $[\text{out}, \text{in}]$, that is $\text{out}$ scales. But for large language models, per-channel is sometimes too coarse — a single channel in a 4096-wide projection might have weights spanning a wide range.

**Group-wise quantization** splits each channel into groups of $G$ consecutive elements and assigns a separate scale to each group:

$$W[i, j] \approx S_{i, \lfloor j/G \rfloor} \cdot W_q[i, j]$$

Common group sizes: $G = 32$, $G = 64$, $G = 128$.

For a weight matrix of shape $[4096, 4096]$ with $G = 128$:
- Per-channel: 4096 scales
- Group-wise: $4096 \times (4096 / 128) = 131,072$ scales

More scales = finer granularity = less quantization error, but more metadata stored in the context binary.

**AIMET group-wise support:**
```python
from aimet_torch.quantsim import QuantizationSimModel

sim = QuantizationSimModel(
    model=model,
    quant_scheme='tf_enhanced',
    default_output_bw=8,
    default_param_bw=4,                    # INT4 weights
    config_file='quantsim_config.json',    # Specifies per-group settings
    dummy_input=dummy_input
)
```

The config file specifies group-wise quantization:
```json
{
    "defaults": {
        "params": {
            "bitwidth": 4,
            "is_symmetric": true
        }
    },
    "params": {
        "weight": {
            "bitwidth": 4,
            "is_symmetric": true,
            "encoding_per_group": true,
            "group_size": 128
        }
    }
}
```

In the exported encodings JSON, group-wise scales appear as arrays:
```json
{
    "param_encodings": {
        "layers.0.self_attn.q_proj.weight": [
            {
                "bitwidth": 4,
                "is_symmetric": true,
                "scale": [0.0034, 0.0028, 0.0041, ...],
                "offset": [0, 0, 0, ...],
                "group_size": 128
            }
        ]
    }
}
```

QNN reads these group scales and compiles them into the context binary. At runtime on HTP, the group-wise dequantization is fused into the GEMM kernel — no separate dequant pass.

**When to use group-wise:** INT4 weight-only quantization for LLMs (W4A16 or W4A8). Per-channel INT8 rarely needs group-wise refinement, but INT4 almost always does.

---

### Cross-Layer Equalization (CLE)

CLE is AIMET's "free lunch" — it improves quantization accuracy without any training data.

**The problem it solves:** In many models, consecutive layers have very different weight ranges. Layer A might have weights in [-0.01, 0.01] while layer B has weights in [-5.0, 5.0]. When both are quantized to INT8, layer A wastes most of its quantization range (only using a tiny fraction of the [-127, 127] integer range), while layer B uses the full range.

**The insight:** For consecutive linear layers (or conv → conv sequences) with ReLU activations in between, you can mathematically rescale the weights of one layer up and the weights of the next layer down, without changing the model's output. This is because ReLU is a positive-homogeneous function: $\text{ReLU}(\alpha x) = \alpha \cdot \text{ReLU}(x)$ for $\alpha > 0$.

**The math:** Given two consecutive layers with weight matrices $W_1$ and $W_2$, and a diagonal scaling matrix $S$:

$$Y = W_2 \cdot \text{ReLU}(W_1 \cdot x) = (W_2 S^{-1}) \cdot \text{ReLU}((S W_1) \cdot x)$$

The scaling $S$ is chosen to equalize the ranges of $W_1$ and $W_2$ across channels:

$$s_i = \sqrt{\frac{\text{range}(W_1^{(i)})}{\text{range}(W_2^{(i)})}}$$

After CLE, both layers have similar per-channel weight ranges, which means both get similar quantization resolution.

**AIMET code:**
```python
from aimet_torch.cross_layer_equalization import equalize_model
equalize_model(model, input_shape=(1, 3, 224, 224))
```

That is the entire API. One function call. It modifies the model weights in-place.

**When CLE helps:** Models with batch normalization folded into convolutions (common in deployment), models with large weight range variation across layers.

**When CLE does NOT help:** Transformer models (attention layers are not simple conv→ReLU→conv sequences), models that don't use ReLU, models where weight ranges are already balanced.

---

### Adaptive Rounding (AdaRound) in AIMET

Standard quantization rounds each weight to the nearest integer:

$$q_i = \lfloor w_i / S \rceil$$

This seems optimal per-element, but it is not optimal for the layer's output. Rounding weight $w_3$ up might increase the error for some inputs, while rounding it down might decrease the error — and the optimal direction depends on the correlations between weights and the typical input distribution.

**AdaRound** learns a binary rounding decision for each weight — round up or round down — by minimizing the layer-wise reconstruction error:

$$\min_{\mathbf{v}} \| W\mathbf{x} - \tilde{W}(\mathbf{v})\mathbf{x} \|_F^2 + \lambda \sum_i h(v_i)$$

where:
- $\mathbf{v} \in [0, 1]^n$ is a continuous relaxation of the rounding variable
- $\tilde{W}(\mathbf{v})$ is the quantized weight with learned rounding
- $h(v_i)$ is a regularizer that pushes $v_i$ toward 0 or 1 (forcing a hard rounding decision)
- $\mathbf{x}$ comes from the calibration dataset

This is solved with gradient descent — typically 10,000 iterations per layer, which takes seconds to minutes per layer on a GPU.

**AIMET code:**
```python
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters

params = AdaroundParameters(
    data_loader=calibration_loader,
    num_batches=100,
    default_num_iterations=10000
)

model = Adaround.apply_adaround(
    model=model,
    dummy_input=torch.randn(1, 3, 224, 224).cuda(),
    params=params,
    path="./adaround_output",
    filename_prefix="resnet50"
)
```

**Impact:** AdaRound typically recovers 0.5–2% accuracy over naive rounding for INT8 PTQ. For models that are borderline (e.g., 1.5% accuracy drop with naive PTQ), AdaRound can bring it within the 1% acceptable threshold.

**Cost:** Requires a calibration dataset (typically the same one used for PTQ). Takes 10–60 minutes for a full model on a single GPU. No labels needed — it minimizes reconstruction error, not task loss.

---

### AIMET Quantization-Aware Training (QAT)

When PTQ (with CLE + AdaRound) is not enough — typically for INT8 quantization of transformer models, or for INT4 quantization of any model — AIMET provides QAT.

AIMET QAT inserts **fake quantization nodes** (also called quantization simulation nodes or "quant-dequant" wrappers) around each layer:

```
Original:         x → Linear → y
With QAT wrapper:  x → FakeQuant → Linear(FakeQuant(W)) → FakeQuant → y
```

During forward pass:
1. Input `x` is fake-quantized: quantized to INT8 and immediately dequantized back to float. This simulates the quantization noise that will occur during real deployment.
2. Weights `W` are fake-quantized similarly.
3. The output is computed in floating-point (because we are still training on GPU).
4. The output is fake-quantized.

During backward pass:
- Gradients flow through the fake-quantization nodes using the **Straight-Through Estimator (STE)** — the gradient of the rounding function (which has zero gradient almost everywhere) is approximated as 1.

**AIMET QAT code:**
```python
from aimet_torch.quantsim import QuantizationSimModel

# Create quantization simulation model
sim = QuantizationSimModel(
    model=model,
    quant_scheme='tf_enhanced',    # See note on quant_scheme below
    default_output_bw=8,           # 8-bit activations
    default_param_bw=8,            # 8-bit weights
    dummy_input=torch.randn(1, 3, 224, 224).cuda()
)

# Calibrate (compute initial quantization parameters)
sim.compute_encodings(
    forward_pass_callback=calibrate_fn,
    forward_pass_callback_args=calibration_loader
)

# Fine-tune (QAT)
optimizer = torch.optim.SGD(sim.model.parameters(), lr=1e-4)
for epoch in range(5):
    for batch in train_loader:
        loss = criterion(sim.model(batch['input']), batch['target'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Export quantized model
sim.export(path="./quantized_model", filename_prefix="resnet50_int8",
           dummy_input=torch.randn(1, 3, 224, 224).cuda())
```

**Key detail:** `sim.export()` produces two files:
1. An ONNX (or torchscript) model with the architecture
2. A JSON **encodings file** containing the scale and zero-point for every quantized tensor

The encodings file is critical — it is what QNN reads to know how to quantize each tensor.

**A note on `quant_scheme='tf_enhanced'`:**

The name `tf_enhanced` is confusing — it does NOT mean "TensorFlow only." In AIMET's terminology:
- `'tf'` = TensorFlow-style quantization: uses scale and offset (zero-point) encoding. This is the **scale/offset** scheme where $q = \text{clamp}(\text{round}(x/S) + Z, 0, 255)$. HTP is designed around this scheme for activations.
- `'tf_enhanced'` = Same scale/offset scheme, but with **enhanced calibration** — AIMET uses a more sophisticated algorithm to find the optimal clipping range (minimizing MSE rather than just using min/max). This is almost always the right choice.
- `'percentile'` = Uses a percentile of the observed range instead of the full min/max.

For LLMs and transformer models specifically, the asymmetric (scale/offset) nature of `tf_enhanced` is critical. LLM activations are almost never centered at zero — post-GELU, post-SiLU, and post-Softmax activations are all asymmetric. Using symmetric quantization for these activations wastes half the INT8 range. The `tf_enhanced` scheme automatically handles this with a non-zero zero-point.

**How many epochs?** Typically 5–15 epochs with a learning rate 10–100× smaller than original training. You are not retraining from scratch — you are fine-tuning the model to tolerate quantization noise.

---

### AIMET Practical Walkthrough

Here is a complete pipeline quantizing a ResNet-50 for Snapdragon deployment:

```python
import torch
import torchvision
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.quantsim import QuantizationSimModel

# 1. Load pretrained model
model = torchvision.models.resnet50(pretrained=True).cuda().eval()

# 2. Apply Cross-Layer Equalization (free improvement, no data needed)
equalize_model(model, input_shape=(1, 3, 224, 224))

# 3. Apply AdaRound (needs calibration data, no labels)
adaround_params = AdaroundParameters(
    data_loader=calibration_loader,  # ~1000 images
    num_batches=50,
    default_num_iterations=10000
)
model = Adaround.apply_adaround(
    model=model,
    dummy_input=torch.randn(1, 3, 224, 224).cuda(),
    params=adaround_params,
    path="./output",
    filename_prefix="resnet50"
)

# 4. Create quantization simulation
sim = QuantizationSimModel(
    model=model,
    quant_scheme='tf_enhanced',
    default_output_bw=8,
    default_param_bw=8,
    dummy_input=torch.randn(1, 3, 224, 224).cuda()
)

# 5. Calibrate
sim.compute_encodings(
    forward_pass_callback=run_calibration,
    forward_pass_callback_args=calibration_loader
)

# 6. Evaluate (check accuracy before QAT)
accuracy = evaluate(sim.model, val_loader)
print(f"PTQ accuracy (CLE + AdaRound): {accuracy:.2f}%")

# 7. (Optional) QAT if accuracy is not good enough
if accuracy < target_accuracy:
    optimizer = torch.optim.SGD(sim.model.parameters(), lr=1e-5)
    for epoch in range(10):
        train_one_epoch(sim.model, train_loader, optimizer)
    accuracy = evaluate(sim.model, val_loader)
    print(f"QAT accuracy: {accuracy:.2f}%")

# 8. Export for QNN
sim.export(path="./export", filename_prefix="resnet50_quantized",
           dummy_input=torch.randn(1, 3, 224, 224).cuda())
# Output: resnet50_quantized.onnx + resnet50_quantized.encodings
```

After step 8, you have an ONNX file and an encodings file. These are the inputs to QNN.

---

## QNN Native Quantization vs. AIMET — When to Use Which

Before diving into QNN, it is important to address a common question: **"Why do I need AIMET at all? QNN can quantize my model directly."**

This is true. The `qnn-onnx-converter` has built-in quantization. You can skip AIMET entirely:

```bash
# QNN Native Quantization (Path B) — no AIMET, no encodings file
qnn-onnx-converter \
    --input_network model.onnx \
    --input_list calibration_images.txt \    # QNN calibrates internally
    --output_path ./qnn_model
```

QNN uses the `--input_list` to run calibration internally and compute scales via basic MinMax or MSE methods. This works. But it is a fundamentally different capability level from AIMET.

**Comparison:**

| Capability | QNN Native Quantization | AIMET |
|-----------|------------------------|-------|
| **Calibration method** | MinMax, MSE | MinMax, MSE, Entropy, Percentile, learned (AdaRound) |
| **Cross-Layer Equalization** | ❌ Not available | ✅ One-line API |
| **AdaRound** | ❌ Not available | ✅ Learns optimal rounding per weight |
| **Quantization-Aware Training** | ❌ Not available — if PTQ fails, you are stuck | ✅ Fine-tune with fake-quant nodes |
| **Per-layer diagnostics** | ❌ Black box — "accuracy dropped 8%" with no explanation | ✅ Inspect each layer's quantization error |
| **Mixed-precision selection** | ❌ All layers same precision (or manual override) | ✅ Automated sensitivity analysis |
| **Group-wise quantization** | Limited | ✅ Full control with config files |
| **Speed of workflow** | ✅ Fast — one command | ❌ Slower — Python scripting, calibration runs |
| **When to use** | Quick prototyping, simple CNNs, models that quantize easily | Production deployment, transformers, accuracy-sensitive models |

**The decision tree:**

```
Is your model a standard CNN (ResNet, MobileNet, EfficientNet)?
  └─ YES → Try QNN Native first. If accuracy drop < 1%, ship it.
           If accuracy drop > 1%, switch to AIMET.
  └─ NO (transformer, custom architecture, LLM) → Go directly to AIMET.
           QNN Native will almost certainly fail for these models.
```

**The "Black Box" Problem in Detail:**

When QNN native quantization drops your TinyLlama's accuracy by 10%, you see:
```
QNN Converter: Model converted successfully.
QNN Accuracy Check: FP32 perplexity: 7.2, INT8 perplexity: 14.8
```

That is all the information you get. You do not know:
- Which of the 22 transformer layers is causing the degradation
- Whether the problem is weight outliers, activation outliers, or attention score quantization
- Whether CLE would have helped (spoiler: it often does, even for transformers with SiLU activations, on the MLP projection layers)
- What the optimal per-layer bit-width allocation is

With AIMET, you can diagnose:
```python
# Per-layer sensitivity analysis
for name, module in sim.model.named_modules():
    if hasattr(module, 'output_quantizer'):
        # Disable quantization for this layer, measure accuracy
        module.output_quantizer.enabled = False
        acc = evaluate(sim.model, val_loader)
        module.output_quantizer.enabled = True
        print(f"{name}: accuracy without quant = {acc:.2f}%")
```

This tells you exactly which layer, when quantized, causes the most damage. You then target that layer with INT16, AdaRound, or QAT — surgical fixes rather than blind retries.

**Bottom line:** QNN Native is the convenience path. AIMET is the engineering path. For anything beyond a demo, use AIMET.

---

### Why Experts Abandon QNN Native Quantization for Transformers

This sidebar exists because the comparison table above might still leave you thinking: *"I'll try QNN Native first, and if it fails, switch to AIMET."* For transformers, that is a waste of time. Here is why QNN Native is structurally incapable of quantizing transformers correctly — and this connects directly to the outlier theory from Chapter 14.

**The fundamental problem:** QNN Native uses MinMax or MSE calibration. Both methods compute a single scale based on the observed range of values. For a transformer linear layer with activation outliers (Chapter 14), the calibration sees something like this:

```
Channel activations observed during calibration:
  Channel 0:   values in [-1.2, 1.8]     ← normal
  Channel 1:   values in [-0.9, 2.1]     ← normal
  Channel 2:   values in [-0.5, 1.4]     ← normal
  ...
  Channel 127: values in [-3.0, 62.0]    ← OUTLIER (Chapter 14 pattern)
```

QNN Native's MinMax calibrator computes:
$$S = \frac{62.0 - (-3.0)}{255} = \frac{65.0}{255} \approx 0.255$$

Step size: 0.255. For the 99% of channels with values in $[-3, 3]$, the number of usable quantization levels is:
$$\frac{6.0}{0.255} \approx 23 \text{ levels}$$

Out of 256 available INT8 levels, 23 are used. **91% of the quantization budget is wasted** on the range $[3, 62]$ that only one outlier channel occupies. This is exactly the *Resolution Collapse* pattern from Chapter 14.

**What AIMET does differently:**

1. **CLE** — even though CLE was designed for conv→ReLU→conv patterns, AIMET applies a generalized version to consecutive linear layers in transformers. The MLP blocks (`gate_proj → SiLU → up_proj → down_proj`) benefit from CLE rebalancing the weight ranges across projections.

2. **AdaRound** — QNN Native rounds every weight to the nearest integer. AdaRound *learns* the optimal rounding direction per weight by minimizing layer output error on calibration data. For the outlier-heavy layers, this recovers 0.5–1.5% accuracy — often the difference between "usable" and "garbage."

3. **Per-layer sensitivity analysis** — AIMET can disable quantization for each layer individually and measure the accuracy impact. This reveals that (for a typical TinyLlama) layers 8, 14, and 19 cause 70% of the accuracy degradation. You promote those three layers to INT16. QNN Native has no mechanism for this discovery.

4. **QAT** — when PTQ still falls short, AIMET enables QAT. The model learns to tolerate quantization noise during fine-tuning. QNN Native has no training path. If PTQ fails, your only option is to go back to the beginning.

**The empirical reality for a TinyLlama-1.1B on Snapdragon 8 Gen 3:**

| Method | Perplexity (FP32 baseline: 7.2) | Recovery Effort |
|--------|--------------------------------|-----------------|
| QNN Native MinMax | 14.8 (+7.6) | None possible |
| QNN Native MSE | 12.1 (+4.9) | None possible |
| AIMET PTQ (default) | 9.4 (+2.2) | 30 min calibration |
| AIMET PTQ + CLE | 8.8 (+1.6) | +2 min (one API call) |
| AIMET PTQ + CLE + AdaRound | 8.1 (+0.9) | +30 min (per-layer optimization) |
| AIMET QAT (5 epochs) | 7.5 (+0.3) | +2 hours training |

The QNN Native path is a dead end at +4.9 perplexity. The AIMET path gets within 0.3 of FP32. There is no comparison.

Snapdragon is the **ultimate test** of the transformer quantization theories discussed in Chapter 14. Every failure pattern — outlier explosion, resolution collapse, attention score distortion — manifests here. And every mitigation technique — CLE, SmoothQuant, AdaRound, QAT — is deployed here through AIMET. The theory is not academic. It is the difference between a model that runs at 10 tokens/sec with good quality and one that produces incoherent text.

---

## QNN — Qualcomm Neural Network SDK

QNN is Qualcomm's model compilation and inference SDK. It takes a quantized model (typically ONNX + encodings from AIMET) and compiles it into a binary that runs on Hexagon HTP.

If AIMET is "make the model quantization-friendly," QNN is "make the model hardware-ready."

---

### What QNN Does

QNN performs several transformations:

1. **Model conversion** — reads ONNX (or other formats) and converts it to QNN's internal graph representation.
2. **Graph optimization** — fuses operators (conv + bias + ReLU → single fused op), removes redundant operations, rewrites patterns into hardware-efficient equivalents.
3. **Operator mapping** — maps each operator to a Hexagon HTP implementation. If an operator has no HTP implementation, it is flagged for CPU fallback.
4. **Quantization encoding** — reads the AIMET encodings file (or performs its own quantization if you did not use AIMET) and bakes the scale/zero-point information into the graph.
5. **Compilation** — produces a **context binary** (.bin file) that contains the compiled graph, weights, and all metadata needed for on-device execution. 

---

### The QNN Graph: From ONNX to Hexagon

The compilation pipeline looks like this:

```
resnet50.onnx + resnet50.encodings
        │
        ▼
   qnn-onnx-converter          ← Converts ONNX to QNN C++ model
        │
        ▼
   QNN model.cpp + model.bin    ← Intermediate representation
        │
        ▼
   qnn-context-binary-generator ← Compiles for target hardware
        │
        ▼
   resnet50_htp.bin             ← Ready to run on Hexagon
```

**Command-line example:**
```bash
# Step 1: Convert ONNX to QNN format
qnn-onnx-converter \
    --input_network resnet50_quantized.onnx \
    --quantization_overrides resnet50_quantized.encodings \
    --output_path ./qnn_model

# Step 2: Generate context binary for HTP
qnn-context-binary-generator \
    --model ./qnn_model/resnet50_quantized.bin \
    --backend libQnnHtp.so \
    --output_dir ./compiled \
    --binary_file resnet50_htp.bin
```

The `--backend` flag is critical. It determines what hardware the model is compiled for:
- `libQnnHtp.so` — Hexagon Tensor Processor (NPU). This is what you want for production.
- `libQnnCpu.so` — CPU reference. Useful for debugging.
- `libQnnGpu.so` — GPU (Adreno). Sometimes used when HTP is not available or for FP16 models.

---

### QNN Operator Support and Limitations

Not every ONNX operator maps to a Hexagon HTP implementation. This is the single biggest source of frustration in the Qualcomm pipeline.

**Well-supported operators (run on HTP at full speed):**
- Conv2d, DepthwiseConv2d, TransposeConv2d
- MatMul, FullyConnected (Linear)
- ReLU, ReLU6, Sigmoid, Tanh, GELU (approximate), Swish
- BatchNorm (folded into Conv at compile time)
- Add, Multiply, Subtract (element-wise)
- MaxPool, AvgPool, GlobalAvgPool
- Reshape, Transpose, Concat, Split
- Softmax
- LayerNorm (recent Hexagon versions)

**⚠ The HTP vs. HVX Split — A Hidden Bottleneck:**

The Hexagon processor contains two distinct compute units: the **HTP** (tensor processor, for matrix math) and the **HVX** (vector extensions, for element-wise and reduction operations). These are not the same hardware.

Critically, **Softmax** and **LayerNorm** often run on **HVX, not HTP**. They involve reductions (sum, max) and non-linear functions (exp, division) that the HTP's integer MAC pipeline cannot execute. Even though QNN does not flag these operators as "CPU fallback," they are running on a different (and slower) part of the Hexagon processor.

This creates a data transfer bottleneck within the Hexagon itself: HTP computes a matrix multiply → result is transferred to HVX for Softmax/LayerNorm → result is transferred back to HTP for the next matrix multiply. For transformer models with alternating Linear-LayerNorm-Linear-Softmax patterns, this HTP↔HVX ping-pong becomes a significant fraction of total inference time.

In profiling output (`qnn-net-run --profiling_level detailed`), look for operators marked as running on `HVX` rather than `HTP`. If Softmax and LayerNorm together take >30% of inference time, this bottleneck is dominating.

**Mitigation strategies:**
- Use **RMSNorm** instead of LayerNorm where possible — RMSNorm skips the mean subtraction, which is one fewer reduction. Some recent Hexagon versions have optimized RMSNorm kernels.
- Use **approximate Softmax** or fused attention kernels that keep the Softmax inside the HTP pipeline.
- On newest SoCs (8 Gen 3, X Elite), Qualcomm has added fused multi-head attention ops that execute entirely on HTP, avoiding the split.

**Partially supported (may fall back depending on parameters):**
- GroupConv (supported for some group counts, not all)
- Resize/Upsample (bilinear supported, bicubic may not be)
- Pad (constant pad supported, reflect pad may not be)
- Gather, ScatterND (depends on axis and index patterns)

**Commonly unsupported (will fall back to CPU):**
- Custom operators
- Complex dynamic control flow (if/else based on tensor values)
- Operators with dynamic output shapes
- Some exotic activations

**How to check:** QNN provides a model validation tool:
```bash
qnn-net-run \
    --model resnet50_quantized.bin \
    --backend libQnnHtp.so \
    --debug
```

The debug output shows which operators were placed on HTP and which fell back. Any fallback is a red flag for performance.

---

### QNN Quantization Encoding Files

If you used AIMET, you already have an encodings file. But understanding the format is important for debugging.

**The encodings JSON is the contract.** It is the single bridge between your PyTorch training world and the QNN compiler. Every quantization parameter — every scale, every zero-point, every bit-width decision — is captured in this file. If this file is wrong, your model will produce garbage on device, even if it was perfect in simulation.

The encodings file is a JSON file that specifies, for every quantized tensor in the model, its quantization parameters:

```json
{
    "activation_encodings": {
        "input_0": [
            {
                "bitwidth": 8,
                "dtype": "int",
                "is_symmetric": false,
                "max": 2.6417,
                "min": -2.1178,
                "offset": -128,
                "scale": 0.018664
            }
        ],
        "conv1.output": [
            {
                "bitwidth": 8,
                "dtype": "int",
                "is_symmetric": false,
                "max": 6.0,
                "min": 0.0,
                "offset": 0,
                "scale": 0.023529
            }
        ]
    },
    "param_encodings": {
        "conv1.weight": [
            {
                "bitwidth": 8,
                "dtype": "int",
                "is_symmetric": true,
                "max": 0.7823,
                "min": -0.7823,
                "offset": 0,
                "scale": 0.006161
            }
        ]
    }
}
```

**Reading this file — what each field means in practice:**

| Field | What It Controls | Debugging Signal |
|-------|-----------------|------------------|
| `bitwidth` | INT8 or INT16 | If a layer is INT16, check if it was forced there by accuracy issues |
| `is_symmetric` | Whether zero-point is 0 | Weights should be `true`, activations typically `false` |
| `scale` | Step size ($S$) | Very large scale = coarse quantization = likely accuracy issue |
| `offset` | Zero-point ($Z$) | Should be 0 for symmetric. Non-zero for asymmetric activations |
| `min/max` | Clipping range | Suspiciously wide range = one outlier stretching the grid |

**Expert debugging technique:** Compare the `min/max` of two adjacent layers. If layer N has `max: 3.2` and layer N+1 has `max: 87.5`, there is likely an outlier explosion between them. That is where CLE or SmoothQuant should be applied.

**Weight encodings should look like this (clean):**
```json
"weight": {
    "bitwidth": 8,
    "is_symmetric": true,
    "scale": 0.0034,
    "offset": 0
}
```
`offset: 0` confirms symmetric. `scale: 0.0034` means the weight range is approximately $\pm 0.0034 \times 127 \approx \pm 0.43$. This is typical for a well-trained layer.

**Activation encodings should look like this (after ReLU):**
```json
"relu_output": {
    "bitwidth": 8,
    "is_symmetric": false,
    "scale": 0.023529,
    "min": 0.0,
    "max": 6.0,
    "offset": 0
}
```
`min: 0.0` because ReLU never produces negatives. `max: 6.0` suggests ReLU6. `offset: 0` because the min is exactly zero — no need for a zero-point shift.

**Red flags in the encodings:**
- `scale > 0.5` for any layer — quantization is extremely coarse.
- `is_symmetric: true` for an activation after ReLU — wastes half the INT8 range.
- `is_symmetric: false` for weights — will trigger slower MAC path on HTP.
- `min` and `max` have the same sign but `offset` is 0 — possible miscalibration.

---

### QNN Context Binary and Caching

The compiled context binary (`.bin` file) is the final artifact that runs on device. It contains:

1. The optimized computation graph
2. All weights (quantized to INT8)
3. Memory allocation plans (which tensors go in VTCM, which go in DDR)
4. The execution schedule (which operations run in what order)

**Important:** Context binaries are hardware-specific. A binary compiled for Snapdragon 8 Gen 2 will NOT run on Snapdragon 8 Gen 1. You must compile for each target hardware separately.

Context binaries can be cached on device. The first inference call loads and initializes the binary (which may take 100ms–2s for large models). Subsequent inference calls reuse the initialized context and are fast.

---

### QNN Practical Walkthrough

Complete compilation from AIMET output to device-ready binary:

```bash
# 1. Convert ONNX model with quantization encodings
qnn-onnx-converter \
    --input_network ./export/resnet50_quantized.onnx \
    --quantization_overrides ./export/resnet50_quantized.encodings \
    --output_path ./qnn_output/resnet50

# 2. Compile for HTP (Snapdragon 8 Gen 2)
qnn-context-binary-generator \
    --model ./qnn_output/resnet50.bin \
    --backend libQnnHtp.so \
    --output_dir ./device_ready \
    --binary_file resnet50_htp.bin

# 3. Test locally (on host, using CPU backend for verification)
qnn-net-run \
    --model ./qnn_output/resnet50.bin \
    --backend libQnnCpu.so \
    --input_list input_list.txt \
    --output_dir ./cpu_outputs

# 4. Test on device (push to Android device via adb)
adb push ./device_ready/resnet50_htp.bin /data/local/tmp/
adb push libQnnHtp.so /data/local/tmp/
adb push qnn-net-run /data/local/tmp/
adb shell "cd /data/local/tmp && ./qnn-net-run \
    --model resnet50_htp.bin \
    --backend libQnnHtp.so \
    --input_list input_list.txt \
    --output_dir ./outputs"

# 5. Pull outputs and compare
adb pull /data/local/tmp/outputs ./device_outputs
python compare_outputs.py ./cpu_outputs ./device_outputs --tolerance 0.01
```

The `input_list.txt` file lists the raw input tensor files (one per line), typically stored as flat binary float32 or uint8 arrays.

---

### Handling the Unsupported: Custom Op Packages

When your model contains an operator that QNN does not recognize — a custom RoPE (Rotary Position Embedding) implementation, a non-standard activation, a specialized normalization — the converter does not compile it. It either errors out or silently routes it to CPU fallback. Neither outcome is acceptable for production.

**Custom Op Packages** are QNN's mechanism for extending the operator set. You write a Hexagon-optimized kernel for your custom op, package it, and tell the converter to use it. The op then runs on HTP like any built-in operator.

**The workflow has four steps:**

```
1. Define the Op      →  CustomOp.xml (interface specification)
       │
       ▼
2. Generate Skeleton  →  qnn-op-package-generator (creates C++ template)
       │
       ▼
3. Implement Kernel   →  Write HVX/HTP implementation in C++
       │
       ▼
4. Register at Convert →  Pass --op_package_lib to qnn-onnx-converter
```

---

**Step 1: Define the Op (XML Specification)**

Create an XML file that describes the op's interface — inputs, outputs, parameters, data types:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<OpDef>
    <OpDefCollection packageName="CustomRoPEPackage">
        <OpDefinition name="CustomRoPE"
                      description="Rotary Position Embedding for transformer models">
            <Input  name="input"    datatype="QNN_DATATYPE_UFIXED_POINT_8"
                    rank="3" layout="BATCH_TIME_FEATURE"/>
            <Input  name="cos_cache" datatype="QNN_DATATYPE_FLOAT_32"
                    rank="2" layout="TIME_FEATURE"/>
            <Input  name="sin_cache" datatype="QNN_DATATYPE_FLOAT_32"
                    rank="2" layout="TIME_FEATURE"/>
            <Output name="output"   datatype="QNN_DATATYPE_UFIXED_POINT_8"
                    rank="3" layout="BATCH_TIME_FEATURE"/>
            <Parameter name="interleaved" datatype="QNN_DATATYPE_BOOL_8"
                       mandatory="false" default="0"/>
        </OpDefinition>
    </OpDefCollection>
</OpDef>
```

Note the data types: `QNN_DATATYPE_UFIXED_POINT_8` means unsigned fixed-point INT8 (quantized). Your custom op must handle quantized data natively — it receives INT8 inputs with associated scales and must produce INT8 outputs.

---

**Step 2: Generate the Skeleton**

```bash
qnn-op-package-generator \
    --config_path CustomRoPE.xml \
    --output_path ./custom_rope_package \
    --backend htp                         # Generate HTP-specific template
```

This produces a directory with C++ source files:
```
custom_rope_package/
├── src/
│   ├── CustomRoPE.cpp          ← Implement your kernel HERE
│   └── CustomRoPEPackageInterface.cpp
├── include/
│   └── CustomRoPE.h
├── CMakeLists.txt
└── config/
    └── CustomRoPE.xml
```

---

**Step 3: Implement the Kernel**

This is the hard part. You must write Hexagon-optimized C++ that operates on quantized integers:

```cpp
// CustomRoPE.cpp (simplified)
#include "CustomRoPE.h"
#include "HTP/core/constraints.h"
#include "HTP/core/op_register.h"

// Register the op with HTP runtime
BEGIN_PKG_OP_DEFINITION(CustomRoPEOp, "CustomRoPE");

// The actual computation — runs on HTP
template<typename T_In, typename T_Out>
GraphStatus customRopeImpl(
    Tensor& out,
    const Tensor& input,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const Tensor& interleaved_param)
{
    // Get quantization parameters
    float in_scale  = input.get_interface_scale();
    int   in_offset = input.get_interface_offset();
    float out_scale = out.get_interface_scale();
    int   out_offset = out.get_interface_offset();

    // Get dimensions
    auto [batch, seq_len, dim] = input.dims();

    // Access raw INT8 data
    const uint8_t* in_data  = (const uint8_t*)input.raw_data();
    uint8_t*       out_data = (uint8_t*)out.raw_data();

    // RoPE computation in integer domain
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < seq_len; t++) {
            for (int d = 0; d < dim / 2; d++) {
                // Dequantize to float for trig ops
                float x0 = (in_data[idx(b,t,2*d)]     - in_offset) * in_scale;
                float x1 = (in_data[idx(b,t,2*d + 1)] - in_offset) * in_scale;

                float cos_val = cos_cache[t * dim/2 + d];
                float sin_val = sin_cache[t * dim/2 + d];

                // Apply rotation
                float y0 = x0 * cos_val - x1 * sin_val;
                float y1 = x0 * sin_val + x1 * cos_val;

                // Requantize to INT8
                out_data[idx(b,t,2*d)]     = clamp_uint8(y0 / out_scale + out_offset);
                out_data[idx(b,t,2*d + 1)] = clamp_uint8(y1 / out_scale + out_offset);
            }
        }
    }
    return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(CustomRoPEOp);
```

**Important:** This simplified version dequantizes to float internally. For production, an expert would use HVX vector intrinsics to perform the trig operations in fixed-point, avoiding float entirely. The performance difference can be 5–10×.

**HVX vs. HTP — Choosing the Right Compute Unit for Your Kernel:**

When writing a custom op kernel, you must decide which Hexagon compute unit it targets. This is not an abstract choice — it determines performance by 10× or more:

| Compute Unit | Best For | Instruction Style | Data Types | VTCM Access |
|-------------|---------|-------------------|------------|-------------|
| **HTP** | Matrix multiplies, convolutions | Tensor/tile-based | INT8, INT16 with INT32 accum | Direct, managed by compiler |
| **HVX** | Element-wise ops, reductions, non-linear functions | SIMD vector (128B/256B) | INT8, INT16, INT32, FP16, FP32 | Manual via `vmem` instructions |

- If your custom op is a **matrix operation** (custom attention variant, specialized GEMM), target HTP. Use the `QnnHtpOpPackage` APIs that feed into the tensor processor pipeline.
- If your custom op is **element-wise or a reduction** (custom activation, custom normalization, RoPE), target HVX. Use HVX vector intrinsics (`Q6_V*` functions).

Most custom ops for transformers (RoPE, custom activations, specialized normalization) are element-wise and should target HVX.

**Keeping Your Custom Op in VTCM (the Expert Move):**

The worst thing a custom op can do is read its input from DDR, compute, and write its output back to DDR — only for the next op to read it from DDR again into VTCM. This "VTCM eviction" pattern adds 2–5μs per layer — small, but it compounds across 22 transformer layers and both forward + KV cache operations.

The `QnnHtp_OptimizationUtility` API provides functions to hint that your custom op should keep its data in VTCM:

```cpp
#include "HTP/core/op_register.h"
#include "HTP/core/optimization.h"

// In your op registration:
BEGIN_PKG_OP_DEFINITION(CustomRoPEOp, "CustomRoPE");

// Declare VTCM residency requirements
DEF_OPTIMIZATION(CustomRoPEOp,
    Op("CustomRoPE")
    .set_vtcm_required(true)             // Request VTCM for this op
    .set_vtcm_size_bytes(256 * 1024)     // Request 256KB of VTCM
    .set_allow_tcm_spill(false)          // Do NOT allow DDR spill
);
```

When `set_vtcm_required(true)` is set, the QNN compiler ensures this op's input and output tensors reside in VTCM. If there is not enough VTCM, the compiler will tile the surrounding operations to make room — rather than spilling your op's data to DDR.

When `set_allow_tcm_spill(false)` is set, the compiler will error if it cannot fit this op in VTCM, rather than silently falling back to DDR. This is the "fail loudly" approach — better to know at compile time than to discover a 10× latency penalty at runtime.

Build the package:
```bash
cd custom_rope_package
mkdir build && cd build
cmake .. -DQNN_SDK_ROOT=/path/to/qnn-sdk -DHEXAGON_TOOLS=/path/to/hexagon-tools
make -j8
# Output: libCustomRoPEPackage.so
```

---

**Step 4: Register at Conversion Time**

When converting your ONNX model, tell `qnn-onnx-converter` about your custom op:

```bash
qnn-onnx-converter \
    --input_network model_with_rope.onnx \
    --quantization_overrides model.encodings \
    --op_package_lib libCustomRoPEPackage.so \
    --op_package_config CustomRoPE.xml \
    --output_path ./qnn_model
```

The `--op_package_lib` flag tells the converter: *"When you encounter an ONNX node named 'CustomRoPE', do not fail. Use this library."*

The `--op_package_config` flag provides the XML interface definition so the converter can validate input/output shapes and types.

---

**⚠ The Silent Fallback Trap for Custom Ops:**

If the custom op library is **not registered correctly** — wrong path, mismatched XML, version incompatibility — the converter may not error out. Instead, it silently marks the custom op for **CPU fallback**. The model compiles. It runs. But:

1. The custom op executes on CPU instead of HTP.
2. Data must transfer HTP → CPU → HTP around every call to this op.
3. If the op is inside a transformer block that repeats 22 times (like TinyLlama), that is 22 CPU round-trips.
4. Latency increases by 44–110ms (22 × 2–5ms per round-trip) — likely more than the entire HTP inference time.

**How to catch this:** Always run with profiling after adding custom ops:
```bash
qnn-net-run \
    --model model_htp.bin \
    --backend libQnnHtp.so \
    --profiling_level detailed \
    --output_dir ./profile

# Grep for CPU execution in profile output
grep -i "cpu\|fallback\|not_on_htp" ./profile/qnn_profiling.log
```

If your custom op appears as a CPU node in the profiling log, the registration failed silently. Recheck the XML, rebuild the library, and re-convert.

---

## Qualcomm AI Hub

Qualcomm AI Hub is a cloud platform that simplifies the AIMET → QNN → device pipeline. Instead of manually running each tool, AI Hub provides an API and web interface that handles compilation, quantization, profiling, and even on-device testing.

---

### What AI Hub Provides

1. **Model compilation as a service** — upload an ONNX or PyTorch model, specify the target device, and AI Hub returns a compiled context binary.
2. **On-device profiling** — AI Hub has a farm of real Snapdragon devices. You can profile your model's latency, memory usage, and operator-level timing without owning the hardware.
3. **Pre-optimized model zoo** — hundreds of popular models (ResNet, MobileNet, YOLO, Whisper, Stable Diffusion, LLaMA variants) already quantized and compiled for various Snapdragon targets.
4. **Accuracy validation** — run inference on-device and compare outputs against a reference.

---

### Pre-Optimized Model Zoo

The model zoo is the fastest path from "I need a model" to "it runs on device." Each model in the zoo has been:
- Quantized (typically INT8 with AIMET)
- Compiled with QNN for multiple Snapdragon targets
- Profiled on real hardware with published latency numbers

Example models and their performance:

| Model | Task | INT8 Latency (8 Gen 2) | Accuracy |
|-------|------|------------------------|----------|
| MobileNetV2 | Classification | 0.8ms | 71.2% top-1 |
| YOLOv8n | Object Detection | 3.2ms | 37.1 mAP |
| Whisper-tiny | Speech-to-Text | 18ms/chunk | ~good |
| Stable Diffusion 1.5 | Image Gen | ~5s/image | N/A |
| LLaMA 2 7B (INT4) | Text Gen | ~25 tok/s | N/A |

If your model is in the zoo (or close to one that is), start here.

---

### Profile-Optimize-Deploy Loop

AI Hub enables a rapid iteration loop:

```python
import qai_hub

# 1. Submit a model for compilation
compile_job = qai_hub.submit_compile_job(
    model="./resnet50.onnx",
    device=qai_hub.Device("Samsung Galaxy S23"),  # Snapdragon 8 Gen 2
    options="--quantize_full_type int8"
)

# 2. Profile on real hardware
profile_job = qai_hub.submit_profile_job(
    model=compile_job.get_target_model(),
    device=qai_hub.Device("Samsung Galaxy S23")
)

# 3. Get results
profile = profile_job.get_results()
print(f"Inference time: {profile.inference_time_ms:.1f} ms")
print(f"Peak memory: {profile.peak_memory_mb:.1f} MB")
print(f"Operators on NPU: {profile.npu_operator_count}")
print(f"Operators on CPU: {profile.cpu_operator_count}")  # Want this to be 0

# 4. If CPU fallback exists, investigate
if profile.cpu_operator_count > 0:
    for op in profile.cpu_operators:
        print(f"  CPU fallback: {op.name} ({op.type}) — {op.reason}")
```

The key metric: **CPU operator count should be zero.** Any CPU fallback means a graph split, which means a data copy, which means latency.

---

### AI Hub Practical Walkthrough

From a PyTorch model to on-device in 10 lines:

```python
import qai_hub
import torch
import torchvision

# Load and trace model
model = torchvision.models.mobilenet_v2(pretrained=True).eval()
example_input = torch.randn(1, 3, 224, 224)
traced = torch.jit.trace(model, example_input)

# Compile for device
compile_job = qai_hub.submit_compile_job(
    model=traced,
    device=qai_hub.Device("Samsung Galaxy S24"),
    input_specs={"x": (1, 3, 224, 224)},
)
compiled_model = compile_job.get_target_model()

# Profile
profile_job = qai_hub.submit_profile_job(
    model=compiled_model,
    device=qai_hub.Device("Samsung Galaxy S24")
)
print(profile_job.get_results())

# Inference on device
inference_job = qai_hub.submit_inference_job(
    model=compiled_model,
    device=qai_hub.Device("Samsung Galaxy S24"),
    inputs={"x": [example_input.numpy()]}
)
output = inference_job.get_output_data()
```

AI Hub handles the AIMET quantization, QNN compilation, and device execution behind the scenes. For prototyping and model selection, this is dramatically faster than the manual pipeline.

For production deployment, you typically download the compiled binary and integrate it into your app using the QAIRT runtime.

---

## QAIRT — Qualcomm AI Runtime

QAIRT is the on-device runtime that actually executes your compiled model. It is the last layer of the stack — the code that runs in your Android app, your embedded Linux system, or your Windows-on-Arm application.

---

### What QAIRT Is

QAIRT is a C/C++ library (with Java/Python bindings) that:
- Loads a compiled context binary
- Allocates input/output buffers
- Dispatches inference to the selected backend (CPU, GPU, or HTP)
- Manages memory transfers between backends
- Handles multi-model scheduling (when multiple models share the NPU)

It is the successor to SNPE (Snapdragon Neural Processing Engine), which was Qualcomm's previous runtime. SNPE is legacy — new projects should use QAIRT/QNN.

**Key clarification for legacy users:** If you have existing SNPE workflows, the migration path is:
- `snpe-dlc-convert` → `qnn-onnx-converter` (model conversion)
- `snpe-net-run` → `qnn-net-run` (on-device profiling and inference)
- `.dlc` files → `.bin` context binaries (model format)
- SNPE runtime API → QNN/QAIRT runtime API (inference in your app)

The core concepts (quantized inference on HTP, encoding files, backend selection) are the same. The API surface and model format changed.

For profiling your transformer model (e.g., TinyLlama) on-device, use `qnn-net-run` with the QAIRT HTP backend:
```bash
qnn-net-run \
    --model tinyllama_htp.bin \
    --backend libQnnHtp.so \
    --input_list inputs.txt \
    --perf_profile burst \
    --profiling_level detailed \
    --output_dir ./profile
```

The detailed profiling output shows per-operator timing, including which operators ran on HTP vs. HVX vs. CPU — critical for finding the bottleneck.

---

### QAIRT vs QNN vs SNPE (Legacy)

This naming is confusing, so let's be precise:

| Tool | Role | Status |
|------|------|--------|
| **SNPE** | Old all-in-one SDK (converter + runtime) | Legacy, deprecated for new models |
| **QNN** | New SDK for model conversion + compilation | Current, replaces SNPE's converter |
| **QAIRT** | New unified runtime | Current, replaces SNPE's runtime |
| **AIMET** | Quantization toolkit (training-side) | Current, independent of SNPE/QNN |
| **AI Hub** | Cloud platform for compilation + profiling | Current, wraps QNN + QAIRT |

In practice, "QNN" is often used loosely to refer to the entire new stack (converter + runtime). Technically, QNN is the SDK and QAIRT is the runtime component. The QNN SDK includes QAIRT.

---

### The QAIRT Execution Pipeline

When you call inference in QAIRT, here is what happens:

```
1. Load context binary
   └─ Deserialize graph structure, weights, execution schedule
   └─ Allocate weight buffers (VTCM or DDR)
   └─ Create execution context

2. Set up I/O buffers
   └─ Allocate input tensor buffers (with quantization parameters)
   └─ Allocate output tensor buffers

3. Fill input buffer
   └─ Your app writes input data (e.g., preprocessed image as uint8)
   └─ If input is float32, QAIRT quantizes it using the input encoding

4. Execute inference
   └─ QAIRT dispatches the graph to the backend
   └─ On HTP: operations execute on Hexagon, intermediates stay in VTCM where possible
   └─ On graph splits: data copies between HTP ↔ CPU (expensive!)

5. Read output buffer
   └─ Output is quantized INT8 (on HTP backend)
   └─ QAIRT dequantizes to float32 using the output encoding
   └─ Your app reads the float32 output
```

**Performance note:** Steps 1–2 happen once (at model load time). Steps 3–5 happen on every inference call. The goal is to make step 4 as fast as possible — which means keeping everything on HTP.

---

### Backend Selection: CPU, GPU, HTP

QAIRT supports three backends, and choosing the right one matters enormously:

**HTP (Hexagon Tensor Processor):**
- Fastest for quantized (INT8/INT16) models
- Requires the model to be quantized
- Requires all operators to be HTP-compatible
- Typical speedup: 5–20× over CPU for quantized models

**GPU (Adreno):**
- Good for FP16 models
- Does not require quantization
- Useful when HTP is not available or the model cannot be quantized
- Slower than HTP for quantized models, faster than CPU for FP16

**CPU:**
- Always available as fallback
- Supports all operators
- Slowest option
- Used for debugging and for operators that are not supported on HTP/GPU

**Code example (C++):**
```cpp
#include "QnnInterface.h"

// Initialize QNN
Qnn_BackendHandle_t backend;
QnnBackend_Config_t* config = nullptr;

// Choose HTP backend
QnnInterface_t qnnInterface;
loadBackend("libQnnHtp.so", &qnnInterface);

// Create context from compiled binary
Qnn_ContextHandle_t context;
qnnInterface.contextCreateFromBinary(
    backend,
    nullptr,  // device handle
    config,
    binaryBuffer,     // your .bin file loaded into memory
    binaryBufferSize,
    &context,
    nullptr
);

// Set up input tensor
Qnn_Tensor_t inputTensor;
// ... configure tensor dimensions, data type, etc.
// Copy input data into tensor buffer

// Execute
qnnInterface.graphExecute(graph, &inputTensor, 1, &outputTensor, 1, nullptr, nullptr);

// Read output
float* output = dequantize(outputTensor.data, outputTensor.scale, outputTensor.offset);
```

---

### QAIRT Practical Walkthrough

**Android deployment (Java/Kotlin):**

```kotlin
// In your Android app
val qnnManager = QnnManager(context)

// Load compiled model
qnnManager.loadModel("resnet50_htp.bin", QnnBackend.HTP)

// Prepare input (e.g., camera frame → preprocessed tensor)
val inputBuffer = preprocessImage(cameraFrame)  // Returns ByteBuffer (uint8)

// Run inference
val outputBuffer = qnnManager.execute(inputBuffer)

// Post-process output
val predictions = softmax(outputBuffer)
val topClass = predictions.argmax()
```

**Python (for prototyping):**
```python
import qnn_wrapper  # Qualcomm's Python bindings

# Load model
model = qnn_wrapper.Model("resnet50_htp.bin", backend="htp")

# Run inference
input_data = preprocess(image)  # numpy array, uint8
output = model.execute(input_data)

# Post-process
predictions = softmax(output)
```

---

## End-to-End Pipeline: Training to On-Device

This section ties everything together. Here is the complete journey of a model from training to running on a Snapdragon device.

---

### Step 1: Train in PyTorch or TensorFlow

Train your model as usual. No quantization considerations needed at this stage — though designing a quantization-friendly architecture (e.g., avoiding exotic activations, preferring ReLU/ReLU6) helps later.

```python
model = MyModel()
train(model, train_dataset, epochs=100)
torch.save(model.state_dict(), "model_fp32.pth")
```

---

### Step 2: Quantize with AIMET

Apply the AIMET pipeline: CLE → AdaRound → (optionally) QAT.

```python
# CLE (no data needed)
equalize_model(model, input_shape=(1, 3, 224, 224))

# AdaRound (calibration data needed)
model = Adaround.apply_adaround(model, dummy_input, adaround_params)

# Create QuantSim and calibrate
sim = QuantizationSimModel(model, quant_scheme='tf_enhanced',
                            default_output_bw=8, default_param_bw=8,
                            dummy_input=dummy_input)
sim.compute_encodings(calibrate_fn, calibration_loader)

# (Optional) QAT
fine_tune(sim.model, train_loader, epochs=10, lr=1e-5)

# Export
sim.export("./export", "model_quantized", dummy_input)
```

Output: `model_quantized.onnx` + `model_quantized.encodings`

---

### Step 3: Export to ONNX

If you used AIMET's `sim.export()`, this is already done. If not:

```python
torch.onnx.export(model, dummy_input, "model.onnx",
                  opset_version=13,
                  input_names=["input"],
                  output_names=["output"],
                  dynamic_axes=None)  # Static shapes for HTP!
```

**Critical:** HTP requires static shapes. Do not use `dynamic_axes` unless you know the QNN compiler can handle it.

---

### Step 4: Compile with QNN

```bash
# Convert ONNX to QNN
qnn-onnx-converter \
    --input_network model_quantized.onnx \
    --quantization_overrides model_quantized.encodings \
    --output_path ./qnn_model

# Compile for HTP
qnn-context-binary-generator \
    --model ./qnn_model/model_quantized.bin \
    --backend libQnnHtp.so \
    --output_dir ./compiled \
    --binary_file model_htp.bin
```

---

### Step 5: Deploy with QAIRT

Push the compiled binary to the device and run inference using QAIRT APIs (C++, Java, or Python bindings as shown above).

---

### Step 6: Profile and Iterate

```bash
# Profile on device
qnn-net-run \
    --model model_htp.bin \
    --backend libQnnHtp.so \
    --input_list inputs.txt \
    --perf_profile burst \
    --profiling_level detailed \
    --output_dir ./profile_output
```

Check the profiling output for:
- **Total inference time** — is it fast enough?
- **Operator breakdown** — which ops take the most time?
- **CPU fallbacks** — any operators running on CPU?
- **Memory usage** — does the model fit in VTCM?

If latency is too high: look at CPU fallbacks first, then consider mixed-precision (INT16 for accuracy-critical layers, INT8 for the rest), then consider architectural changes.

---

## How HTP Actually Computes: The Integer-Only Truth

When you look at the QAIRT API, it appears to accept float32 inputs and return float32 outputs. This is a convenience wrapper. Internally, the HTP is an **integer-only engine.** It never touches floating-point numbers.

Here is what actually happens inside the HTP for a single linear layer:

**Floating-point equation (what you think happens):**
$$Y = W \cdot X + b$$

**Integer-only equation (what HTP actually computes):**
$$Y_q = W_q \cdot X_q + b_q$$

where $W_q$, $X_q$ are the quantized INT8 tensors, and $b_q$ is a precomputed INT32 bias that absorbs the zero-point corrections and the original floating-point bias.

The full expansion:
$$Y_q = \text{requantize}\left(\sum_i W_{q,i} \cdot X_{q,i} + b_q, \quad S_{\text{out}}, Z_{\text{out}}\right)$$

where:
- The MAC $\sum_i W_{q,i} \cdot X_{q,i}$ is computed in INT8×INT8 → INT32
- $b_q$ is precomputed at compile time: $b_q = \text{round}(b_{\text{float}} / (S_w \cdot S_x)) - Z_x \sum_i W_{q,i}$
- The result is **requantized** from INT32 to INT8 using the output scale $S_{\text{out}}$ and zero-point $Z_{\text{out}}$

The combined scale factor $S_{\text{combined}} = (S_w \cdot S_x) / S_{\text{out}}$ is the only "floating-point-like" operation — and even this is implemented as a fixed-point multiply-and-shift on HTP. No floating-point unit is involved at any stage.

**Why this matters:** When accuracy diverges between QuantSim (on your GPU) and on-device inference, the root cause is almost always this requantization step. QuantSim simulates quantization but computes in float32. The HTP computes everything in integers with fixed-point scale application. Tiny rounding differences at each requantization accumulate across layers.

---

## Troubleshooting the NPU

This section covers the three most common "the model compiles but doesn't work" scenarios.

---

### Scenario 1: Model Compiles, but Accuracy Is Zero

**Symptom:** The model runs on device, returns outputs, but every prediction is wrong. Accuracy is at random-chance level or worse.

**Root cause (90% of cases): Input scale mismatch.**

Your model was trained with ImageNet normalization:
```python
# Training preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),          # [0, 255] → [0.0, 1.0]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )                                # → approx [-2.1, 2.6]
])
```

The AIMET calibration saw inputs in the range $[-2.1, 2.6]$ and set the input encoding accordingly:
```json
"input_0": { "min": -2.1178, "max": 2.6417, "scale": 0.018664, "offset": -128 }
```

But your on-device app feeds raw camera pixels:
```kotlin
val pixels: ByteArray = camera.getFrame()  // Values: 0–255
qnnManager.execute(pixels)  // WRONG! Expects normalized [-2.1, 2.6]
```

The HTP quantizes the input using `scale: 0.018664` and `offset: -128`. Raw pixel value 128 becomes:
$$q = \text{round}(128 / 0.018664) + (-128) \approx 6731$$

This overflows INT8 (max 127). The result is clamped, and every input value becomes the same clamped value. All predictions are identical.

**Fix options:**
1. **Fold preprocessing into the model** before quantization — add a `Normalize` layer as the first operation. AIMET calibrates through it, and the encodings automatically account for raw pixel input.
2. **Match the input format** — normalize on-device before feeding to the NPU.
3. **Recalibrate** with raw pixel inputs — change AIMET's calibration to use unnormalized inputs, so the input encoding expects [0, 255].

---

### Scenario 2: Model Runs, but Output Is Slightly Wrong

**Symptom:** Classification accuracy drops from 76% (FP32) to 68% (on-device INT8). Not random, but clearly degraded.

**Debugging checklist:**
1. **Compare QuantSim vs. on-device outputs** for the same input. If QuantSim shows 75.5% but on-device shows 68%, the issue is in the QNN compilation or runtime, not in the quantization itself.
2. **Check the encodings file** for outlier layers — look for `scale` values >0.1 (coarse quantization).
3. **Try INT16 for the first and last layer** — these are often the most sensitive:
   ```python
   # Force first/last layer to INT16
   sim.quantizer_config('model.conv1', bitwidth=16)
   sim.quantizer_config('model.fc', bitwidth=16)
   ```
4. **Apply CLE before re-quantizing** — if you skipped CLE, try it. It is free and often recovers 0.5–1%.
5. **Apply AdaRound** — another 0.5–1% recovery.
6. **QAT** — the last resort, but typically recovers the remaining gap.

---

### Scenario 3: Latency Is 10× Worse Than Expected

**Symptom:** You expected 5ms inference, but profiling shows 50ms.

**Debugging checklist:**
1. **Check for CPU fallbacks** — the #1 cause. Profile with `--profiling_level detailed` and look for operators on CPU.
2. **Check HTP vs. HVX split** — Softmax/LayerNorm on HVX creates data transfer overhead.
3. **Check VTCM spills** — if the compiler warning says "VTCM budget exceeded," activation tensors are spilling to DDR.
4. **Check batch size** — batch >1 on mobile is almost always wrong. Set batch=1.
5. **Check perf_profile** — if you are profiling in `default` or `power_saver` mode, the NPU is throttled. Use `burst` for benchmarking.

---

## Simulation vs. Reality: The QuantSim Gap

**Critical warning:** AIMET's `QuantizationSimModel` (QuantSim) is a **mathematical simulation** of quantization. It inserts fake-quantize nodes into a PyTorch graph and runs on your GPU in float32. It simulates quantization noise but does NOT simulate:

- **Latency** — QuantSim tells you nothing about how fast the model will run on HTP.
- **Operator support** — QuantSim will happily fake-quantize a `grid_sample` operator that QNN cannot compile for HTP.
- **CPU fallbacks** — QuantSim does not know which operators will be offloaded to CPU.
- **HTP vs. HVX splits** — QuantSim does not model the intra-Hexagon data transfer cost.
- **VTCM tiling** — QuantSim does not simulate memory constraints.
- **Fixed-point requantization** — QuantSim uses float32 arithmetic; HTP uses integer arithmetic with fixed-point scale multiplication. Tiny rounding differences accumulate.

**The only way to confirm deployment-ready behavior is to run on device.** Specifically:
1. Run `qnn-net-run` with `--backend libQnnHtp.so --profiling_level detailed` to confirm all ops are on HTP.
2. Compare numerical outputs between QuantSim and on-device inference. Acceptable: <0.5% relative difference per output element. Red flag: >2% difference.
3. Profile latency on the target device in `burst` mode.

QuantSim is a necessary step — it catches gross quantization errors (e.g., complete accuracy collapse) early. But it is not sufficient. The "Silent Fallback" pattern (model compiles, runs, produces reasonable-looking outputs, but 3 operators are on CPU and latency is 5× worse than expected) is only caught by on-device profiling.

---

## Transformer on Snapdragon: The Expert Checklist

For deploying LLMs (TinyLlama, Phi, Mistral, LLaMA variants) on Snapdragon, here is the distilled checklist:

**Quantization scheme:**
- [ ] Weights: **Symmetric, per-channel** (or per-group for INT4). `is_symmetric: true`, `offset: 0`.
- [ ] Activations: **Asymmetric, per-tensor**. Use `tf_enhanced` quant scheme in AIMET.
- [ ] KV cache: **INT8 symmetric** — cuts cache memory by 2× vs FP16, essential for long sequences.

**Preprocessing:**
- [ ] Run **CLE first** — even though it helps less for transformers than CNNs, it is free and can help projection layers.
- [ ] Run **SmoothQuant** (via AIMET or manual per-channel scaling) — this is the #1 technique for transformer activations with outliers.
- [ ] Apply **AdaRound** — especially on the QKV projection and output projection weights.

**Operator compatibility:**
- [ ] Replace exact GELU with **approximate GELU** (`GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))` — the version QNN supports).
- [ ] Verify **Softmax** is running on HTP (Gen 3+) or accept HVX execution (Gen 2 and below).
- [ ] Verify **LayerNorm / RMSNorm** HTP support on your target SoC.
- [ ] Remove any **custom ops** — they WILL fall back to CPU.

**Memory:**
- [ ] Target **batch size 1** for generation (autoregressive, memory-bound).
- [ ] Compile with **VTCM budget** matching your target SoC.
- [ ] For seq_len > 512: compile **multiple context binaries** for different sequence lengths, or use padded static shapes.
- [ ] Force **KV cache current-token slice** into VTCM if possible.

**Validation:**
- [ ] Profile on-device with `qnn-net-run --profiling_level detailed`.
- [ ] Confirm **zero CPU fallbacks**.
- [ ] Compare QuantSim vs. on-device output numerically.
- [ ] Measure **tokens/second** in autoregressive generation, not just single-forward-pass latency.

---

## VTCM Tiling and Windowing: The Performance Cliff

VTCM (Vector Tightly Coupled Memory) is the single most important hardware resource on the Hexagon NPU. Understanding how the QNN compiler uses it — and what happens when it runs out — is the difference between a model that runs at 10 tokens/sec and one that crawls at 1 token/sec.

---

### How Tiling Works

A matrix multiply $Y = X \times W$ where $X$ is $[M \times K]$ and $W$ is $[K \times N]$ requires storing three tensors simultaneously:
- Input tile of $X$
- Weight tile of $W$
- Output tile of $Y$

If $M = 512$ (sequence length), $K = 2048$ (hidden dim), $N = 2048$ (output dim), the full tensors are:
- $X$: $512 \times 2048 \times 1\text{B} = 1\text{MB}$
- $W$: $2048 \times 2048 \times 1\text{B} = 4\text{MB}$
- $Y$: $512 \times 2048 \times 4\text{B (INT32 accumulator)} = 4\text{MB}$
- **Total: 9MB** — far exceeds the 4MB VTCM on Snapdragon 8 Gen 3.

The QNN compiler must **tile** this operation. It breaks the matrices into chunks that fit in VTCM:

```
Full matrix multiply:   X [512 × 2048] × W [2048 × 2048] = Y [512 × 2048]

Tiled execution (example: 128×128 tiles):

  For tile_m in [0, 128, 256, 384]:          ← 4 row tiles of X
    For tile_n in [0, 128, 256, ...1920]:     ← 16 column tiles of W
      For tile_k in [0, 128, 256, ...1920]:   ← 16 inner dim tiles
        
        1. Load X_tile [128 × 128] from DDR → VTCM     (16 KB)
        2. Load W_tile [128 × 128] from DDR → VTCM     (16 KB)
        3. Compute Y_tile += X_tile × W_tile in VTCM    (INT8 MAC)
        4. (Accumulate in INT32 Y_tile in VTCM)         (64 KB)
      
      5. Requantize Y_tile from INT32 → INT8             (16 KB)
      6. Write Y_tile [128 × 128] from VTCM → DDR       (16 KB)

VTCM budget per iteration: 16 + 16 + 64 = 96 KB  ✓ fits easily
Total tiles: 4 × 16 × 16 = 1024 tile-multiply iterations
```

Each tile loads from DDR, computes in VTCM, and the result either stays in VTCM (if the next operation can consume it immediately) or writes back to DDR (if there is not enough VTCM for the next layer's tiles).

---

### The Performance Cliff: VTCM Hit vs. DDR Spill

The difference between VTCM-resident and DDR-spilled computation is stark:

| Metric | VTCM (on-chip) | DDR (off-chip) |
|--------|----------------|----------------|
| Bandwidth | ~200–400 GB/s | ~25–50 GB/s |
| Latency | ~1 cycle | ~100+ cycles |
| Energy per access | ~1 pJ | ~100 pJ |

When all tile data fits in VTCM, the MAC units are always fed — compute-bound, running at peak TOPS. When tiles spill to DDR, the MAC units stall waiting for data — memory-bound, running at a fraction of peak.

**Worked example: TinyLlama self-attention on Snapdragon 8 Gen 3 (4MB VTCM):**

The self-attention $\text{Attn} = \text{Softmax}(Q K^T / \sqrt{d}) \cdot V$ involves:
1. $Q K^T$: $[512 \times 64] \times [64 \times 512] = [512 \times 512]$ per head — 256KB output (INT32). **Fits in VTCM.** ✅
2. Softmax over $[512 \times 512]$: 256KB. **Fits.** ✅ (but runs on HVX, not HTP)
3. $\text{Attn} \times V$: $[512 \times 512] \times [512 \times 64] = [512 \times 64]$ — 32KB output. **Fits.** ✅

Total per-head working set: ~544KB. With 32 heads, if computed sequentially per head: **544KB** at a time. Fits comfortably in 4MB VTCM.

But if the compiler tries to compute all 32 heads in parallel: $32 \times 544\text{KB} = 17\text{MB}$. **Does not fit.** The compiler must serialize across heads and tile within each head.

**The tiling strategy the compiler chooses — sequential vs. parallel, tile size, which tensors stay in VTCM between operations — determines the DDR traffic.** The `--vtcm_mb` flag tells the compiler how much VTCM it can use. If you set it lower than the hardware provides (e.g., because another model is sharing the NPU), the compiler uses smaller tiles and generates more DDR traffic.

---

### When Tiling Fails: The DDR Bottleneck

The QNN compiler is good at tiling standard operations (MatMul, Conv2d). But tiling can fail or produce poor results in several scenarios:

1. **Custom ops without VTCM hints** — if your custom op does not declare its VTCM requirements (via `set_vtcm_required`), the compiler may not reserve VTCM for it. The op's inputs and outputs spill to DDR, and the surrounding tiled operations must re-load data.

2. **Large intermediate tensors that cannot be tiled** — some operations produce outputs that must be fully materialized before the next operation can consume them. Example: the full $[512 \times 512]$ attention matrix must exist before Softmax. If sequence length grows to 2048, this becomes $[2048 \times 2048] \times 4\text{B} = 16\text{MB}$ (INT32) — far exceeding VTCM. The compiler must tile Softmax itself, which is complex and often suboptimal.

3. **Non-contiguous memory access patterns** — transpose, permute, and gather operations rearrange data in memory. After a transpose, data that was contiguous (and VTCM-friendly) may now be strided across DDR. The next operation must load it in small, non-contiguous chunks — much slower than streaming contiguous blocks.

**How to diagnose tiling problems:**
```bash
# Compile with verbose logging
qnn-context-binary-generator \
    --model model.bin \
    --backend libQnnHtp.so \
    --vtcm_mb 4 \
    --log_level verbose \
    --output_dir ./compiled 2>&1 | tee compile.log

# Search for VTCM warnings
grep -i "vtcm\|spill\|tile\|ddr" compile.log
```

Look for messages like:
- `"VTCM budget exceeded for op X, falling back to DDR"` — direct DDR spill.
- `"Tiling op X with tile size [M, N] due to VTCM constraint"` — tiling happening (OK if tile sizes are reasonable).
- `"Unable to tile op X, requires full materialization"` — worst case, the compiler gave up tiling.

---

## Common Pitfalls on Qualcomm Hardware

**1. Dynamic shapes:**
HTP does not support truly dynamic shapes. If your model accepts variable-length sequences, you must either:
- Pad to a fixed maximum length and compile for that shape
- Compile multiple context binaries for different shapes (e.g., seq_len=64, 128, 256)

**2. Unsupported operators:**
A single unsupported op splits the graph and adds 2–5ms overhead. Common culprits: custom GELU implementations (use approximate GELU instead), grid_sample, deformable convolutions, complex indexing operations.

**3. Batch normalization not folded:**
If batch normalization layers are not folded into preceding convolutions before quantization, they appear as separate operators and waste cycles. AIMET and QNN both attempt BN folding, but custom architectures may not fold cleanly.

**4. Wrong quantization scheme (the bit-range waste problem):**
Using symmetric quantization for activations that are strictly positive (post-ReLU) wastes half the INT8 range. Use asymmetric for activations. Conversely, using asymmetric for weights when symmetric would suffice adds unnecessary complexity and slower MAC paths.

**Worked example — why asymmetric activations matter:**

A ReLU output produces values in $[0, 6.0]$ (ReLU6). With **symmetric INT8** quantization:
$$S = \frac{6.0}{127} = 0.0472, \quad \text{range mapped: } [-6.0, +6.0]$$
$$\text{Usable levels for } [0, 6.0]: 128 \text{ out of } 256 \text{ (50\% wasted)}$$

The 128 levels for $[-6.0, 0)$ are **never used** — ReLU never produces negative values. You are paying for 256 levels but only using 128.

With **asymmetric INT8** quantization:
$$S = \frac{6.0 - 0.0}{255} = 0.0235, \quad Z = 0$$
$$\text{Usable levels for } [0, 6.0]: 256 \text{ out of } 256 \text{ (0\% wasted)}$$

Step size halved: $0.0235$ vs $0.0472$. **Double the resolution.** For the same 8 bits.

This is why `tf_enhanced` (which uses asymmetric activations by default) is the right choice for HTP. And this is especially critical for LLM activations:
- Post-SiLU in LLaMA/TinyLlama: values typically in $[-0.5, 8.0]$ — heavily skewed positive.
- Post-Softmax in attention: values in $[0.0, 1.0]$ — strictly positive.
- Post-RMSNorm: values centered near zero but not symmetric — slight asymmetry matters at INT8 precision.

For each of these, asymmetric quantization preserves more information per bit.

**5. Large models exceeding VTCM (the tiling problem):**
VTCM is the secret sauce of HTP performance, but it is small — 4MB on Snapdragon 8 Gen 3, 8MB on X Elite. If an activation tensor for a single layer exceeds the available VTCM, the HTP cannot hold the full tensor on-chip. It must **tile** the operation: compute a slice, write it to DDR, load the next slice's inputs, compute, write, repeat.

This is a critical failure pattern for LLMs on Snapdragon. Consider TinyLlama with a hidden dimension of 2048 and a sequence length of 512:
- Activation tensor for one linear layer: $512 \times 2048 \times 1\ \text{byte (INT8)} = 1\text{MB}$
- That fits in 4MB VTCM — fine for a single layer.
- But KV cache for 22 layers at seq_len 512, 32 heads, head_dim 64: $2 \times 22 \times 512 \times 32 \times 64 \times 1\text{B} \approx 46\text{MB}$
- The KV cache does NOT fit in VTCM. It lives in DDR.

Every attention layer must read KV cache from DDR → compute attention → write back. The DDR bandwidth (typically 25–50 GB/s on mobile) becomes the bottleneck, not HTP compute. This is why **KV cache quantization** (INT8 instead of FP16) and **forced VTCM residency** for the current-token KV slice are critical for LLM inference on Snapdragon.

QNN provides hints for VTCM allocation:
```bash
qnn-context-binary-generator \
    --model model.bin \
    --backend libQnnHtp.so \
    --vtcm_mb 4 \                   # Total VTCM budget
    --enable_vtcm_limit_from_context  # Let compiler manage allocation
```

**6. INT8 overflow in accumulation:**
Matrix multiplication accumulates INT8 × INT8 products into an INT32 accumulator. For very large inner dimensions (e.g., 4096-wide matrix multiplies in transformers), the accumulated value can approach INT32 limits. This is rare but can cause silent numerical errors. QNN handles this automatically with intermediate requantization, but custom op implementations may not.

**7. Input preprocessing mismatch:**
Your model was trained with ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) applied to float32 [0, 1] inputs. On device, you are feeding uint8 [0, 255] pixels. If the input quantization encoding does not account for this normalization, every prediction will be wrong. Solution: fold the preprocessing into the model before quantization, or ensure the input encoding matches the raw pixel range.

---

## Performance Tuning: Getting the Last 10%

Once the basics are working (all ops on HTP, correct accuracy), here is how to squeeze out the last bits of performance:

**1. Use burst performance mode:**
Snapdragon devices have power/thermal governors that throttle the NPU. For benchmarking or latency-critical inference:
```bash
qnn-net-run --perf_profile burst ...  # Temporary high-performance mode
```
In production, use `sustained_high_performance` for consistent behavior.

**2. Pre-allocate buffers:**
Allocate input/output buffers once and reuse them across inference calls. Buffer allocation is surprisingly expensive on some devices.

**3. Use direct buffer access:**
Avoid copying data between CPU and NPU buffers. If your input comes from the camera, use hardware-backed buffers that both the camera HAL and HTP can access directly.

**4. Compile with optimization flags:**
```bash
qnn-context-binary-generator \
    --model model.bin \
    --backend libQnnHtp.so \
    --vtcm_mb 4 \          # Tell the compiler how much VTCM to use
    --optimization_level 3  # Maximum optimization
```

**5. Profile at the operator level:**
Identify the slowest operators and consider:
- Can they be replaced with faster equivalents? (e.g., depthwise separable instead of standard conv)
- Can they be fused with adjacent operators?
- Would INT16 for just these operators improve accuracy enough to reduce the number of QAT epochs needed?

**6. Multi-model scheduling:**
If multiple models share the NPU (e.g., detection + classification + pose), use QAIRT's scheduling APIs to avoid contention:
```cpp
// Set priority
QnnHtpDevice_CustomConfig_t htpConfig;
htpConfig.priority = QNN_HTP_PRIORITY_HIGH;  // or NORMAL, LOW
```

---

## Summary

The Qualcomm quantization stack is a pipeline with two quantization paths:

1. **AIMET** (the precision scalpel) quantizes your model with CLE → AdaRound → QAT and exports ONNX + encodings. This is the **production path** — non-negotiable for transformers and LLMs.
2. **QNN Native** (the black box) can quantize via `--input_list`, but has no recovery tools when accuracy drops. Acceptable only for simple CNNs.
3. **QNN** compiles the quantized model into a hardware-specific context binary, with support for Custom Op Packages when you need unsupported operators on HTP.
4. **QAIRT** executes the compiled binary on the Hexagon NPU. This is the current runtime — **not SNPE**, which is legacy. New projects in 2026 should use `libQnnHtp.so` via QAIRT exclusively.
5. **AI Hub** provides a shortcut for steps 1–3 via a cloud API + pre-optimized model zoo.

The critical success factors:
- **All operators on HTP** — zero CPU fallbacks. Use `--profiling_level detailed` to verify.
- **Per-channel symmetric weights + per-tensor asymmetric activations** — the scheme HTP's MAC hardware is designed for. Symmetric weights eliminate the runtime $Z_w$ term. Asymmetric activations avoid wasting half the INT8 range on non-existent negative values.
- **Static shapes & VTCM alignment** — HTP requires static shapes. The compiler tiles operations to fit VTCM (4–8MB). If tiles spill to DDR, throughput drops 5–10×.
- **Custom ops registered correctly** — silent CPU fallback for unregistered ops is a 2–5ms penalty per call × N layers. Always profile to confirm.

This chapter is the proving ground for every transformer quantization theory in this book. The activation outlier problem (Chapter 14), the SmoothQuant fix (Chapter 15), the group-wise weight quantization (Chapter 16), the GPTQ/AWQ calibration (Chapter 17), the KV cache bottleneck (Chapter 18) — all of them manifest on Snapdragon hardware. The Qualcomm stack is not a separate topic. It is the deployment reality where theoretical quantization concepts either work or fail.

When the pipeline is working correctly, INT8 models on Hexagon HTP achieve inference times that are 5–20× faster than FP32 on CPU, with accuracy within 1% of the original model. For edge deployment at scale — billions of Snapdragon devices — this stack is the path from research to production.

