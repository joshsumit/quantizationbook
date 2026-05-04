# Chapter 21: The NVIDIA Stack — From Training to Data-Center Inference

## Why a Dedicated Chapter on NVIDIA?

If Qualcomm owns the edge, NVIDIA owns the data center. Every major cloud provider — AWS, Azure, GCP, Oracle — runs NVIDIA GPUs for AI inference. When you deploy a model to a cloud endpoint, a T4, A10, A100, L4, or H100 GPU is almost certainly behind it.

NVIDIA's inference stack is not just "run PyTorch in production." Production inference uses **TensorRT**, a dedicated inference compiler that optimizes and quantizes models for NVIDIA Tensor Cores. For large language models, **TensorRT-LLM** extends this with transformer-specific optimizations. And **NVIDIA ModelOpt** (formerly TensorRT Model Optimizer) provides the training-side quantization and compression tools.

This chapter walks through every layer, from "I have a trained PyTorch model" to "it serves 10,000 requests per second on an H100."

---

## 10-Minute Quickstart: Build, Run, and Benchmark Your First TensorRT Engine

> **This quickstart is the only entry path.** If you cannot finish these four steps, stop and fix your environment before reading anything else. The rest of the chapter builds on the artifacts you create here.

### Checklist: Before You Start

Before touching any NVIDIA tool, verify these prerequisites:

- [ ] **GPU:** NVIDIA GPU with compute capability ≥ 7.0 (Volta or newer). Check: `nvidia-smi`
- [ ] **Driver:** NVIDIA driver installed and supporting the CUDA version your container needs. Check: `nvidia-smi` → "CUDA Version" in top-right
- [ ] **Docker + nvidia-container-toolkit:** GPU-enabled Docker working. Check: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`

> **⚠ Container Toolkit version mismatch — the #1 beginner failure.** The NVIDIA Container Toolkit version must be compatible with both your host driver and the CUDA version inside the container. Symptoms of a mismatch: `docker run --gpus all` hangs, produces cryptic CDI errors, or fails silently with no GPU visible inside the container. Fix: always install the *latest* Container Toolkit (`nvidia-ctk --version` to check), ensure your host driver meets the minimum version for the container's CUDA (e.g., CUDA 12.4 containers need driver ≥ 550), and restart Docker after any toolkit upgrade (`sudo systemctl restart docker`).
- [ ] **NGC login:** `docker login nvcr.io` completed with your NGC API key
- [ ] **Disk space:** ≥50 GB free (NGC containers are 10–20 GB each; engines can be several GB)
- [ ] **Network access:** Can reach `nvcr.io` and `github.com` (or have air-gapped alternatives — see fallback in Step 1)
- [ ] **HuggingFace token (LLM path only):** `export HF_TOKEN=...` set if using gated models like LLaMA

### Prerequisites Smoke Test

Before anything else, verify your environment. Every command below must succeed:

```bash
# 1. GPU visible to host?
nvidia-smi
# Expected: table showing your GPU (e.g., "Tesla T4", "A100-SXM4-80GB")
# If this fails: install or update the NVIDIA driver.

# 2. Docker can see the GPU?
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
# Expected: same GPU table, but from inside the container.
# If this fails: install nvidia-container-toolkit and restart Docker.

# 3. NGC registry login (required to pull NVIDIA containers)
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key from https://ngc.nvidia.com/setup/api-key>
# If you skip this, container pulls will fail with auth errors.
```

> **Common beginner blocker:** Many first-time users skip the NGC login step and get a cryptic "unauthorized" error when pulling containers. The username is literally the string `$oauthtoken` (not your email), and the password is the API key from your NGC account.

### Step 1: Pull the TensorRT Container and Get a Test Model

```bash
# Pull the TensorRT development container
docker pull nvcr.io/nvidia/tensorrt:24.05-py3

# Start an interactive session with GPU access
docker run --gpus all -it --rm -v $(pwd):/workspace nvcr.io/nvidia/tensorrt:24.05-py3

# Inside the container: download a well-known ONNX model (ResNet-50)
pip install onnx
python -c "
import urllib.request, os
url = 'https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx'
if not os.path.exists('resnet50.onnx'):
    urllib.request.urlretrieve(url, 'resnet50.onnx')
    print('Downloaded resnet50.onnx')
else:
    print('resnet50.onnx already exists')
"
```

> **If GitHub is blocked (air-gapped / restricted network):** The TensorRT container ships sample ONNX models. Use one of those instead:
> ```bash
> # Option A: use TensorRT sample models (already in the container)
> ls /usr/src/tensorrt/data/   # list available sample models
> cp /usr/src/tensorrt/data/resnet50/ResNet50.onnx ./resnet50.onnx
>
> # Option B: export from torchvision (no internet needed, torch is in the container)
> python -c "
> import torch, torchvision
> m = torchvision.models.resnet50(pretrained=False)   # random weights, fine for benchmarking
> torch.onnx.export(m, torch.randn(1,3,224,224), 'resnet50.onnx', opset_version=17,
>                   input_names=['input'], output_names=['output'])
> print('Exported resnet50.onnx from torchvision')
> "
>
> # Option C: copy from host volume (mount your host directory with -v)
> cp /workspace/my_models/resnet50.onnx ./resnet50.onnx
> ```

### Step 2: Build an FP16 Engine and Benchmark It

```bash
# Build FP16 engine (simplest possible TensorRT workflow)
trtexec --onnx=resnet50.onnx --saveEngine=resnet50_fp16.plan --fp16

# Expected output (last few lines):
# [I] GPU Compute Time: min = 0.3ms, max = 0.5ms, mean = 0.35ms
# [I] Throughput: 2857.14 qps
```

That's it — you now have a compiled TensorRT engine. The `.plan` file is a serialized binary optimized for your specific GPU.

> **What you just did (ONNX → Builder → Engine → Runtime in 5 lines):**
> 1. You exported/downloaded a model in ONNX format — a portable graph representation.
> 2. The TensorRT **builder** read the ONNX graph and enumerated thousands of kernel implementations ("tactics" — explained in detail in the "What Are Tactics?" section below).
> 3. The builder benchmarked each tactic on your specific GPU and selected the fastest combination.
> 4. The builder serialized the optimized graph + selected kernels into a `.plan` file — your **engine**.
> 5. When you run the engine (via `trtexec --loadEngine`), the TensorRT **runtime** loads the plan and executes inference with near-zero overhead.
>
> This is the entire TensorRT mental model: expensive build once, cheap inference forever.

### Step 3: Upgrade to INT8

> **⚠ RANDOM CALIBRATION WARNING ⚠**
> The command below uses `--int8` without a calibration dataset. TensorRT will calibrate with **random data**. This is fine for benchmarking throughput, but the **outputs will be numerically wrong**. Do NOT ship a randomly-calibrated engine to production. Real calibration with representative data is covered in the "Calibration" section below.

```bash
# Build INT8 engine (TensorRT uses its built-in calibration with random data for benchmarking)
trtexec --onnx=resnet50.onnx --saveEngine=resnet50_int8.plan --int8 --fp16

# Compare throughput between FP16 and INT8
trtexec --loadEngine=resnet50_fp16.plan --batch=32 --iterations=100
trtexec --loadEngine=resnet50_int8.plan --batch=32 --iterations=100
```

> **Why `--int8 --fp16` together? (Mixed-precision search space)**
>
> You are not telling TensorRT to use both types on every operation. You are defining the **search space** — the set of precisions the builder is *allowed* to consider per layer. Here is what happens inside:
>
> 1. **Priority:** TensorRT tries INT8 first for every layer, because INT8 Tensor Core tactics offer the highest throughput.
> 2. **Fallback:** If a layer has no INT8 kernel (e.g., LayerNorm on pre-Hopper GPUs), or the INT8 tactic is slower than FP16 for that layer's specific dimensions, the builder falls back to FP16 — not all the way to FP32.
> 3. **Why not `--int8` alone?** Without `--fp16`, any layer that can't run in INT8 must fall back to FP32. FP32 runs on standard CUDA cores, not Tensor Cores, creating a massive bottleneck. The INT8→FP32→INT8 transitions are far more expensive than INT8→FP16→INT8 because FP16 stays on Tensor Cores.
>
> The result is a **mixed-precision engine** — the compute-heavy layers (convolutions, matrix multiplies) run in INT8, while precision-sensitive layers (residual additions, the final classifier, normalization) may run in FP16. This is not a compromise; it is the intended design. Modern NVIDIA GPUs are built for exactly this mixed-precision pattern.
>
> **Rule:** Always pair `--int8` with `--fp16`. Using `--int8` alone is almost never what you want.

> **What you should see:** INT8 throughput is 1.3–1.8× higher than FP16 on Turing/Ampere GPUs. If the speedup is less than 1.2×, your model may be memory-bandwidth-bound rather than compute-bound (see "Troubleshooting" later in this chapter).

> **Note:** The `--int8` flag with `trtexec` and no calibration data uses random calibration, which is fine for benchmarking throughput but will produce wrong outputs. For correct INT8 inference, you need real calibration data — covered in the "Calibration" section below.

### Step 4: What to Do If It Fails

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `nvidia-smi` not found | NVIDIA driver not installed | Install driver from nvidia.com |
| `docker: Error response` + "could not select device driver" | `nvidia-container-toolkit` missing | Install it and restart Docker daemon |
| "unauthorized" on `docker pull nvcr.io/...` | NGC login missing or wrong credentials | Run `docker login nvcr.io` with API key |
| "ONNX parse failed" during `trtexec` | ONNX opset not supported by this TensorRT version | Re-export with a lower opset (e.g., `opset_version=17`) |
| Engine builds but throughput is the same as FP32 | GPU doesn't have INT8 Tensor Cores (Volta or older) | Use FP16 instead, or upgrade GPU |

### Expected Artifacts Checklist

After completing the quickstart, you should have these files:

```
/workspace/
├── resnet50.onnx           ← Original ONNX model (exported from PyTorch or downloaded)
├── resnet50_fp16.plan       ← Compiled TensorRT engine (FP16 precision)
└── resnet50_int8.plan       ← Compiled TensorRT engine (INT8 precision)
```

> **File naming note:** TensorRT serialized engines use either `.plan` or `.engine` as the file extension — they are the same format. This chapter uses `.plan` consistently.

### Beginner Exit Criteria

**You can proceed to the rest of this chapter only when all three are true:**
1. You can build an FP16 engine (`resnet50_fp16.plan` exists and `trtexec --loadEngine` runs without error)
2. You can build an INT8 engine and see higher throughput than FP16
3. You can inspect the engine's binding names (needed later for Triton config)

### Verification Commands (Copy-Paste Gate)

Run these three commands after the quickstart. All three must succeed before you continue:

> **These commands are benchmarking, not inference.** `trtexec --loadEngine` feeds **random synthetic data** into the engine and measures throughput/latency. It does not run your real images or produce meaningful predictions. You are verifying that the engine *loads, runs, and achieves expected performance* — not that the outputs are correct. Real inference with actual data is covered in "Step 4: Validate Accuracy" in the End-to-End Pipeline section below.

```bash
# 1. Engine loads and runs successfully? (synthetic data, no real input)
trtexec --loadEngine=resnet50_fp16.plan --iterations=10
# Expected: "[I] Throughput: ... qps" — this confirms the engine is valid and can execute
# This is NOT running inference on real images — it is timing the engine with random tensors

# 2. Binding names (you will need these for Triton config.pbtxt):
trtexec --loadEngine=resnet50_fp16.plan --verbose 2>&1 | grep -i "binding"
# Expected: lines showing input/output tensor names, e.g.:
# [V] Input  binding: "data", dimensions: [1,3,224,224], type: kFLOAT
# [V] Output binding: "resnetv17_dense0_fwd", dimensions: [1,1000], type: kFLOAT
# (Names vary by ONNX export — note them down, they must match your Triton config exactly)

# 3. Performance comparison (synthetic benchmark, not accuracy test):
trtexec --loadEngine=resnet50_fp16.plan --batch=1 --iterations=50 --warmUp=200
trtexec --loadEngine=resnet50_int8.plan --batch=1 --iterations=50 --warmUp=200
# Expected: INT8 latency should be lower (or throughput higher) than FP16
# This tells you quantization is working at the hardware level — accuracy validation comes later
```

### Toy Failure Exercise (Learn by Breaking)

Beginners learn fastest by seeing a failure and fixing it. This exercise deliberately creates a broken setup so you can recognize the error message and know how to fix it.

**Context:** So far, you have a `.plan` engine file — a compiled model that runs on TensorRT. But an engine file alone is not a server. To serve the engine over HTTP/gRPC (so clients can send requests), you need **Triton Inference Server**. Triton requires two things: (1) the engine file placed in a specific directory structure called a *model repository*, and (2) a `config.pbtxt` file that tells Triton the model's input/output names, data types, and dimensions. Think of `config.pbtxt` as the "wiring diagram" that connects incoming HTTP requests to the engine's actual tensor bindings.

In production, you would not hand-write `config.pbtxt` from scratch — Triton can auto-generate a minimal config from the engine file (covered in "Step 5: Deploy with Triton" below). But understanding what's inside `config.pbtxt` matters, because auto-generation doesn't set dynamic batching, doesn't know your preferred batch sizes, and gets it wrong when the engine has non-standard tensor names. The most common beginner mistake? Getting the tensor names wrong. Let's see what that looks like:

```bash
# 1. Set up a Triton model repository with a DELIBERATELY wrong config:
mkdir -p model_repository/resnet50_broken/1
cp resnet50_fp16.plan model_repository/resnet50_broken/1/model.plan

# This config.pbtxt tells Triton "the input tensor is called WRONG_NAME"
# — but the engine's actual input has a different name (whatever trtexec --verbose showed you).
cat > model_repository/resnet50_broken/config.pbtxt << 'EOF'
name: "resnet50_broken"
platform: "tensorrt_plan"
max_batch_size: 1
input [
  {
    name: "WRONG_NAME"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "ALSO_WRONG"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]
EOF

# 2. Start Triton (it will fail to load the model):
# tritonserver --model-repository=model_repository 2>&1 | grep -i "error"
# Expected error: "input 'WRONG_NAME' could not be found in model"

# 3. Fix it: replace WRONG_NAME / ALSO_WRONG with the real binding names
#    you found in the verification step above (trtexec --verbose | grep binding).
```

> **Lesson:** The tensor names in `config.pbtxt` must exactly match the engine's binding names. There is no renaming, no fuzzy matching. Get the names from `trtexec --loadEngine=... --verbose | grep binding` and copy them verbatim. You'll write a proper `config.pbtxt` when you reach "Step 5: Deploy with Triton" in the End-to-End Pipeline section — for now, you just need to know that this wiring exists and that wrong names produce the error above.

---

## Choose Your Path

The rest of this chapter covers the NVIDIA stack in depth. Depending on your workload, you'll use different tools:

**Track A — CNNs and Non-LLM Models (e.g., ResNet, YOLO, BERT-classification):**
You will use: PyTorch → ONNX export → TensorRT (or ModelOpt + TensorRT) → Triton.
Start reading from "What Is the NVIDIA Stack?" and follow the TensorRT sections.

**Track B — Large Language Models (e.g., LLaMA, Mistral, GPT):**
You will use: PyTorch/HuggingFace → ModelOpt quantization → TensorRT-LLM → Triton.
Start reading from "What Is the NVIDIA Stack?" but focus on the TensorRT-LLM and TRT-LLM Recipe Matrix sections.

**Track C — "I just want the fastest path to production":**
Follow these three options in order of complexity:
- **Option A (simplest):** `trtexec --onnx=model.onnx --fp16` — FP16, no quantization, works everywhere
- **Option B (INT8 with calibration):** `trtexec --onnx=model.onnx --int8 --fp16 --calib=calibration.cache` — needs calibration data
- **Option C (explicit Q/DQ via ModelOpt):** ModelOpt PTQ → ONNX with Q/DQ nodes → TensorRT build — recommended for transformers and production

### Minimum Viable Pipeline (One-Page View)

These two diagrams show the end-to-end path for each track. Everything in this chapter connects back to one of these flows:

**CNN / Non-LLM Path:**
```
 PyTorch model
      │
      ▼
 torch.onnx.export()         ← ONNX graph (portable)
      │
      ▼
 trtexec --onnx=... --int8   ← TensorRT build (GPU-specific engine)
      │
      ▼
 model_repository/model/1/model.plan
      │
      ▼
 tritonserver --model-repository=...   ← Triton serves HTTP/gRPC
```

**LLM Path:**
```
 HuggingFace checkpoint
      │
      ▼
 ModelOpt quantize (or quantize.py)   ← PTQ: SmoothQuant / AWQ / FP8
      │
      ▼
 trtllm-build --checkpoint_dir=...    ← TRT-LLM engine build
      │
      ▼
 model_repository/llm/1/              ← engine_dir in config.pbtxt
      │
      ▼
 tritonserver (tensorrtllm backend)   ← Triton serves with in-flight batching
```

> **Rule:** Every tool in this chapter fits into one of these two pipelines. If you are not sure where a section belongs, trace it back to this diagram.

---

## Glossary — 10 Terms You Must Know

> **Read this box before continuing.** These terms appear on nearly every page. If you forget one, come back here.

| # | Term | Definition |
|---|------|-----------|
| 1 | **Engine / Plan** | A serialized binary file (`.plan` or `.engine`) containing a model compiled and optimized for a specific GPU. Not portable across GPU architectures by default. |
| 2 | **Tactic** | A specific kernel implementation for an operation. TensorRT benchmarks multiple tactics per layer and selects the fastest one for the target GPU. This concept is fundamental — see "What Are Tactics?" below for the full explanation. |
| 3 | **Workspace** | Temporary GPU memory that TensorRT uses during inference for intermediate computations. Larger workspace allows TensorRT to try more (potentially faster) tactics. |
| 4 | **Optimization Profile** | A specification of min/opt/max shapes for dynamic input dimensions. TensorRT auto-tunes kernels for the `opt` shape and guarantees correctness for any shape in the min–max range. |
| 5 | **Plugin** | A custom C++/CUDA extension that implements an operation not natively supported by TensorRT. Required when your model uses ops that TensorRT can't parse from ONNX. |
| 6 | **Backend** | In Triton Inference Server, the runtime that executes a model. `tensorrt_plan` is the backend for TensorRT engines; `tensorrtllm` is the backend for TRT-LLM models. |
| 7 | **Q/DQ Nodes** | QuantizeLinear / DequantizeLinear operations in an ONNX graph. They specify exact quantization scales, making quantization explicit and reproducible rather than relying on TensorRT's internal heuristics. |
| 8 | **KV Cache** | Key-Value cache — stores past attention key/value pairs during LLM autoregressive generation. Grows with sequence length and can consume more memory than the model weights for long contexts. |
| 9 | **Tensor Cores** | Specialized hardware units inside each NVIDIA Streaming Multiprocessor (SM) that perform matrix multiply-accumulate on low-precision types (FP16, INT8, FP8). All quantized speedups come from Tensor Cores. |
| 10 | **Compute Capability** | SM version (e.g., SM 8.0) that identifies the GPU architecture and determines which precision instructions (INT8 IMMA, FP8 HMMA) are available. |

Additional terms used in expert sections:

| Term | Definition |
|------|-----------|
| **Builder** | The TensorRT component that compiles a model into an engine. Runs offline (build-time). |
| **Model Repository** | A directory structure that Triton uses to discover and load models. Each model has a versioned subdirectory containing the engine file and a `config.pbtxt` configuration. |
| **IMMA / HMMA** | Integer Matrix Multiply-Accumulate / Half-precision Matrix Multiply-Accumulate — the actual Tensor Core instruction sets for INT8 and FP16/FP8 respectively. |
| **NCCL** | NVIDIA Collective Communications Library — handles multi-GPU communication (tensor parallelism) over NVLink or PCIe. |

### Build-Time vs. Runtime vs. Serving — What Runs Where

A common source of confusion is which tools run when. Here's the clear separation:

```
BUILD-TIME (offline, on your workstation or CI):
┌──────────────────────────────────────────────┐
│  TensorRT Builder + calibration              │
│  + kernel auto-tuning                        │
│  → produces: .plan engine file               │
│  (Takes minutes to hours. Runs once.)        │
└──────────────────────────────────────────────┘

RUNTIME (online, on the inference server):
┌──────────────────────────────────────────────┐
│  TensorRT Runtime loads .plan                │
│  → executes inference on GPU                 │
│  (Takes milliseconds per request.)           │
└──────────────────────────────────────────────┘

SERVING (production infrastructure):
┌──────────────────────────────────────────────┐
│  Triton Inference Server                     │
│  → hosts the TensorRT Runtime               │
│  → handles HTTP/gRPC, batching, routing      │
│  (Runs continuously as a service.)           │
└──────────────────────────────────────────────┘
```

### What Are "Tactics"? (The Core Concept)

The glossary above defines a tactic as "a specific kernel implementation for an operation." That one-liner understates how central this concept is. If you understand tactics, every TensorRT behavior — slow builds, GPU-specific engines, mysterious performance regressions — makes sense. If you don't, TensorRT feels like a black box.

**The analogy:** Think of a high-level operation in your ONNX graph — say, a 3×3 Convolution — as a *recipe*. A tactic is a specific way the GPU *executes* that recipe. The same convolution can be computed by many different algorithms, each tuned for different hardware conditions.

**What varies between tactics:**

| Dimension | Examples | Why It Matters |
|-----------|----------|---------------|
| **Algorithm** | Direct convolution, Winograd (fewer multiplications), FFT (frequency-domain), implicit GEMM | Winograd is faster for 3×3 kernels but uses more memory; FFT wins for very large kernels |
| **Memory tiling** | How input/output data is broken into tiles to fit GPU L1/L2 cache or shared memory | Larger tiles = fewer memory round-trips but more register pressure |
| **Precision path** | FP32 CUDA cores, FP16 Tensor Cores, INT8 Tensor Cores (IMMA) | INT8 IMMA has 2× the throughput of FP16, but requires aligned dimensions |
| **Data layout** | NCHW (channel-first), NHWC (channel-last), NC/32HW32 (blocked) | Tensor Cores prefer NHWC; if two adjacent layers disagree on layout, a Reformat kernel is needed |

**Why there are thousands:**
TensorRT generates a *cross-product* of these dimensions. For a single convolutional layer:

```
(5 algorithms) × (10 tiling strategies) × (4 block sizes) × (3 data layouts) = 600 candidate tactics
```

Multiply by 50–200 layers in a typical model, and the builder is evaluating tens of thousands of kernel variants.

**The "Race" — why builds are slow:**
TensorRT cannot predict which tactic will be fastest by static analysis. GPU performance is too sensitive to memory bandwidth, cache hit rates, and instruction scheduling. So TensorRT does the only reliable thing: it *literally runs every candidate* on your GPU and times them.

1. **Enumerate** — query cuDNN/cuBLAS for every kernel that can compute this layer at the requested precision
2. **Benchmark** — launch each kernel, measure microseconds (this is why you see "Timing Runner" messages during builds)
3. **Select** — the fastest tactic wins and is hardcoded into the `.plan` file

This benchmarking is why an engine build can take minutes to hours — and why the result is specific to the GPU it was built on.

**The hardware lock-in consequence:**
The "fastest tactic" for an A100 (large L2 cache, 108 SMs) is almost certainly different from the fastest tactic for a T4 (small L2 cache, 40 SMs). This is why a `.plan` file built on one GPU architecture won't run on another — the serialized engine contains the *specific winning tactics* for the build-time GPU.

**The Workspace connection:**
The glossary defines Workspace as "temporary GPU memory for intermediate computations." Here's the tactical implication: some fast algorithms (like Winograd convolution) require large scratch buffers. If you limit `--workspace` to, say, 256 MB, TensorRT silently disqualifies every tactic that needs more than 256 MB of scratch space — potentially excluding the fastest kernels. The default workspace is usually sufficient, but if you see unexpectedly slow inference, check whether a workspace cap is filtering out the best tactics.

> **Bottom line:** TensorRT is not a compiler in the traditional sense. It is a *search engine for GPU kernels*. The build process is an empirical search over thousands of tactics; the engine file is the serialized result of that search. Every concept in this chapter — tactic timing caches, engine non-portability, build-time vs. runtime, even why INT8 sometimes isn't faster — traces back to this one idea.

### Where to Run What — Tool to Container Mapping

A beginner doesn't know which tool lives where. Here is the definitive map:

| Tool | Available In | Install if Missing |
|------|-------------|-------------------|
| `trtexec` | NGC `tensorrt` container (pre-installed) | Cannot pip-install; use the container |
| `polygraphy` | NGC `tensorrt` container (pre-installed) | `pip install polygraphy` (but needs TRT Python bindings) |
| `onnx-graphsurgeon` | NGC `tensorrt` container (pre-installed) | `pip install onnx-graphsurgeon` |
| `tensorrt` (Python) | NGC `tensorrt` container (pre-installed) | `pip install tensorrt` (needs matching CUDA) |
| `tensorrt_llm` | NGC `tritonserver:*-trtllm-*` container | `pip install tensorrt_llm` (complex deps — prefer container) |
| `tritonclient` | **Not** in server container — install on your client machine | `pip install tritonclient[http]` or `tritonclient[grpc]` |
| `nsys` (Nsight Systems) | NGC `tensorrt` and `tritonserver` containers | Install from NVIDIA developer site |
| `ncu` (Nsight Compute) | NGC `tensorrt` container | Install from NVIDIA developer site |
| `modelopt` | Separate install | `pip install nvidia-modelopt` (needs PyTorch + CUDA) |

> **Rule:** When in doubt, use the NGC container. It has everything except `tritonclient` (which goes on your client, not the server).

---

## What Is "The NVIDIA Stack"?

```
┌─────────────────────────────────────────────────────────┐
│                    YOUR TRAINED MODEL                    │
│              (PyTorch / TensorFlow / ONNX)               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   NVIDIA ModelOpt                        │
│        (Training-side quantization & compression)        │
│   • PTQ  • QAT  • Sparsity  • Distillation              │
└────────────────────────┬────────────────────────────────┘
                         │  Quantized model (ONNX or checkpoint)
                         ▼
┌─────────────────────────────────────────────────────────┐
│                     TensorRT                             │
│            (Inference Compiler & Runtime)                 │
│   • Layer fusion  • Kernel auto-tuning                    │
│   • INT8/FP8 calibration  • Engine building               │
└────────────────────────┬────────────────────────────────┘
                         │  Serialized engine (.plan)
                         ▼
┌─────────────────────────────────────────────────────────┐
│               TensorRT-LLM (for LLMs)                    │
│        (Transformer-specific inference runtime)           │
│   • KV cache management  • In-flight batching             │
│   • Weight-only / SmoothQuant / FP8                       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│               Triton Inference Server                    │
│             (Model serving infrastructure)                │
│   • Dynamic batching  • Multi-model  • Scaling            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                 NVIDIA GPU Hardware                       │
│   • Tensor Cores (INT8, FP8, FP16, TF32)                 │
│   • cuBLAS / cuDNN (kernel libraries)                     │
│   • HBM (High Bandwidth Memory)                          │
└─────────────────────────────────────────────────────────┘
```

Each layer:
- **ModelOpt** — makes your model quantization-friendly (training-side, like AIMET for Qualcomm).
- **TensorRT** — compiles and optimizes the model for GPU inference (like QNN for Qualcomm).
- **TensorRT-LLM** — extends TensorRT specifically for transformer-based LLMs.
- **Triton** — serves the compiled model in production (load balancing, batching, multi-model).
- **cuDNN / cuBLAS** — the low-level kernel libraries that execute the actual matrix multiplications on Tensor Cores.

---

### The Base Layer: CUDA, Drivers, and Containers

Before any of the above tools work, you need a compatible base environment. This is the #1 source of "works on my machine, breaks in production" failures on NVIDIA:

**The compatibility chain:**
```
NVIDIA Driver → CUDA Toolkit → cuDNN / cuBLAS → TensorRT → TensorRT-LLM → Triton
```

Each component has specific version requirements. A driver that supports CUDA 12.4 may not work with a TensorRT version built against CUDA 12.6. The canonical reference is the **TensorRT Support Matrix**, which lists every supported combination of driver, CUDA, cuDNN, and OS.

**The production rule: use NGC containers.**

NVIDIA GPU Cloud (NGC) containers are pre-built Docker images with a known-good combination of all components:

```bash
# TensorRT development container (includes trtexec, Python bindings, samples)
docker pull nvcr.io/nvidia/tensorrt:24.05-py3

# Triton Inference Server (includes TRT backend)
docker pull nvcr.io/nvidia/tritonserver:24.05-py3

# TensorRT-LLM (includes quantization scripts)
docker pull nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3
```

**Critical rule:** Build engines inside the same container image used in production. An engine built in a different CUDA/TensorRT/cuDNN combination may fail to load or produce incorrect results.

> **Container Hygiene — driver compatibility:** The NVIDIA driver on the **host** machine must be equal to or newer than the CUDA version inside the container. For example, if the NGC container uses CUDA 12.4, the host driver must support CUDA ≥12.4 (driver ≥550.54). Check with `nvidia-smi` on the host — the "CUDA Version" in the top-right corner is the maximum CUDA version the driver supports. If it's lower than what the container expects, you'll get cryptic `CUDA_ERROR_NO_DEVICE` or `cudaErrorInsufficientDriver` at runtime — not at container pull time.

> **Why not just use PyTorch?** Eager-mode PyTorch is often measurably slower than a compiled TensorRT engine at the same precision — the gap varies by model, but the causes are structural: PyTorch dispatches kernels one at a time through the Python runtime, with no cross-layer fusion, no kernel auto-tuning, and no memory layout optimization. TensorRT's compilation cost is the "tax" you pay once to eliminate this overhead permanently. For a single inference call, PyTorch is fine. For serving millions of requests, TensorRT pays for itself immediately. **Don't take this on faith — measure it on your model with `trtexec` vs. your PyTorch baseline.**

---

## The Hardware: Tensor Cores and Their Quantization Support

> **Optional.** You do not need to understand Tensor Core internals to build and serve a TensorRT engine. If you completed the quickstart and want to get to calibration and serving, skip ahead to the **Intermediate** tier ("TensorRT — The Core Inference Engine"). Come back here when you need to understand *why* INT8 is faster or *which* GPU supports FP8.

NVIDIA GPUs execute quantized operations on **Tensor Cores** — specialized hardware units inside each Streaming Multiprocessor (SM) that perform matrix multiply-accumulate operations on low-precision data types.

Understanding what Tensor Cores support at the hardware level explains everything about why TensorRT makes the choices it does.

---

### Volta, Turing, Ampere, Hopper — What Changed for Quantization

Each GPU architecture generation expanded quantization support:

| Architecture | GPU Examples | Year | Quantization Support | Key Feature |
|-------------|-------------|------|---------------------|-------------|
| **Volta** | V100 | 2017 | FP16 Tensor Cores | First Tensor Cores, FP16 only |
| **Turing** | T4, RTX 2080 | 2018 | FP16 + INT8 | INT8 Tensor Cores via IMMA (DP4A) |
| **Ampere** | A100, A10, RTX 3090 | 2020 | FP16 + INT8 + TF32 + BF16 | Structured sparsity (2:4), IMMA v2 |
| **Ada Lovelace** | L4, L40, RTX 4090 | 2022 | FP16 + INT8 + FP8 (arch-dependent) | FP8 tensor capability advertised; TensorRT FP8 acceleration is workload + version dependent — always validate via TensorRT Support Matrix and `trtexec --fp8` on target |
| **Hopper** | H100, H200 | 2023 | FP16 + INT8 + FP8 (native) | Native FP8 Tensor Cores, Transformer Engine |
| **Blackwell** | B100, B200, GB200 | 2024 | FP16 + INT8 + FP8 + **NVFP4** | NVFP4 Tensor Cores (two-level scaling), 2nd-gen Transformer Engine |

> **Pro-Tip — NVFP4 is not just "FP4":** Blackwell uses **NVFP4** (NVIDIA's Microscaling implementation), which is a two-level scaling strategy. A micro-block of 16 values shares a single E4M3 (FP8) scale factor, and there is a second-level FP32 per-tensor scale on top. The result is a "4.5-bit" representation — 4 data bits + amortized scale overhead — that achieves FP8-level accuracy at 4-bit bandwidth. This is why NVFP4 models maintain accuracy that surprises people expecting the usual INT4 degradation: the FP8 micro-scales preserve fine-grained range information that fixed-point INT4 cannot.

> **Ada Lovelace note:** Some Ada GPUs advertise FP8 tensor capability, but TensorRT FP8 enablement and acceleration is workload-dependent and varies across TensorRT releases. Always confirm via the TensorRT Support Matrix and a micro-benchmark (`trtexec --fp8`) on your specific GPU before committing to FP8 on Ada.

**The takeaway:** Each generation adds lower-precision support. Volta could only accelerate FP16. Turing added INT8. Hopper added FP8. Blackwell adds FP4. Lower precision = more operations per cycle = higher throughput.

---

### TensorRT-Supported Quantized Types

TensorRT supports the following quantized data types (as of TensorRT 10.x):

| Type | Bits | Format | Scaling | Use Case |
|------|------|--------|---------|----------|
| **INT8** | 8 | Integer | Scale + zero-point (per-tensor or per-channel) | CNNs, general models |
| **INT4** | 4 | Integer | Scale + zero-point (per-group, weight-only) | LLM weight compression |
| **FP8 E4M3** | 8 | Floating-point | Per-tensor or per-channel scaling (amax-based) | Hopper+ weights and activations |
| **FP4 E2M1** | 4 | Floating-point | Block-wise scaling | Blackwell weight compression |

**Connecting to the book's framework:** INT8/INT4 follow the scale+zero-point contract from Chapter 3. FP8/FP4 follow floating-point grids (Chapter 19) but still use scaling — the grid is logarithmic rather than uniform, but the "representational constraint" (Chapter 2) still applies.

Always check the TensorRT Support Matrix for the canonical list of what types are supported on which GPU + TensorRT version combination.

---

### The NVIDIA Compute Capability Map

Just as Chapter 20 maps Snapdragon generations to capabilities, here is the NVIDIA compute capability map for quantization:

| Compute Capability | Architecture | Key Quantization Instructions | Sparsity Support | Quantization Implication |
|-------------------|-------------|-------------------------------|-----------------|-------------------------|
| SM 7.0 | Volta (V100) | FP16 HMMA | None | FP16 only; no INT8 acceleration |
| SM 7.5 | Turing (T4) | DP4A, INT8 IMMA | None | First INT8 tensor core support |
| SM 8.0 | Ampere (A100) | INT8 IMMA v2, TF32 | **2:4 structured sparsity** | Sparse Tensor Cores can double INT8 throughput (624→1248 TOPS on A100) |
| SM 8.6 | Ampere (A10, RTX 3090) | INT8 IMMA v2 | 2:4 structured sparsity | Same as SM 8.0 but different core count |
| SM 8.9 | Ada Lovelace (L4, L40) | INT8 IMMA, FP8 (version-dependent) | 2:4 structured sparsity | Check TensorRT Support Matrix for FP8 |
| SM 9.0 | Hopper (H100) | FP8 HMMA, INT8 IMMA | 2:4 structured sparsity | Native FP8, Transformer Engine |
| SM 10.0 | Blackwell (B200) | FP4, FP8 block scaling | 2:4 structured sparsity + **Microscaling (MX) formats** | FP4 tensor cores, 2nd-gen TE; MX formats enable hardware-level block-wise scaling for FP4/FP6/FP8, defining how sub-8-bit quantization scales in future architectures |

> **Sparsity + Quantization compounding:** On Ampere and later, you can combine 2:4 structured sparsity with INT8 quantization. The Sparse Tensor Core skips zero-valued weights (2 out of every 4) and computes on the remaining INT8 values. The result is approximately 4× the FP16 dense throughput — 2× from INT8 precision and 2× from sparsity. ModelOpt provides the tooling to prune weights into the 2:4 pattern and then quantize.

> **Microscaling (MX) formats on Blackwell:** The "block-wise scaling" listed for FP4 E2M1 on Blackwell is implemented via Microscaling (MX) formats — an industry standard (adopted by NVIDIA, AMD, Intel, and others) where a small block of values (e.g., 32 elements) shares a single scale factor stored at higher precision. This is the hardware-level foundation for how sub-8-bit floating-point quantization (FP4, FP6, FP8 with block scaling) will scale across future GPU generations. Think of MX as "groupwise quantization (Chapter 16) implemented in silicon."

**Why this matters:** When TensorRT builds an engine, it selects kernels based on compute capability. An engine built for SM 9.0 uses FP8 HMMA instructions that don't exist on SM 8.0 hardware — this is the fundamental reason engines are GPU-specific.

---

### INT8 Tensor Cores

Starting with Turing, NVIDIA Tensor Cores can perform INT8 matrix multiplication with INT32 accumulation:

$$C_{int32} = A_{int8} \times B_{int8} + C_{int32}$$

This is the same pattern as Qualcomm's HTP — multiply in low precision, accumulate in high precision to avoid overflow.

**Throughput advantage (peak theoretical):** On an A100, INT8 Tensor Core throughput is **2× FP16** throughput:
- FP16: 312 TFLOPS
- INT8: 624 TOPS

This means a model quantized to INT8 can theoretically run 2× faster than FP16. In practice, the speedup is 1.3–1.8× because memory bandwidth (not compute) is often the bottleneck, and requantization between layers adds overhead.

> **Connecting to Chapter 16:** If your workload is memory-bound (e.g., batch-1 autoregressive LLM decode), then INT8 compute throughput doesn't matter unless you also reduce memory traffic. Weight-only quantization (W4A16, W8A16) gives you the memory bandwidth reduction without requiring activation quantization. If your workload is compute-bound (large batch CNN inference, prefill phase of LLMs), INT8 compute throughput directly translates to speedup.

**Key constraint:** Tensor Cores operate on specific matrix tile sizes. For Ampere: 16×16×16 tiles. Matrix dimensions must be multiples of 16 for maximum efficiency. Non-aligned dimensions get padded, wasting some throughput. TensorRT handles this padding automatically, but an expert architect ensures matrix dimensions are multiples of 8 or 16 (depending on architecture) to avoid "dead cycles" in the Tensor Core pipeline.

> **Hardware depth — Register Pressure and Occupancy:** Beyond raw TOPS, quantization helps GPUs in a subtler way. INT8 values occupy half the register space of FP16. This reduces *register pressure*, allowing the GPU scheduler to keep more warps active simultaneously (higher occupancy). Higher occupancy means better latency hiding and higher sustained throughput. This is analogous to how quantization reduces VTCM pressure on Qualcomm's HTP (Chapter 20), but expressed through the GPU's warp-based execution model.

> **Hardware depth — The Memory Hierarchy and Why Quantization Gives More Than 2×:**
>
> Quantization isn't just about faster Tensor Core math — it fundamentally changes how data moves through the GPU's memory hierarchy. An NVIDIA data-center GPU has three levels of memory, each with dramatically different bandwidth:
>
> | Memory Level | Typical Size (A100/H100) | Bandwidth | Latency |
> |-------------|------------------------|-----------|---------|
> | **HBM (Global Memory)** | 40–80 GB | 2–3.4 TB/s | ~400 cycles |
> | **L2 Cache** | 40–50 MB | ~12 TB/s | ~200 cycles |
> | **Shared Memory / L1 (SRAM)** | 192–228 KB per SM | ~19 TB/s | ~30 cycles |
>
> **Why this matters for quantization:** When you quantize weights from FP16 to INT8, the model is half the size. A 25 MB layer that barely fits in L2 at FP16 now fits comfortably at 12.5 MB in INT8. If the FP16 version spills to HBM (2 TB/s) but the INT8 version stays in L2 (~12 TB/s), the effective bandwidth improvement is ~6×, far exceeding the theoretical 2× Tensor Core compute gain. This is why real-world INT8 speedups sometimes reach 3–5× — the compute speedup and the cache residency improvement multiply together.
>
> **The practical implication:** For models that are borderline L2-resident (total weight size near 40–50 MB), quantization to INT8 or INT4 can produce outsized speedups that surprise engineers who only think about Tensor Core throughput. Conversely, for models that are already fully L2-resident at FP16 (small models), the speedup will be closer to the theoretical compute-only 1.3–1.8× range.

---

### FP8 Tensor Cores (Hopper and Beyond)

Hopper (H100) introduced native FP8 Tensor Cores. FP8 has two formats:

- **E4M3** — 4 exponent bits, 3 mantissa bits. Range: ±448, precision: ~0.1%. Used for weights and activations in forward pass.
- **E5M2** — 5 exponent bits, 2 mantissa bits. Range: ±57344, precision: ~0.5%. Used for gradients (wider range needed).

**Why FP8 matters for quantization:**

FP8 offers a middle ground between INT8 and FP16:
- **More forgiving than INT8** — FP8 is a floating-point format, so it naturally handles a wider dynamic range. It avoids INT8-style zero-points and asymmetric encoding, but **scale management is still required.** Each tensor needs a scaling factor (amax-based) to place values inside the representable range, especially for E4M3 with its ±448 limit.
- **Faster than FP16** — FP8 throughput on H100 is 2× FP16 (same as INT8).
- **Simpler workflow than INT8** — no zero-points, no asymmetric encoding, no complex calibration histogram search. But you still need per-tensor or per-channel scaling factors, typically computed from observed amax values during a calibration pass or maintained dynamically.

**The key distinction from INT8:** INT8 quantization uses the scale+zero-point contract (Chapter 3). FP8 uses a floating-point grid but still requires scaling to avoid overflow/underflow. Think of it as: "FP8 is more forgiving than INT8, but scale management is still required."

**H100 FP8 throughput (peak theoretical — SXM variant):**
- FP8: 1,979 TFLOPS (dense) / 3,958 TFLOPS (sparse)
- FP16: 990 TFLOPS (dense) / 1,979 TFLOPS (sparse)
- INT8: 1,979 TOPS (dense) / 3,958 TOPS (sparse)

> **Important:** These are marketing peak numbers and vary by SKU (SXM vs PCIe), clock speed, and whether sparsity is enabled. Peak throughput ≠ achieved throughput. Always measure achieved throughput with `trtexec` or Nsight on your actual hardware and workload.

FP8 and INT8 have the same peak throughput — but FP8 has a simpler quantization workflow (no zero-point, no asymmetric encoding, just scaling factors).

**NVIDIA's Transformer Engine** automatically manages FP8 scaling during training and inference. It maintains per-tensor scaling factors (computed from amax history) and updates them dynamically, handling the tricky details of FP8 range management. During inference, TensorRT manages FP8 scaling as part of its quantization framework — the user provides or calibrates the scaling factors, and TensorRT fuses the scale operations into the compute kernels.

---

# ── INTERMEDIATE TIER ──

> **Intermediate Goals:** Correctness + reproducibility + deployability. After this tier, your INT8 accuracy is validated (Polygraphy diff gate), Triton serving works with dynamic batching, and dynamic shapes are supported with profiles.
>
> **Skip guidance:** If you already have a validated INT8/FP8 engine in Triton and need to optimize performance, jump to the **Advanced** tier. If you need multi-GPU, SLO tuning, or deep debugging, jump to **Expert**.

## TensorRT — The Core Inference Engine

TensorRT is NVIDIA's inference compiler and runtime. It takes a trained model (ONNX, PyTorch, or TensorFlow), optimizes it for a specific GPU, and produces a serialized **engine** file that runs with minimal overhead.

TensorRT is to NVIDIA GPUs what QNN is to Qualcomm Hexagon — the bridge between a framework-level model and hardware-optimized execution.

---

### What TensorRT Actually Does

TensorRT performs several optimizations during the **build** phase:

1. **Layer fusion** — combines multiple sequential operations into a single kernel launch. Conv + BatchNorm + ReLU → one fused kernel instead of three.
2. **Precision calibration** — determines optimal INT8 scales for each tensor using a calibration dataset.
3. **Kernel auto-tuning** — benchmarks multiple kernel implementations for each operation on the target GPU and selects the fastest one.
4. **Memory optimization** — plans memory allocation to minimize peak usage and maximize data reuse.
5. **Tensor format selection** — chooses the optimal memory layout (NCHW, NHWC, or NVIDIA's internal blocked formats) for each operation.

The build phase is expensive — it can take minutes to hours for large models. But it runs once. The resulting engine is fast to load and fast to execute.

---

### The TensorRT Optimization Pipeline

```
Input Model (ONNX)
       │
       ▼
┌──────────────────────┐
│   Parser              │  ← Reads ONNX/UFF/Caffe, builds network definition
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│   Graph Optimization  │  ← Layer fusion, dead code elimination,
│                       │     constant folding, tensor format selection
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│   Precision Selection │  ← Mark layers as FP32/FP16/INT8/FP8
│   + Calibration       │     Run calibration dataset for INT8
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│   Kernel Auto-Tuning  │  ← Time multiple kernel implementations
│   (per operation)     │     per operation, select fastest
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│   Engine Serialization│  ← Produce .plan file for deployment
└──────────────────────┘
```

**Key insight:** TensorRT engines are **GPU-specific by default.** An engine built on an A100 will not run on a T4. You must rebuild for each target GPU. This is because the kernel auto-tuning selects different kernels for different GPU architectures.

However, TensorRT provides two mechanisms that relax this constraint:

1. **Version-compatible engines** — allow an engine built with one TensorRT version to run on a newer TensorRT version (within constraints). Requires specific build flags (`--versionCompatible` in trtexec) and can increase plan file size. Note: version-compatible engines require the "trusted plan" security mechanism.

2. **Hardware compatibility mode** — allows an engine to run across multiple GPU architectures within the same TensorRT version, at a performance cost. TensorRT generates more generic kernels that work across architectures instead of architecture-specific optimized kernels.

```bash
# Build a version-compatible engine
trtexec --onnx=model.onnx --saveEngine=model.plan --versionCompatible

# Build with hardware compatibility
trtexec --onnx=model.onnx --saveEngine=model.plan --hardwareCompatibilityLevel=ampere+
```

> **Production rule:** Default to building engines on the exact target GPU and TensorRT version used in production. Use version/hardware compatibility only when fleet heterogeneity demands it, and always benchmark to quantify the performance cost.

---

### Layer Fusion in TensorRT

Layer fusion is TensorRT's most impactful optimization. Each kernel launch on a GPU has overhead (~5–10μs). A model with 200 layers would spend 1–2ms just on kernel launch overhead. Fusion reduces the number of kernel launches dramatically.

**Common fusion patterns:**

```
Conv + BatchNorm + ReLU     → Single fused kernel
Conv + Add + ReLU           → Single fused kernel (residual connections)
MatMul + Add (bias)         → Single fused kernel
Shuffle + Reshape            → Eliminated (metadata-only, no compute)
```

**Quantization-specific fusions:**
```
Quantize + Conv + Dequantize     → INT8 Conv kernel (no explicit quant/dequant)
DQ(W) + MatMul + DQ(X) → Q(Y)   → INT8 MatMul with fused requantization
```

Without fusion, quantized models would need explicit quantize and dequantize operations around every layer — each requiring a kernel launch and memory read/write. TensorRT fuses these into the compute kernels themselves, so the INT8 conversion happens inside the matrix multiply kernel at zero additional cost.

**Worked example:** ResNet-50 has ~53 layers. After TensorRT fusion:
- FP32: ~53 kernel launches → ~20 fused kernels
- INT8: ~53 layers + ~106 quant/dequant → ~18 fused kernels (quant/dequant absorbed)

---

### TensorRT Calibration: How It Chooses Scales

When you enable INT8 mode, TensorRT needs quantization parameters (scales) for every activation tensor. Unlike weights (which are known at build time), activations depend on the input data. This is the same calibration challenge described in Chapter 9 (Calibration and Observers), but implemented inside TensorRT's builder rather than in PyTorch.

> **Cross-reference:** If your calibration data doesn't represent production inputs, the computed scales will be wrong — this is the **Calibration Mismatch** failure pattern from Chapter 13. If your model is a transformer with activation outliers, INT8 calibration may trigger **Resolution Collapse** (Chapter 14) — SmoothQuant (Chapter 15) or FP8 are the fixes.

TensorRT solves this with **calibration**: you provide a representative dataset (typically 500–1000 samples), TensorRT runs inference in FP32, records the activation distributions, and then selects optimal quantization parameters.

TensorRT provides three calibration algorithms:

---

#### Entropy Calibration

**How it works:** For each activation tensor, TensorRT:
1. Records the histogram of values during calibration
2. For each candidate clipping threshold $T$, computes the KL divergence between the original FP32 distribution and the quantized INT8 distribution
3. Selects the threshold $T$ that minimizes KL divergence

$$T^* = \arg\min_T D_{KL}(P_{fp32} \| Q_{int8}(T))$$

**Why KL divergence?** KL divergence measures how much information is lost when approximating the FP32 distribution with the INT8 distribution. Minimizing it means the quantized distribution is as close as possible to the original, in an information-theoretic sense.

**Trade-off:** Entropy calibration may clip outliers — it does not necessarily preserve the max value. If 0.1% of values are at 100.0 but 99.9% are below 5.0, entropy calibration might set the threshold at 8.0, clipping the outliers but preserving resolution for the bulk of values.

This is the **default** calibration method in TensorRT and works well for most CNN models.

---

#### MinMax Calibration

**How it works:** Simply uses the minimum and maximum observed values as the quantization range:

$$S = \frac{\max(|x_{observed}|)}{127}$$

**When to use:** When outlier preservation is critical — e.g., when even the maximum activation value carries important information. This is conservative but wastes resolution if outliers are rare.

---

#### Percentile Calibration

**How it works:** Uses the 99.99th percentile (or another configurable percentile) of the observed values instead of the true maximum:

$$S = \frac{\text{percentile}(|x|, 99.99)}{127}$$

**When to use:** When there are extreme outliers that should be clipped, but you want more control than entropy calibration provides.

#### Which Calibration Algorithm for Which Model?

| Model Type | Recommended Algorithm | Why |
|-----------|----------------------|-----|
| **CNNs (ResNet, EfficientNet, YOLO)** | Entropy (default) | Activation distributions are smooth bell-curves; entropy finds the optimal clip point that preserves the most information |
| **Transformers (BERT, ViT)** | MinMax or Percentile (99.99%) | Transformers have activation outliers (Chapter 14) that entropy calibration may clip too aggressively, causing accuracy collapse. MinMax preserves outliers at the cost of resolution |
| **LLMs (LLaMA, Mistral, GPT)** | **Skip TensorRT calibrator entirely** — use ModelOpt SmoothQuant/AWQ or FP8 | LLM activation outlier channels require outlier-aware algorithms (SmoothQuant, Chapter 15). Standard TensorRT calibrators don't handle per-channel outlier migration |

> **Rule:** For any transformer-based model, prefer explicit Q/DQ via ModelOpt over TensorRT's built-in calibrator. The built-in calibrator is designed for CNN-like smooth distributions and can produce poor scales for attention-heavy models.

---

### TensorRT Precision Modes: FP32, FP16, INT8, FP8

TensorRT supports mixed-precision inference. You can set the **default** precision and then override individual layers:

```python
# Builder configuration
config = builder.create_builder_config()

# Enable FP16 (almost always a good idea on any modern GPU)
config.set_flag(trt.BuilderFlag.FP16)

# Enable INT8 (requires calibration)
config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = MyCalibrator(calibration_data)

# Enable FP8 (Hopper and later)
config.set_flag(trt.BuilderFlag.FP8)
```

**When both FP16 and INT8 are enabled,** TensorRT automatically decides per-layer which precision to use, based on which produces the fastest execution while maintaining acceptable accuracy. Some layers may run in FP16 (because the INT8 kernel is slower for that layer's specific dimensions), while others run in INT8.

You can also force specific layers to specific precisions:
```python
# Force a sensitive layer to FP16
layer = network.get_layer(idx)
layer.precision = trt.float16
layer.set_output_type(0, trt.float16)
```

This is the TensorRT equivalent of mixed-precision quantization — critical layers stay in higher precision while the bulk of computation runs in INT8.

---

### Explicit vs. Implicit Quantization (and Why Implicit Is Deprecated)

TensorRT has two fundamentally different quantization workflows. Understanding the distinction is critical for production use:

**Implicit quantization (legacy — deprecated):**
- You enable `BuilderFlag.INT8` and provide a calibrator
- TensorRT decides which layers to quantize based on its internal heuristics
- The ONNX model has no quantization nodes — TensorRT adds quantization internally
- You have limited control over what gets quantized and how

**Why implicit fails in practice — kernel splitting and broken vertical fusion:**
Without Q/DQ nodes, TensorRT has to *guess* where precision boundaries lie. When it guesses wrong, the result is **kernel splitting**: TensorRT inserts a `Reformat` layer between adjacent ops because it has no explicit signal that they should fuse. This reformat layer copies and converts the tensor — adding latency and killing the speedup you expected from quantization.

**Concrete example — vertical fusion broken by a misplaced DQ:**

A classic TensorRT optimization is **vertical fusion**: `Conv → Bias → ReLU` collapses into a single kernel launch. Now consider what happens when implicit quantization places a precision boundary in the wrong place:

```
✅ Correct (explicit Q/DQ, vertical fusion preserved):
  Q(input) → [Conv + Bias + ReLU]_INT8 → DQ(output)
  ↑ One fused kernel, one launch, Tensor Core path

❌ Broken (implicit mode guessed wrong):
  Q(input) → Conv_INT8 → Reformat(INT8→FP16) → Bias_FP16 → ReLU_FP16 → Reformat(FP16→INT8)
  ↑ Three kernels, two reformats, vertical fusion destroyed
```

The second path can be *slower than FP16* because the reformat overhead exceeds the INT8 speedup. This is the fundamental reason implicit quantization is unreliable: you cannot predict where TensorRT will place precision boundaries, and a single bad boundary can cascade through the graph.

With explicit Q/DQ nodes, you control where quantization starts and stops. If you place Q before Conv and DQ after ReLU, TensorRT knows the entire `Conv → Bias → ReLU` chain is INT8 and fuses it into one Tensor Core kernel. If ModelOpt places a DQ node sub-optimally (e.g., between Conv and ReLU), you can use `onnx-graphsurgeon` to move it — see the Q/DQ Inspection Mini-Lab below.

**Explicit quantization (recommended):**
- The ONNX model contains explicit `QuantizeLinear` / `DequantizeLinear` (Q/DQ) nodes
- These nodes specify the exact scale and zero-point for each quantized tensor
- TensorRT reads the Q/DQ nodes and uses them directly — no heuristic decisions
- You have full control and reproducibility

```
Implicit:  ONNX model → TensorRT calibrates → TensorRT decides quantization
Explicit:  ONNX model + Q/DQ nodes → TensorRT uses provided scales → deterministic
```

**Why explicit Q/DQ is preferred:**
1. **Reproducibility** — the same ONNX file always produces the same quantized engine
2. **Portability** — the Q/DQ ONNX can be consumed by other runtimes (ONNXRuntime, etc.)
3. **Alignment with ModelOpt/QAT** — ModelOpt exports Q/DQ nodes; TRT consumes them directly
4. **Explicit is the future** — implicit quantization is deprecated in modern TensorRT

> **Decision box — which workflow to use:**
> - **CNN quick PTQ (legacy, still works):** TensorRT calibrator with `BuilderFlag.INT8`. Acceptable for simple CNNs where you just want fast INT8 and don't need Q/DQ portability.
> - **Transformers / production (recommended):** Explicit Q/DQ via ModelOpt (`mtq.quantize` → `mtq.export`). This is the only path for SmoothQuant, AWQ, FP8, and any model where you need reproducible, debuggable quantization.
>
> **The principle:** Explicit quantization is a contract between the developer and the compiler: "I have verified these scales; do not move them."

> **Implicit quantization is deprecated.** New projects should always use explicit Q/DQ. If you have legacy pipelines using implicit calibration, migrate by: (1) running ModelOpt PTQ to produce a Q/DQ ONNX, then (2) building the engine from that ONNX without a calibrator.

**Explicit Q/DQ flow:**
```python
# ModelOpt produces ONNX with Q/DQ nodes
import modelopt.torch.quantization as mtq
model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=calibrate)
mtq.export(model, "model_qdq.onnx", dummy_input)

# TensorRT consumes Q/DQ ONNX — no calibrator needed
config.set_flag(trt.BuilderFlag.INT8)
# No calibrator! Scales come from Q/DQ nodes in the ONNX
engine = builder.build_serialized_network(network, config)
```

**What Q/DQ nodes actually look like in the graph:**

When you hear "explicit Q/DQ," you might wonder what the ONNX graph physically contains. Here is what changes. In an unquantized ONNX graph, a Conv layer looks like:

```
weights (FP32) ──→ Conv ──→ output (FP32)
input (FP32)  ──↗
```

In an explicit Q/DQ ONNX graph, Q/DQ nodes wrap the quantized tensors:

```
weights (FP32) → QuantizeLinear(scale=0.02, zp=0) → DequantizeLinear → Conv → QuantizeLinear → DequantizeLinear → output (FP32)
input (FP32)   → QuantizeLinear(scale=0.05, zp=0) → DequantizeLinear ──↗
```

Each `QuantizeLinear` node stores a `scale` and `zero_point` — these are the exact quantization parameters computed during calibration or QAT. TensorRT reads these nodes and fuses the Q/DQ operations into the Conv kernel itself, so no separate quantize/dequantize kernels run at inference time.

**Inspecting Q/DQ nodes with onnx-graphsurgeon:**

You can inspect (or manually insert) Q/DQ nodes using NVIDIA's `onnx-graphsurgeon`:

```python
import onnx
import onnx_graphsurgeon as gs
import numpy as np

# Load the Q/DQ ONNX model
graph = gs.import_onnx(onnx.load("model_qdq.onnx"))

# Find all QuantizeLinear / DequantizeLinear nodes
qdq_nodes = [n for n in graph.nodes if n.op in ("QuantizeLinear", "DequantizeLinear")]
print(f"Found {len(qdq_nodes)} Q/DQ nodes in the graph")

# Inspect a specific Q/DQ node's scale and zero-point
for node in qdq_nodes[:5]:
    scale = node.inputs[1]  # scale is the second input
    zp = node.inputs[2] if len(node.inputs) > 2 else "none (symmetric)"
    print(f"  {node.op} '{node.name}': scale={scale}, zero_point={zp}")

# Manually add Q/DQ nodes to a specific tensor (advanced usage):
# This is what ModelOpt does automatically during mtq.quantize()
conv_node = [n for n in graph.nodes if n.op == "Conv"][0]
input_tensor = conv_node.inputs[0]

# Create Q/DQ nodes for the input
scale = gs.Constant("input_scale", np.array(0.05, dtype=np.float32))
zp = gs.Constant("input_zp", np.array(0, dtype=np.int8))
q_out = gs.Variable("input_quantized", dtype=np.int8)
dq_out = gs.Variable("input_dequantized", dtype=np.float32)

q_node = gs.Node(op="QuantizeLinear", inputs=[input_tensor, scale, zp], outputs=[q_out])
dq_node = gs.Node(op="DequantizeLinear", inputs=[q_out, scale, zp], outputs=[dq_out])

graph.nodes.extend([q_node, dq_node])
conv_node.inputs[0] = dq_out  # Wire the DQ output into the Conv input

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "model_with_qdq.onnx")
```

> **The mental model:** Q/DQ nodes are the "explicit contract" between your quantization tool (ModelOpt, QAT, GPTQ) and TensorRT. Without them, TensorRT has to *guess* where and how to quantize (implicit mode). With them, TensorRT follows your *instructions* exactly. This is why explicit Q/DQ is portable and reproducible — the quantization decisions travel with the ONNX file, not hidden inside a TensorRT calibration cache.

### Q/DQ Inspection Mini-Lab (Intermediate)

After running ModelOpt PTQ, always inspect the Q/DQ scales before building. Suspicious values indicate calibration problems:

```python
# Available in: NGC tensorrt container (onnx-graphsurgeon is pre-installed)
import onnx, onnx_graphsurgeon as gs, numpy as np

graph = gs.import_onnx(onnx.load("model_qdq.onnx"))
qdq_nodes = [n for n in graph.nodes if n.op == "QuantizeLinear"]

print(f"Total QuantizeLinear nodes: {len(qdq_nodes)}")
for node in qdq_nodes[:10]:  # first 10
    scale_val = node.inputs[1].values if hasattr(node.inputs[1], 'values') else "dynamic"
    print(f"  {node.name}: scale={scale_val}")
```

**What's suspicious — red flag thresholds:**

| Scale Value | What It Means | Action |
|------------|---------------|--------|
| `0.0` | Zero scale — quantization will map everything to zero | Bug in calibration. Re-run with more representative data |
| `NaN` or `inf` | Numerical overflow during calibration | Check for exploding activations in the model; normalize inputs |
| `< 1e-10` | Extremely tiny scale — values are near-zero, wasting all quantization bins | Layer may not need quantization; consider keeping in FP16 |
| `> 100` | Very large scale — activations have extreme outliers | Classic outlier issue (Chapter 14). Apply SmoothQuant (Chapter 15) |
| All scales identical | Calibration data was constant or calibrator didn't run | Check that calibration loop actually processes diverse inputs |

---

### Silent Precision Fallback — The NVIDIA Equivalent of CPU Fallback

Just as Qualcomm can silently fall back from HTP to CPU (Chapter 20), NVIDIA can silently fall back from INT8 to FP16. If TensorRT's builder determines that the INT8 kernel for a specific layer is slower or unsupported, it silently selects the FP16 kernel instead.

This is usually fine for performance (TensorRT picks the faster kernel), but it means your "INT8 engine" may have layers running in FP16 without you knowing.

**How to detect it:**
```bash
# Verbose build shows per-layer precision decisions
trtexec --onnx=model.onnx --int8 --fp16 --verbose 2>&1 | grep "Tactic"
```

**How to force errors instead of silent fallback:**
```python
# Force TensorRT to error if it can't find a valid INT8 kernel
config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)
```

With `REJECT_EMPTY_ALGORITHMS`, TensorRT will fail the build instead of silently falling back. Use this during development to understand which layers truly support INT8 on your target GPU.

---

### TensorRT Builder API Walkthrough

Complete example: building an INT8 TensorRT engine from an ONNX model.

```python
import os
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

# Logger
logger = trt.Logger(trt.Logger.WARNING)

# 1. Create builder and network
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)

# 2. Parse ONNX model
parser = trt.OnnxParser(network, logger)
with open("resnet50.onnx", "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

# 3. Configure builder
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB workspace

# Enable FP16 + INT8
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.INT8)

# 4. INT8 Calibrator
class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, input_shape, cache_file="calibration.cache"):
        super().__init__()
        self.data_loader = iter(data_loader)
        self.cache_file = cache_file
        self.input_shape = input_shape   # e.g., (32, 3, 224, 224)
        self.batch_size = input_shape[0]
        self.device_input = cuda.mem_alloc(
            int(np.prod(input_shape) * np.float32().nbytes)
        )

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        try:
            batch = next(self.data_loader)  # must yield np.ndarray float32
            batch = np.ascontiguousarray(batch.astype(np.float32))
            assert batch.shape == self.input_shape, (
                f"Expected {self.input_shape}, got {batch.shape}"
            )
            cuda.memcpy_htod(self.device_input, batch)
            return [int(self.device_input)]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        if os.path.isfile(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# input_shape must match your dataloader's batch shape
config.int8_calibrator = EntropyCalibrator(
    calibration_loader, input_shape=(32, 3, 224, 224)
)
```

> **You must provide: `calibration_loader`** — a Python iterator that yields NumPy arrays of shape `(batch_size, C, H, W)` with dtype `float32`, preprocessed identically to your training data. A minimal example:
> ```python
> import torchvision.transforms as T
> from torch.utils.data import DataLoader
> transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
>                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
> dataset = torchvision.datasets.ImageFolder("/path/to/calib_images", transform=transform)
> calibration_loader = DataLoader(dataset, batch_size=32, shuffle=False)
> # Then wrap: calibration_loader = (batch[0].numpy() for batch in calibration_loader)
> ```
> **Common calibration failures:** shape mismatch (dataloader yields wrong batch size), non-contiguous arrays (fix with `np.ascontiguousarray`), wrong dtype (must be `float32` even for INT8 calibration), preprocessing that differs from training (different normalization = bad scales → Calibration Mismatch, Chapter 13).

### Minimal Working Calibration Dataset (Beginner-Friendly)

If you don't have a calibration dataset ready, here is a self-contained example that works inside the NGC TensorRT container with no extra downloads:

```python
# Minimal calibration dataset — works inside NGC tensorrt container
# Dependencies: numpy (pre-installed), torch + torchvision (pre-installed in NGC)
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets import FakeData  # built-in, no download needed

# Option A: Synthetic data (for testing the pipeline — NOT for production accuracy)
def make_synthetic_calibration_loader(batch_size=32, num_batches=50):
    """Yields numpy float32 batches of shape (batch_size, 3, 224, 224)."""
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])])
    dataset = FakeData(size=batch_size * num_batches, image_size=(3, 256, 256),
                       transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for images, _ in loader:
        yield images.numpy()  # shape: (batch_size, 3, 224, 224), dtype: float32

# Option B: Real data from a local folder (for production)
# def make_real_calibration_loader(image_dir, batch_size=32):
#     transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
#                            T.Normalize(mean=[0.485, 0.456, 0.406],
#                                        std=[0.229, 0.224, 0.225])])
#     dataset = torchvision.datasets.ImageFolder(image_dir, transform=transform)
#     loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
#     for images, _ in loader:
#         yield images.numpy()

# Usage with the EntropyCalibrator:
calibration_loader = make_synthetic_calibration_loader(batch_size=32, num_batches=50)
# config.int8_calibrator = EntropyCalibrator(calibration_loader, input_shape=(32, 3, 224, 224))
```

> **Warning:** Synthetic data produces valid calibration scales for pipeline testing but NOT production-quality scales. For deployment, replace with 500–1000 representative images from your production dataset.

```python
# 5. Build engine (this is the slow step - kernel auto-tuning)
engine_bytes = builder.build_serialized_network(network, config)

# 6. Save engine
with open("resnet50_int8.plan", "wb") as f:
    f.write(engine_bytes)
```

**Build time:** For ResNet-50, expect 2–10 minutes depending on the GPU. For large models, expect 30–60 minutes. The auto-tuning step benchmarks many kernel variants.

**Calibration cache:** The calibrator saves computed scales to a cache file. On subsequent builds, the cache is reused — so calibration data is only needed once.

---

### TensorRT Engine Serialization and Deployment

Once built, the engine is a binary blob that contains:
- The optimized computation graph
- All weights (in the selected precision)
- The selected kernel implementations
- Memory allocation plans

**Loading and running an engine:**

> **You must provide:** A `preprocess()` function that applies the same normalization, resize, and channel ordering used during training (e.g., ImageNet mean/std normalization, RGB channel order, resize to 224×224). A `softmax()` function (e.g., `scipy.special.softmax` or a simple NumPy implementation). These are model-specific and cannot be generated by TensorRT.

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Load engine
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
with open("resnet50_int8.plan", "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

# Create execution context
context = engine.create_execution_context()

# --- Binding resolution (do NOT assume input is binding 0) ---
# TensorRT engines can have multiple inputs/outputs in any order.
# Always resolve by name, not by index.
input_binding_idx = engine.get_binding_index("input")     # name must match ONNX input name
output_binding_idx = engine.get_binding_index("output")    # name must match ONNX output name

input_shape = engine.get_binding_shape(input_binding_idx)   # e.g., (1, 3, 224, 224)
output_shape = engine.get_binding_shape(output_binding_idx) # e.g., (1, 1000)

# Allocate device buffers
d_input = cuda.mem_alloc(int(np.prod(input_shape) * np.float32().nbytes))
d_output = cuda.mem_alloc(int(np.prod(output_shape) * np.float32().nbytes))
h_output = np.empty(output_shape, dtype=np.float32)

# Prepare input — preprocess() must match your training pipeline exactly
# Example: input_data = (resize_and_crop(image) - mean) / std
input_data = np.ascontiguousarray(preprocess(image).astype(np.float32))
cuda.memcpy_htod(d_input, input_data)

# Build bindings list in index order
bindings = [None] * engine.num_bindings
bindings[input_binding_idx] = int(d_input)
bindings[output_binding_idx] = int(d_output)

# Run inference
context.execute_v2(bindings=bindings)

# Get output
cuda.memcpy_dtoh(h_output, d_output)

# softmax: e.g., scipy.special.softmax or np.exp(x) / np.sum(np.exp(x))
predictions = np.exp(h_output) / np.sum(np.exp(h_output), axis=-1, keepdims=True)
```

> **Why binding resolution matters:** If your model has multiple inputs (e.g., `input_ids` + `attention_mask`) or multiple outputs, assuming input=binding 0 and output=binding 1 will silently produce wrong results. Always resolve by name using `engine.get_binding_index("name")`.

> **Common beginner trap:** The input and output tensor names in your Triton `config.pbtxt` must exactly match the binding names in the engine. You cannot rename them in the config — they are baked into the engine at build time. To find the names, use: `trtexec --loadEngine=model.plan --verbose 2>&1 | grep "Binding"`.

**Note:** The input to a TensorRT INT8 engine is typically **float32**. TensorRT handles the quantization internally — the first operation in the engine quantizes the input, all internal operations run in INT8, and the last operation dequantizes the output. You do not need to manually quantize/dequantize.

---

## TensorRT-LLM — Quantization for Large Language Models

Standard TensorRT was designed for CNN-style models: fixed input/output shapes, relatively small models, batch inference. LLMs are a fundamentally different workload.

---

### Why Standard TensorRT Is Not Enough for LLMs

LLMs have properties that break the standard TensorRT workflow:

1. **Autoregressive generation** — the model runs token-by-token, with each forward pass producing one token. The output of step $t$ is the input to step $t+1$. Standard TensorRT expects a single forward pass with fixed shapes.

2. **KV cache** — past key-value pairs must be stored and grown across generation steps. This is a large, dynamically growing memory allocation that standard TensorRT does not manage.

3. **Variable sequence lengths** — different requests have different prompt lengths and generation lengths. Batching requests with different lengths requires padding or more sophisticated scheduling.

4. **Model size** — LLMs are 7B to 70B+ parameters. At FP16, a 70B model requires 140GB of GPU memory — more than a single GPU. Tensor parallelism (splitting the model across GPUs) is required but standard TensorRT does not support it.

5. **Memory bandwidth bottleneck** — autoregressive generation is memory-bound (each step reads the entire weight matrix for a tiny compute), making weight quantization extremely impactful for throughput.

TensorRT-LLM addresses all of these.

---

### TensorRT-LLM Architecture

TensorRT-LLM is a Python library that:
1. Defines the model architecture using TensorRT primitives (not PyTorch)
2. Applies quantization and compilation
3. Manages the KV cache, batching, and multi-GPU execution at runtime

```
PyTorch checkpoint (HuggingFace format)
        │
        ▼
TensorRT-LLM model definition
        │  (Python API that builds TRT network)
        ▼
Quantization (weight-only, SmoothQuant, FP8, etc.)
        │
        ▼
TensorRT engine build
        │
        ▼
TensorRT-LLM runtime
   • KV cache manager
   • In-flight batching
   • Tensor parallelism
   • Paged attention
```

The key difference from standard TensorRT: TensorRT-LLM is **model-aware**. It knows about transformer architectures, attention mechanisms, and autoregressive generation. Standard TensorRT treats the model as an opaque graph.

**Multi-GPU support:** TensorRT-LLM supports tensor parallelism (splitting layers across GPUs) and pipeline parallelism (splitting stages across GPUs) for models too large for a single GPU. Communication uses NCCL over NVLink (within a node) or InfiniBand (across nodes). Key topology pitfalls:
- Tensor parallelism degree must divide the number of attention heads evenly
- NVLink provides 900 GB/s (H100); PCIe provides ~64 GB/s — TP across PCIe-only connections is 10–15× slower
- Pipeline parallelism adds latency (pipeline bubbles) but reduces per-GPU memory; useful when TP alone can't fit the model

---

### Weight-Only Quantization in TensorRT-LLM

For LLMs, **weight-only quantization** (W4A16 or W8A16) is often more effective than full INT8 quantization (W8A8). Why? Because autoregressive generation is memory-bandwidth-bound: you read the entire weight matrix every token but only compute a small number of operations (batch size 1 during generation).

Reducing weight precision from FP16 to INT4 cuts memory bandwidth by 4×, which directly translates to ~3–4× throughput improvement for generation.

**W4A16:** Weights are INT4 (4-bit), activations stay in FP16. The INT4 weights are dequantized to FP16 on-the-fly during matrix multiplication. The dequantization is fused into the GEMM kernel, so there is no overhead.

**W8A16:** Weights are INT8, activations FP16. Less aggressive, better accuracy, ~2× memory bandwidth reduction.

```python
from tensorrt_llm import LLaMAForCausalLM
from tensorrt_llm.quantization import QuantMode

# Build LLaMA with W4A16 quantization
quant_mode = QuantMode.use_weight_only(use_int4_weights=True)

model = LLaMAForCausalLM.from_hugging_face(
    "meta-llama/Llama-2-7b-hf",
    quant_mode=quant_mode
)
engine = model.build(max_batch_size=8, max_input_len=2048, max_output_len=512)
engine.save("llama2_7b_w4a16.plan")
```

---

### SmoothQuant Integration

TensorRT-LLM supports SmoothQuant (Chapter 15) as a built-in quantization option. SmoothQuant enables W8A8 quantization (both weights and activations in INT8) for transformer models, which provides compute speedup in addition to memory bandwidth savings.

```python
quant_mode = QuantMode.use_smooth_quant(per_token=True, per_channel=True)

model = LLaMAForCausalLM.from_hugging_face(
    "meta-llama/Llama-2-7b-hf",
    quant_mode=quant_mode,
    smoothquant_val=0.5  # alpha parameter from SmoothQuant
)
```

The `smoothquant_val` (α) controls the migration strength — how aggressively to shift quantization difficulty from activations to weights. 0.5 is a common default. Higher values (0.7–0.9) shift more difficulty to weights, which helps when activation outliers are extreme.

---

### GPTQ and AWQ Support

TensorRT-LLM can consume models that were pre-quantized with GPTQ or AWQ (Chapter 17):

```python
# Load GPTQ-quantized model
model = LLaMAForCausalLM.from_hugging_face(
    "TheBloke/Llama-2-7B-GPTQ",
    quant_mode=QuantMode.from_description(
        quantize_weights=True,
        quantize_activations=False,
        per_group=True,
        use_int4_weights=True
    )
)

# Load AWQ-quantized model
model = LLaMAForCausalLM.from_hugging_face(
    "TheBloke/Llama-2-7B-AWQ",
    quant_mode=QuantMode.from_description(
        quantize_weights=True,
        quantize_activations=False,
        per_group=True,
        use_int4_weights=True
    )
)
```

GPTQ and AWQ pre-quantized models avoid the need to run calibration in TensorRT-LLM — the quantization parameters are already computed and stored in the checkpoint.

---

### FP8 Quantization in TensorRT-LLM

On Hopper GPUs (H100/H200), FP8 quantization offers the best ease-of-use vs. performance trade-off:

```python
quant_mode = QuantMode.from_description(
    quantize_weights=True,
    quantize_activations=True,
    per_token=True,
    per_channel=True,
    use_fp8=True
)

model = LLaMAForCausalLM.from_hugging_face(
    "meta-llama/Llama-2-7b-hf",
    quant_mode=quant_mode
)
```

**FP8 vs INT8 on H100:**
- FP8: same throughput as INT8 on Tensor Cores
- FP8: much simpler calibration (just need scaling factors, no zero-points, no asymmetric encoding)
- FP8: better accuracy for most models (floating-point format handles dynamic range better)
- FP8: E4M3 for forward pass, E5M2 for backward if training

For new deployments on Hopper+, FP8 is usually the right default choice.

---

### KV Cache Quantization

The KV cache stores past key-value pairs for all layers, all heads, across all tokens in the sequence. For long-context models (32K–128K tokens), the KV cache can consume more memory than the model weights.

TensorRT-LLM supports KV cache quantization to INT8 or FP8:

```python
# INT8 KV cache
quant_mode = QuantMode.from_description(
    quantize_weights=True,
    quantize_activations=True,
    use_fp8=True,
    quantize_kv_cache=True,       # Quantize KV cache to INT8
)
```

**Impact:** Quantizing the KV cache to INT8 cuts its memory by 2× (from FP16), allowing either longer sequences or larger batch sizes. The accuracy impact is typically minimal because the KV cache values are read back and dequantized before the attention computation.

> **Connecting to Chapter 18:** The KV cache is the "second memory wall" for LLMs. As context length grows, KV cache size exceeds weight size, and weight quantization alone is not enough. KV cache quantization (FP8 or NVFP4) pushes the "crossover point" further, extending the batch size and sequence length you can serve before hitting the memory wall.

### Paged KV-Caching: Solving Memory Fragmentation

Quantization reduces the *size* of each KV entry, but it doesn't solve *fragmentation*. Without paging, TensorRT-LLM pre-allocates a contiguous block of memory for each request's maximum sequence length. If requests have different lengths, the shorter ones waste their pre-allocated tails — and you run out of memory even when total KV usage is well below capacity.

**PagedAttention** (virtual memory for GPUs) solves this by allocating KV cache in fixed-size blocks (pages) rather than contiguous per-request arrays. As a request generates more tokens, new pages are allocated on demand. When a request finishes, its pages are returned to the pool instantly.

```
Without paging:                         With paging:
┌──────────────────────────┐            ┌───┬───┬───┬───┬───┬───┐
│ Request A: 2048 tokens   │            │ A │ A │ B │ C │ A │ B │ ← pages
│     (1500 wasted)        │            └───┴───┴───┴───┴───┴───┘
├──────────────────────────┤              No waste: pages freed
│ Request B: 2048 tokens   │              as requests complete.
│     (800 wasted)         │              New requests reuse pages.
└──────────────────────────┘
```

**How to enable paged KV in TRT-LLM:**

```python
# In trtllm-build:
trtllm-build \
    --checkpoint_dir ./ckpt \
    --output_dir ./engine \
    --paged_kv_cache enable \
    --tokens_per_block 64 \
    --max_num_tokens 8192
```

**Diagnostic: are you KV-bound or compute-bound?**

If throughput doesn't increase when moving from FP16 to INT8 weights, you are likely bottlenecked by KV-cache memory movement, not weight computation. To check:

```bash
# Compare FP16 vs INT8 weight-only at same context length:
# If throughput improvement is <10%, KV cache is the bottleneck.
# Fix: enable KV cache quantization (--kv_cache_dtype fp8 or int8)
# Fix: enable paged KV cache (--paged_kv_cache enable)
# Fix: reduce max sequence length if your workload allows it
```

> **The principle:** Quantization (FP8/INT8) reduces the *size* of each KV entry. PagedAttention prevents *fragmentation* of KV memory. You usually need both for long-context serving.

---

### TRT-LLM Quantization Recipe Matrix

TensorRT-LLM's quantization feature set is now broad. Here is the complete recipe matrix — the deployment-side answer to Chapters 15–19:

| Recipe | Weights | Activations | KV Cache | Best GPU | Use Case |
|--------|---------|-------------|----------|----------|----------|
| **FP16 (baseline)** | FP16 | FP16 | FP16 | Any | Reference, small models |
| **W8A16** | INT8 | FP16 | FP16 | Turing+ | Conservative memory reduction |
| **W4A16 GPTQ** | INT4 (grouped) | FP16 | FP16 | Turing+ | Memory-bound decode, pre-quantized models |
| **W4A16 AWQ** | INT4 (grouped) | FP16 | FP16 | Turing+ | Memory-bound decode, salient-weight-aware |
| **W8A8 SmoothQuant** | INT8 | INT8 | FP16 | Ampere+ | Compute-bound prefill, balanced workloads |
| **W4A8** | INT4 | INT8 | FP16 | Ampere+ | Aggressive compression + compute speedup |
| **FP8 (per-tensor)** | FP8 E4M3 | FP8 E4M3 | FP16 | Hopper+ | Best ease-of-use on Hopper, strong accuracy |
| **FP8 (block scaling)** | FP8 E4M3 | FP8 E4M3 | FP8 | Hopper+ | Row-wise/block FP8 for better accuracy at scale |
| **FP8 + FP8 KV** | FP8 E4M3 | FP8 E4M3 | FP8 E4M3 | Hopper+ | Maximum throughput on Hopper with long contexts |
| **FP4 / NVFP4** | FP4 E2M1 | FP8/FP16 | NVFP4 | Blackwell | Maximum compression on Blackwell |
| **FP4 + NVFP4 KV** | FP4 E2M1 | FP8 | NVFP4 | Blackwell | Blackwell with long-context KV cache reduction |

**How to choose:**
- **Memory-bound decode (batch=1, long context):** W4A16 AWQ or GPTQ — maximum memory bandwidth reduction
- **Compute-bound prefill (large batch):** W8A8 SmoothQuant — activations also quantized for compute speedup
- **Hopper+ (new deployments):** FP8 — best accuracy/performance/simplicity trade-off
- **Blackwell:** FP4/NVFP4 — maximum compression with hardware acceleration
- **Long-context KV pressure:** Add KV cache FP8 or NVFP4 to any recipe above

### LLM Recipe by Workload Phase (Advanced)

LLM inference has three distinct phases, and each has a different bottleneck. Map each phase to the right recipe:

| Phase | Bottleneck | Best Recipe | Why |
|-------|-----------|-------------|-----|
| **Prefill** (processing the prompt) | Compute-bound (large matrix multiplies) | W8A8 SmoothQuant or FP8 | Both weights AND activations are quantized → Tensor Core speedup matters here |
| **Decode** (generating tokens one by one) | Memory-bandwidth-bound (read entire weight matrix per token) | W4A16 AWQ or GPTQ | Only weight bandwidth matters; activation quantization adds complexity without helping |
| **Long-context** (>8K tokens) | KV-cache memory | Add KV cache FP8/NVFP4 to any recipe above | Weight recipe alone isn't enough; KV cache dominates memory at long contexts |

> **Practical consequence:** For a production LLM serving mixed workloads (some short prompts, some long), you may want FP8 (good for both prefill and decode on Hopper) plus FP8 KV cache. If you're on Ampere and serving batch-1 chatbot traffic, W4A16 AWQ + INT8 KV cache is the sweet spot.

### Recipe Selection as a Function (Advanced)

For teams that want a deterministic decision process:

```
INPUT:
  gpu_generation:     Turing | Ampere | Hopper | Blackwell
  batch_regime:       single (1-4) | medium (8-32) | large (64+)
  context_length:     short (<2K) | medium (2K-8K) | long (8K-128K)
  accuracy_tolerance: strict (<0.1 ppl) | moderate (<0.5 ppl) | relaxed (<1.0 ppl)

OUTPUT → recommended_recipe, fallback_recipe:

  Hopper + any batch + strict:       FP8,               fallback: FP16
  Hopper + any batch + moderate:     FP8 + FP8 KV,      fallback: FP8
  Hopper + single + relaxed:         W4A16 AWQ + FP8 KV, fallback: FP8

  Ampere + single + any:             W4A16 AWQ,          fallback: W8A16
  Ampere + large + moderate:         W8A8 SmoothQuant,   fallback: W8A16
  Ampere + any + long:               W4A16 AWQ + INT8 KV, fallback: W8A16 + INT8 KV

  Blackwell + any + any:             NVFP4 + NVFP4 KV,  fallback: FP8
  Turing + any + any:                W4A16 GPTQ,         fallback: W8A16
```

> **Rule:** Always start with the recommended recipe. If accuracy doesn't meet your threshold, move to the fallback. If the fallback is still insufficient, investigate SmoothQuant alpha tuning or QAT.

**TRT-LLM quantization scripts:**
```bash
# FP8 quantization (Hopper+)
python quantize.py --model_dir meta-llama/Llama-2-7b-hf \
    --output_dir ./llama2_fp8 --dtype float16 --qformat fp8

# INT4 AWQ
python quantize.py --model_dir meta-llama/Llama-2-7b-hf \
    --output_dir ./llama2_awq --dtype float16 --qformat int4_awq

# W8A8 SmoothQuant
python quantize.py --model_dir meta-llama/Llama-2-7b-hf \
    --output_dir ./llama2_sq --dtype float16 --qformat w8a8_sq \
    --smoothquant 0.5

# FP8 with FP8 KV cache
python quantize.py --model_dir meta-llama/Llama-2-7b-hf \
    --output_dir ./llama2_fp8_kv --dtype float16 --qformat fp8 \
    --kv_cache_dtype fp8
```

---

### TensorRT-LLM Practical Walkthrough

Complete example: deploying LLaMA 2 7B with INT4 weights on an A100:

> **⚠ HuggingFace Gating Prerequisite:** LLaMA models require HuggingFace token + Meta model access approval. Without this, the download will fail silently or return a 401 error.
> ```bash
> # 1. Request access at https://huggingface.co/meta-llama/Llama-2-7b-hf
> #    (requires a HuggingFace account and Meta approval — typically ~1 hour)
>
> # 2. Create a HuggingFace token at https://huggingface.co/settings/tokens
>
> # 3. Set the token in your environment (inside the container):
> export HF_TOKEN="hf_your_token_here"
> # Or: huggingface-cli login
> ```

```bash
# Available in: NGC tritonserver:*-trtllm-* container

# 1. Install TensorRT-LLM (if not using the NGC container)
pip install tensorrt_llm

# 2. Convert HuggingFace checkpoint to TRT-LLM format
python convert_checkpoint.py \
    --model_dir meta-llama/Llama-2-7b-hf \
    --output_dir ./llama2_7b_ckpt \
    --dtype float16 \
    --use_weight_only \
    --weight_only_precision int4

# 3. Build TensorRT engine
trtllm-build \
    --checkpoint_dir ./llama2_7b_ckpt \
    --output_dir ./llama2_7b_engine \
    --max_batch_size 8 \
    --max_input_len 2048 \
    --max_output_len 512 \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16

# 4. Run inference
python run.py \
    --engine_dir ./llama2_7b_engine \
    --tokenizer_dir meta-llama/Llama-2-7b-hf \
    --max_output_len 100 \
    --input_text "Explain quantization in simple terms:"
```

**Illustrative results on A100 (80GB):**

> **These are example numbers from a specific measurement setup, not guaranteed baselines.** Your results will vary based on model variant, batch size, sequence length, TRT-LLM version, and GPU SKU. Always measure on your own hardware and configuration using `trtllm-build` + the TRT-LLM benchmarking scripts.

| Quantization | Model Size | Generation Speed | Accuracy (perplexity) |
|-------------|-----------|-----------------|----------------------|
| FP16 | 14 GB | ~40 tok/s | 5.47 (baseline) |
| W8A16 | 7 GB | ~75 tok/s | 5.49 |
| W4A16 (GPTQ) | 3.5 GB | ~110 tok/s | 5.63 |
| W8A8 (SmoothQuant) | 7 GB | ~90 tok/s | 5.54 |
| FP8 (H100 only) | 7 GB | ~120 tok/s | 5.48 |

The W4A16 configuration fits the entire model in ~3.5GB, leaving the rest of the A100's 80GB for KV cache — enabling large batch sizes or very long contexts.

---

## NVIDIA TensorRT Model Optimizer (ModelOpt)

ModelOpt is NVIDIA's training-side quantization and compression toolkit — the NVIDIA equivalent of Qualcomm's AIMET. It runs in PyTorch and provides quantization, sparsity, and distillation techniques that produce models ready for TensorRT compilation.

---

### What ModelOpt Does

ModelOpt provides:

1. **Post-Training Quantization (PTQ)** — calibrate and quantize a pre-trained model without retraining.
2. **Quantization-Aware Training (QAT)** — fine-tune a model with simulated quantization noise.
3. **Sparsity** — prune model weights to 2:4 structured sparsity for Ampere+ Tensor Cores.
4. **Knowledge distillation** — train a smaller/quantized model using a larger teacher model.
5. **Export** — produce ONNX models with explicit quantization nodes that TensorRT can consume directly.

---

### Post-Training Quantization with ModelOpt

```python
import modelopt.torch.quantization as mtq

# Define quantization config
quant_cfg = mtq.INT8_DEFAULT_CFG  # INT8 weights + INT8 activations

# Or for weight-only:
# quant_cfg = mtq.W4A16_AWQ_CFG  # INT4 weights, FP16 activations

# Quantize model
model = mtq.quantize(model, quant_cfg, forward_loop=calibration_fn)

# calibration_fn runs the calibration dataset through the model:
def calibration_fn(model):
    for batch in calibration_loader:
        model(batch)
```

ModelOpt automatically:
- Inserts quantization nodes around all compatible layers
- Runs the calibration function to observe activation ranges
- Computes optimal scales using entropy or MinMax calibration
- Freezes the quantization parameters

**Supported configurations:**

| Config | Weights | Activations | Use Case |
|--------|---------|-------------|----------|
| `INT8_DEFAULT_CFG` | INT8 | INT8 | CNNs, standard models |
| `FP8_DEFAULT_CFG` | FP8 E4M3 | FP8 E4M3 | Hopper GPUs |
| `W4A16_AWQ_CFG` | INT4 (grouped) | FP16 | LLMs, weight-only |
| `W8A8_SMOOTHQUANT_CFG` | INT8 | INT8 | LLMs with SmoothQuant |
| `INT4_AWQ_CFG` | INT4 | INT4 | Aggressive compression |

---

### Quantization-Aware Training with ModelOpt

When PTQ accuracy is not acceptable:

```python
import modelopt.torch.quantization as mtq

# Step 1: Quantize (PTQ first)
model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=calibration_fn)

# Step 2: Fine-tune (QAT)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
for epoch in range(10):
    for batch in train_loader:
        output = model(batch['input'])
        loss = criterion(output, batch['target'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Step 3: Export to ONNX with quantization nodes
mtq.export(model, "model_int8_qat.onnx", dummy_input)
```

The exported ONNX contains explicit QuantizeLinear/DequantizeLinear nodes. TensorRT recognizes these and uses the pre-computed scales instead of running its own calibration.

---

### Sparsity Support

Ampere and later GPUs support **2:4 structured sparsity**: out of every 4 consecutive weight values, exactly 2 must be zero. The Tensor Core hardware can skip the zero multiplications, achieving up to 2× speedup.

ModelOpt provides sparsity tools:

> **⚠ Fine-tuning is not optional.** Unlike quantization (where PTQ alone often works for CNNs), 2:4 sparsity almost always requires fine-tuning to recover accuracy. Zeroing out 50% of weights is a destructive operation — without retraining, expect 2–5% accuracy loss on CNNs and significant degradation on LLMs. Budget 5–20 epochs of fine-tuning (or 1–2% of original training compute) as a rule of thumb.

```python
import modelopt.torch.sparsity as mts

# Apply 2:4 structured sparsity (this zeros out 2 of every 4 weights)
model = mts.sparsify(model, mts.SPARSITY_2_4_CFG, forward_loop=calibration_fn)

# Fine-tune to recover accuracy — THIS STEP IS MANDATORY
# Without it, accuracy will degrade significantly
for epoch in range(5):  # 5-20 epochs typical; monitor val loss
    train_one_epoch(model, train_loader)
    val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch}: val_acc={val_acc:.2f}%")  # Should approach dense baseline

# Export
mts.export(model, "model_sparse.onnx", dummy_input)
```

**Sparsity + Quantization together:** You can combine 2:4 sparsity with INT8 quantization. The model is both sparse and quantized — the Tensor Core computes on the non-zero INT8 values, achieving both the sparsity speedup and the quantization speedup.

```python
# First sparsify
model = mts.sparsify(model, mts.SPARSITY_2_4_CFG, forward_loop=calibration_fn)
# Then quantize
model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=calibration_fn)
# Fine-tune
fine_tune(model)
# Export
mtq.export(model, "model_sparse_int8.onnx", dummy_input)
```

---

### ModelOpt Practical Walkthrough

Complete pipeline: ResNet-50 from PyTorch to quantized ONNX:

```python
import torch
import torchvision
import modelopt.torch.quantization as mtq

# 1. Load model
model = torchvision.models.resnet50(pretrained=True).cuda().eval()

# 2. Define calibration function
def calibrate(model):
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= 100:  # 100 batches of calibration
                break
            model(images.cuda())

# 3. PTQ with INT8
model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=calibrate)

# 4. Evaluate
accuracy = evaluate(model, val_loader)
print(f"INT8 PTQ accuracy: {accuracy:.2f}%")

# 5. Export to ONNX
dummy = torch.randn(1, 3, 224, 224).cuda()
mtq.export(model, "resnet50_int8.onnx", dummy)

# 6. Build TensorRT engine
# (Use TensorRT builder as shown above, with the quantized ONNX)
```

---

### ModelOpt for LLM Quantization

ModelOpt is not just for CNNs — it is NVIDIA's recommended tool for LLM post-training quantization, and TensorRT-LLM's quantization scripts directly reference ModelOpt. ModelOpt supports advanced PTQ algorithms specifically designed for transformer architectures:

| Algorithm | Config | What It Does | When to Use |
|-----------|--------|-------------|-------------|
| **SmoothQuant** | `W8A8_SMOOTHQUANT_CFG` | Migrates outliers from activations to weights (Chapter 15) | W8A8 quantization for compute-bound LLM workloads |
| **AWQ** | `W4A16_AWQ_CFG` | Protects salient weight channels (Chapter 17) | W4A16 for memory-bound LLM decode |
| **FP8** | `FP8_DEFAULT_CFG` | Per-tensor FP8 E4M3 scaling | Hopper+ deployments with best accuracy |
| **Block-wise INT4** | `INT4_BLOCKWISE_CFG` | Per-group INT4 with fine-grained scales | Aggressive LLM compression |

**The ModelOpt → TRT-LLM pipeline:**
```
ModelOpt quantize (PyTorch) → export checkpoint → trtllm-build → TRT-LLM engine
```

ModelOpt produces quantized checkpoints (or Q/DQ ONNX); TensorRT/TRT-LLM consumes them. Think of ModelOpt as "fake quant in PyTorch; speedup happens after export."

```python
# ModelOpt PTQ for an LLM (minimal example)
import modelopt.torch.quantization as mtq
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = model.cuda().eval()

# Calibrate with representative text
def calibrate(model):
    for batch in calibration_dataloader:
        model(batch["input_ids"].cuda())

# Quantize to FP8
model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop=calibrate)

# Export for TRT-LLM consumption
mtq.export(model, "llama2_7b_fp8_checkpoint")
```

> **ModelOpt vs. TensorRT-native calibration:** ModelOpt is required for sparsity (2:4), distillation-aware quantization, and advanced LLM PTQ algorithms (SmoothQuant, AWQ). Native TensorRT calibration (implicit) is sufficient only for basic CNN PTQ and is deprecated. For any new project — especially transformers — start with ModelOpt.

---

### Torch-TensorRT: The PyTorch-First Alternative Path

Not everyone wants to deal with ONNX export. **Torch-TensorRT** is an integration that compiles PyTorch models directly to TensorRT engines through `torch.compile()`:

```python
import torch
import torch_tensorrt

model = MyModel().cuda().eval()

# Compile with INT8 quantization (via ModelOpt)
optimized = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(shape=[1, 3, 224, 224], dtype=torch.float32)],
    enabled_precisions={torch.float16, torch.int8},
)

# Use like a normal PyTorch model
output = optimized(input_tensor)
```

Torch-TensorRT supports PTQ for INT8, FP8, and FP4 via ModelOpt. It is the recommended path for teams that want TensorRT performance without leaving the PyTorch ecosystem. The trade-off: less control over the engine build process compared to the ONNX path.

---

# ── ADVANCED TIER ──

> **Advanced Goals:** Predictable performance + wide model coverage. After this tier, you can explain where time goes (layer profile + Nsight), eliminate reformat overhead where possible, and handle unsupported ops via rewrite/surgery/plugin decision tree.
>
> **Skip guidance:** If you already have predictable performance and need multi-GPU serving, SLO tuning, or fleet-scale governance, jump to **Expert**.

## cuDNN and cuBLAS: The Kernel Layer

Underneath TensorRT, the actual computation happens in **cuDNN** (CUDA Deep Neural Network library, for convolutions and normalization) and **cuBLAS** (CUDA Basic Linear Algebra Subprograms, for matrix multiplications). These libraries provide the GPU kernels — the hand-optimized assembly code that runs on Tensor Cores.

---

### How Quantized Kernels Are Selected (Tactic Selection)

When TensorRT builds an engine, it calls cuDNN and cuBLAS to get a list of available kernels for each operation. Each library provides multiple kernel implementations for the same operation, optimized for different:
- Matrix dimensions
- Tensor memory layouts (row-major, column-major, blocked)
- Precision (FP32, FP16, INT8, FP8)
- Fusion patterns (e.g., Conv+ReLU vs Conv alone)

TensorRT benchmarks each kernel variant on the target GPU and selects the fastest. This is the "kernel auto-tuning" step that makes engine building slow but inference fast.

**What happens during tactic selection (in detail):**

For each layer in the network, TensorRT:

1. **Enumerates all candidate tactics** — queries cuDNN/cuBLAS for every kernel that can compute this layer with the requested precision and input/output formats. A single Conv layer might have 20–50 candidate tactics.

2. **Benchmarks each tactic** — runs each kernel multiple times with representative data and measures execution time. This is why engine builds are slow: a model with 100 layers × 30 tactics = 3,000 micro-benchmarks.

3. **Considers format propagation** — a tactic might be fast in isolation, but if it requires a different tensor layout (e.g., NHWC→NCHW conversion) than the next layer, the reformat overhead can negate the kernel speedup. TensorRT solves this as a global graph optimization, not a greedy per-layer decision.

4. **Selects the fastest end-to-end combination** — the final engine uses the tactic combination that minimizes total inference time, including any necessary reformat operations between layers.

**Why this matters for quantization:** INT8 and FP8 tactics operate on different data layouts than FP16 tactics. A mixed-precision engine (some layers INT8, some FP16) may spend significant time on reformat operations between layers. This is why the `--dumpProfile` output sometimes shows high "Reformat" time — it's the cost of crossing precision boundaries.

**Tactic timing cache:** Because tactic selection is expensive, TensorRT can cache the results. On subsequent builds (same GPU, same TensorRT version), the cache eliminates re-benchmarking:

```bash
# First build: slow (benchmarks all tactics)
trtexec --onnx=model.onnx --int8 --fp16 --timingCacheFile=timing.cache --saveEngine=model.plan

# Second build: fast (reuses cached tactic timings)
trtexec --onnx=model_v2.onnx --int8 --fp16 --timingCacheFile=timing.cache --saveEngine=model_v2.plan
```

For INT8:
- **cuDNN** provides fused Conv+Bias+ReLU INT8 kernels with INT32 accumulation
- **cuBLAS** provides INT8 GEMM kernels (IMMA instructions on Turing+)

---

### IMMA Instructions for INT8

**IMMA** (Integer Matrix Multiply-Accumulate) is the Tensor Core instruction set for INT8 computation. On Turing and later:

$$C_{m \times n}^{int32} += A_{m \times k}^{int8} \times B_{k \times n}^{int8}$$

The Tensor Core processes a 8×8×32 tile in a single cycle:
- Reads 8×32 = 256 INT8 values from matrix A
- Reads 32×8 = 256 INT8 values from matrix B
- Computes 8×8 = 64 INT32 dot products (each dot product sums 32 INT8×INT8 products)
- Accumulates into 8×8 = 64 INT32 values

**Alignment requirements:** For maximum Tensor Core utilization:
- Matrix dimensions should be multiples of 16 (ideally 128 for best occupancy)
- Data must be in specific layouts (column-major for A, row-major for B, or their blocked equivalents)

TensorRT and cuBLAS handle these alignment requirements automatically, padding when necessary.

---

### FP8 GEMM in cuBLAS

On Hopper GPUs, cuBLAS provides FP8 GEMM kernels:

$$C_{fp16} = \text{scale}_A \times A_{fp8} \times \text{scale}_B \times B_{fp8}$$

Each input tensor has a per-tensor scaling factor. The multiplication is performed in FP8 on Tensor Cores, and the result is accumulated in FP32 (inside the Tensor Core), then converted to FP16 or BF16 for output.

**cuBLAS FP8 API:**
```cpp
cublasLtMatmul(
    handle,
    matmulDesc,    // Specifies FP8 compute type
    &alpha,
    A, Adesc,      // FP8 E4M3 matrix A with scale
    B, Bdesc,      // FP8 E4M3 matrix B with scale
    &beta,
    C, Cdesc,      // FP16 output
    D, Ddesc,      // FP16 output (can be same as C)
    ...
);
```

The scaling factors are critical for FP8 — without proper scaling, FP8's limited range (±448 for E4M3) causes overflow or underflow. NVIDIA's Transformer Engine manages these scales automatically during training and inference.

---

## Triton Inference Server — Serving Quantized Models

Triton is NVIDIA's model serving infrastructure. It is not a quantization tool — it is what runs your quantized TensorRT engine in production, handling HTTP/gRPC requests, batching, multi-model management, and scaling.

---

### What Triton Does

Triton sits between your application (sending inference requests) and the GPU (running the model):

```
Client (HTTP/gRPC request)
        │
        ▼
Triton Inference Server
   ├── Request queue
   ├── Dynamic batcher (groups requests into batches)
   ├── Scheduler (routes to correct model/GPU)
   ├── Model executor (runs TensorRT engine)
   └── Response handler
        │
        ▼
Client (receives prediction)
```

Why does this matter for quantization? Because quantized models change the throughput/latency trade-offs, and Triton's batching configuration must be tuned accordingly.

---

### Loading TensorRT Engines in Triton

Triton loads TensorRT engines from a **model repository** — a directory with a specific structure. Triton discovers models by scanning this directory at startup. Here is a complete example showing multiple models (CNN + LLM) in a single repository:

```
model_repository/                         ← Root (passed via --model-repository)
├── resnet50_int8/                        ← CNN model (TensorRT plan)
│   ├── config.pbtxt                      ← Model configuration
│   └── 1/                               ← Version 1
│       └── model.plan                    ← Your TensorRT engine file
├── yolov8_fp16/                          ← Another CNN model
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan
├── llama2_7b_fp8/                        ← LLM model (TRT-LLM backend)
│   ├── config.pbtxt
│   └── 1/                               ← Version directory (engine path in config)
│       └── (empty — engine_dir set in config.pbtxt parameters)
└── resnet50_with_plugin/                 ← Model with custom plugin
    ├── config.pbtxt
    ├── 1/
    │   └── model.plan
    └── plugins/
        └── libmycustomplugin.so          ← Custom plugin shared library
```

**Key rules for the model repository:**
1. Each model lives in its own directory, named exactly as it will be addressed in API calls
2. The version subdirectory (`1/`, `2/`, etc.) allows serving multiple versions simultaneously
3. The engine file must be named `model.plan` (for `tensorrt_plan` platform) or the name specified in `config.pbtxt`
4. Input/output tensor names in `config.pbtxt` must exactly match the engine's binding names — you cannot rename them

**config.pbtxt:**
```protobuf
name: "resnet50_int8"
platform: "tensorrt_plan"
max_batch_size: 64

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 5000
}
```

Note: the input/output data types are FP32 even though the engine internally runs INT8. TensorRT handles the quantization/dequantization internally.

**Option 1: Auto-generate config (quick start)**

Triton can infer `config.pbtxt` directly from the engine file. Launch Triton with `--strict-model-config=false` and it will read the engine's binding names, data types, and dimensions automatically:

```bash
# No config.pbtxt needed — Triton reads the engine's metadata
tritonserver --model-repository=/models --strict-model-config=false
```

This is the fastest way to get a model serving. Triton generates a minimal config with the correct tensor names and types. However, the auto-generated config has **no dynamic batching, no preferred batch sizes, and no queue delay tuning** — it's a bare-minimum starting point.

**Option 2: Write config explicitly (production)**

For production, write `config.pbtxt` manually to control batching, instance count, and scheduling. Use the auto-generated config as a starting point, then add the settings you need.

**Triton minimal config rules (when writing manually):**
1. You must specify `platform` (e.g., `tensorrt_plan`) or `backend` (e.g., `tensorrtllm`)
2. You must specify `max_batch_size` (0 means batching disabled)
3. You must specify input/output tensor names, data types, and dims (must match the engine)
4. If using dynamic shapes, the TensorRT engine must have been built with optimization profiles

> **Practical workflow:** Start with `--strict-model-config=false` to let Triton auto-detect tensor names and types. Copy the auto-generated config from Triton's log output. Then add `dynamic_batching`, `instance_group`, and other production settings to that config.

---

### Dynamic Batching with Quantized Models

Quantized models are faster per-inference, which means the batcher needs to be tuned differently:

- **Without quantization (FP32):** Each inference takes 5ms. With a 5ms batching window, you accumulate ~N requests. Throughput: N/5ms.
- **With INT8 quantization:** Each inference takes 1.5ms. With the same 5ms window, you accumulate more requests. But if requests arrive slowly, you waste time waiting. Reduce the batching window to 2ms.

**Tuning the batcher for quantized models:**
```protobuf
dynamic_batching {
  preferred_batch_size: [ 16, 32, 64 ]  # Larger batches (faster per-item)
  max_queue_delay_microseconds: 2000     # Shorter wait (model is faster)
}
```

The general principle: quantized models can handle larger batch sizes at the same latency target, which improves throughput.

### Triton Correctness Gate (Intermediate)

Before declaring your Triton deployment ready, verify that the Triton endpoint produces the same results as your local TensorRT runtime. Triton adds batching, request scheduling, and network serialization — any of these can introduce subtle bugs:

```python
# Available in: any environment with tritonclient (pip install tritonclient[http])
import tritonclient.http as httpclient
import numpy as np

# 1. Get reference output from local TensorRT engine
# (use run_trt_engine() from the Engine Serialization section)
local_output = run_trt_engine("model_int8.plan", test_input)

# 2. Get output from Triton endpoint
client = httpclient.InferenceServerClient("localhost:8000")
inputs = [httpclient.InferInput("input", test_input.shape, "FP32")]
inputs[0].set_data_from_numpy(test_input.astype(np.float32))
triton_result = client.infer("resnet50_int8", inputs)
triton_output = triton_result.as_numpy("output")

# 3. Compare — should be identical (bitwise or very close)
max_diff = np.max(np.abs(local_output - triton_output))
assert max_diff < 1e-5, f"Triton output differs from local: max_diff={max_diff}"
# If this fails: check config.pbtxt tensor names, data types, and dims
```

> **If Triton output differs from local:** The #1 cause is a tensor name mismatch in `config.pbtxt` — see the Preprocessing Mismatch Checklist. The #2 cause is a shape mismatch (e.g., Triton adds a batch dimension that your config doesn't expect).

### Dynamic Shapes → Profiles → Triton (The Chain)

> **Boxed rule:** If any input dimension varies at serving time (batch size, sequence length, image resolution), the TensorRT engine MUST have optimization profiles. Triton will not invent profiles — it uses whatever the engine provides, and rejects inputs outside the min–max range.

The chain works like this:

```
1. ONNX model has dynamic dims    →  input: [batch, 3, 224, 224] where batch is dynamic
2. trtexec build with profiles     →  --minShapes=input:1x3x224x224
                                       --optShapes=input:16x3x224x224
                                       --maxShapes=input:64x3x224x224
3. Triton serves within that range →  max_batch_size: 64 in config.pbtxt
                                      (requests with batch > 64 will be rejected)
```

**What happens when profiles are missing:**
- If you build without `--minShapes/--optShapes/--maxShapes`, the engine has fixed shapes
- Triton will reject any request with a different shape
- Error message: `"unexpected shape for input 'input', model expects [1,3,224,224]"`

```bash
# Correct: build with dynamic batch profile
trtexec --onnx=model.onnx --saveEngine=model.plan --fp16 \
    --minShapes=input:1x3x224x224 \
    --optShapes=input:16x3x224x224 \
    --maxShapes=input:64x3x224x224

# Then in config.pbtxt:
# max_batch_size: 64
# dims: [ 3, 224, 224 ]   ← batch dim is handled by max_batch_size, not dims
```

---

### Triton for LLM Serving

The CNN-focused Triton config above does not apply to LLM serving. LLMs have fundamentally different serving patterns:

- **Streaming tokens** — LLMs generate tokens one at a time; clients need decoupled (streaming) responses, not a single batch response
- **In-flight batching** — different requests are at different generation steps; the TRT-LLM runtime handles batching internally, not Triton's dynamic batcher
- **Sequence state** — the KV cache must persist across generation steps within a request

**TRT-LLM Triton backend deployment:**

Triton serves TRT-LLM models via the `tensorrtllm` backend. The model repository structure differs from CNN models:

```
model_repository/
└── llama2_7b/
    ├── config.pbtxt
    └── 1/
        └── (empty — engine path specified in config)
```

```protobuf
# config.pbtxt for TRT-LLM model
name: "llama2_7b"
backend: "tensorrtllm"
max_batch_size: 8

model_transaction_policy {
  decoupled: true    # Enable streaming responses
}

parameters {
  key: "engine_dir"
  value: { string_value: "/engines/llama2_7b_fp8" }
}

parameters {
  key: "max_tokens_in_paged_kv_cache"
  value: { string_value: "8192" }
}

parameters {
  key: "batch_scheduler_policy"
  value: { string_value: "inflight_fused_batching" }
}
```

> **Key difference from CNN serving:** For LLMs, Triton delegates batching, KV cache management, and scheduling to the TRT-LLM runtime. Triton's role becomes request routing, HTTP/gRPC handling, and observability — not batch assembly.

---

### Triton Practical Setup

```bash
# 1. Start Triton server with your model repository
docker run --gpus=1 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.05-py3 \
    tritonserver --model-repository=/models
```

> **You must provide:** `preprocess()` — your model-specific preprocessing function (same normalization, resize, and channel order used during training). The tensor name `"input"` must exactly match the engine binding name. Install the client with `pip install tritonclient[http]`.

```python
# 2. Send inference request (Python client)
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient("localhost:8000")

# Prepare input — preprocess() must match your training pipeline
# Example: input_data = normalize(resize(load_image("cat.jpg")))
input_data = preprocess(image).astype(np.float32)

# "input" must match the engine's input binding name exactly
inputs = [httpclient.InferInput("input", input_data.shape, "FP32")]
inputs[0].set_data_from_numpy(input_data)

# Run inference — "resnet50_int8" must match the model directory name in model_repository/
result = client.infer("resnet50_int8", inputs)

# "output" must match the engine's output binding name exactly
output = result.as_numpy("output")

# Apply softmax to get probabilities
predictions = np.exp(output) / np.sum(np.exp(output), axis=-1, keepdims=True)
print(f"Top prediction: class {np.argmax(predictions)}, confidence {np.max(predictions):.3f}")
```

---

## End-to-End Pipeline: Training to Deployment

---

### Step 1: Train in PyTorch

Train as usual. No quantization considerations at this stage.

```python
model = MyModel()
train(model, train_dataset, epochs=100)
torch.save(model.state_dict(), "model_fp32.pth")
```

---

### Step 2: Export to ONNX

```python
dummy = torch.randn(1, 3, 224, 224).cuda()
torch.onnx.export(model, dummy, "model.onnx",
                  opset_version=17,
                  input_names=["input"],
                  output_names=["output"])
```

**Tip:** Use opset 17+ for best TensorRT compatibility. Check the TensorRT ONNX operator support matrix for your TensorRT version.

> **Standardize tensor names end-to-end.** Use `"input"` and `"output"` (or your model's actual names) consistently across ONNX export → `trtexec` → Triton `config.pbtxt` → client code. To confirm the engine's binding names after build:
> ```bash
> trtexec --loadEngine=model.plan --verbose 2>&1 | grep -i "binding"
> # Copy the exact names into your Triton config.pbtxt and client code
> ```

---

### Step 3: Calibrate and Build with TensorRT

**Option A: Let TensorRT calibrate (simple, good for CNNs):**
```python
config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = EntropyCalibrator(calibration_data)
engine = builder.build_serialized_network(network, config)
```

**Option B: Use ModelOpt for quantization, then build (better for transformers):**
```python
model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=calibrate)
mtq.export(model, "model_quantized.onnx", dummy)
# Then build TensorRT engine from the quantized ONNX (no calibration needed)
```

**Option C: For LLMs, use TensorRT-LLM directly:**
```bash
trtllm-build --checkpoint_dir ./ckpt --output_dir ./engine \
    --use_weight_only --weight_only_precision int4
```

---

### Step 4: Validate Accuracy — The Mandatory Correctness Gate

> **⚠ Do not proceed to Triton deployment until this gate passes.** A deployed engine with wrong outputs is worse than no engine at all. This mirrors the diagnostic discipline from Chapter 20: measure before you serve.

Validation has three mandatory levels, in order. Do not skip any:

**Level 1: ONNXRuntime (FP32) vs TensorRT (FP16)** — catches graph conversion errors:
```bash
# Available in: NGC tensorrt container (pip install polygraphy if missing)
polygraphy run model.onnx \
  --onnxrt --trt \
  --fp16 \
  --atol 1e-3 --rtol 1e-3 \
  --val-range input:[0,1]
# Expected: "PASSED" — if this fails, the ONNX → TRT conversion broke something
```

**Level 2: TensorRT FP16 vs TensorRT INT8** — catches quantization errors:
```bash
polygraphy run model.onnx \
  --trt --trt \
  --fp16 --int8 \
  --atol 5e-3 --rtol 5e-3 \
  --val-range input:[0,1]
# Expected: "PASSED" with small diffs — if this fails, quantization is hurting accuracy
```

**Level 3: Task metric evaluation** — catches cases where numerical diff is acceptable but task accuracy is not:
```bash
# This is model-specific — you must implement your own evaluation loop
# Example: top-1 accuracy for classification, mAP for detection, perplexity for LLMs
```

> **What "PASSED" and "FAILED" look like in Polygraphy output:**
> ```
> # PASSED example:
> [I] PASSED | Output: 'output' | max_absdiff=0.000412 | max_reldiff=0.001234 | median_absdiff=0.000031
>
> # FAILED example:
> [E] FAILED | Output: 'output' | max_absdiff=1.234567 | max_reldiff=0.987654 | median_absdiff=0.345678
> # → Investigate: which layers caused the diff? Use polygraphy debug reduce
> ```

**Tolerance selection rule — how to choose atol/rtol:**

| Model Type | Recommended atol | Recommended rtol | When numerical diff is NOT the right metric |
|-----------|-----------------|-----------------|---------------------------------------------|
| Classification (CNN) | 1e-3 | 1e-3 | When top-1 prediction changes but diff is small (check top-1 match instead) |
| Object Detection | 5e-3 | 5e-3 | When bounding boxes shift by >1 pixel — use mAP as the primary metric |
| LLM (logits) | 1e-2 | 1e-2 | LLM logits are high-dimensional; tiny diffs in logit space can flip tokens. Use perplexity or downstream task accuracy instead |
| Segmentation | 1e-3 | 1e-3 | When pixel-level accuracy matters more than raw activation diff |

> **Rule:** If Polygraphy passes at the numerical level but task accuracy drops, the issue is usually in the most sensitive layers (layer norm, attention softmax, final classification head). Use `polygraphy debug reduce` to isolate which layer causes the task-level regression.

If you need programmatic control (e.g., running task-level evaluation on a full dataset):

> **You must provide:** `model_fp32` — your PyTorch model for reference outputs. `run_trt_engine()` — a function that loads a TensorRT engine and runs inference (use the engine loading code from the "Serialization and Deployment" section above). `evaluate()` / `evaluate_trt()` — your task-specific accuracy evaluation functions (e.g., top-1 accuracy for classification, mAP for detection, perplexity for LLMs).

```python
# pseudocode — adapt to your evaluation pipeline
import numpy as np

# FP32 reference (PyTorch)
with torch.no_grad():
    fp32_output = model_fp32(test_input.cuda()).cpu().numpy()

# TensorRT INT8 (using the engine loading pattern from above)
trt_output = run_trt_engine("model_int8.plan", test_input.numpy())

# Numerical comparison
max_diff = np.max(np.abs(fp32_output - trt_output))
mean_diff = np.mean(np.abs(fp32_output - trt_output))
print(f"Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

# Task-level accuracy (you implement these based on your evaluation pipeline)
fp32_acc = evaluate(model_fp32, test_dataset)     # e.g., top-1 accuracy
int8_acc = evaluate_trt("model_int8.plan", test_dataset)
print(f"FP32: {fp32_acc:.2f}%, INT8: {int8_acc:.2f}%, Drop: {fp32_acc - int8_acc:.2f}%")
```

**Acceptable accuracy drop guidelines:**
- CNNs: <1% top-1 accuracy drop
- Object detection: <0.5 mAP drop
- LLMs: <0.5 perplexity points increase
- If drop exceeds these thresholds, suspect calibration data quality first (Chapter 9 — **Calibration Mismatch**, Chapter 13), then consider SmoothQuant for transformers (Chapter 15 — **Resolution Collapse**, Chapter 14), or fall back to FP16 for the affected layers

### Preprocessing Mismatch Checklist

> **Reference this list everywhere `preprocess()` is mentioned** — engine run, Triton client, troubleshooting. Preprocessing mismatch is the #1 cause of "the model works in PyTorch but gives wrong results in TensorRT."

Before debugging quantization accuracy, rule out preprocessing:

- [ ] **Same normalization:** mean and std must match training (e.g., ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- [ ] **Same channel order:** RGB vs BGR — OpenCV loads BGR by default; PyTorch `torchvision` expects RGB
- [ ] **Same resize/crop:** bilinear vs nearest, center crop vs letterbox, resize before crop vs after
- [ ] **Same value range:** [0, 1] vs [0, 255] — failing to divide by 255 is a common silent error
- [ ] **Same dtype:** input to TensorRT must be float32 (even for INT8 engines), not uint8
- [ ] **Same spatial order:** NCHW (PyTorch default) vs NHWC — ONNX export preserves PyTorch's NCHW; if your client sends NHWC, outputs will be wrong

---

### Step 5: Deploy with Triton

Set up the model repository, write config.pbtxt, launch Triton:

```bash
# Available in: NGC tritonserver container
tritonserver --model-repository=/models --log-verbose=1
```

> **Expected output (model loaded successfully):**
> ```
> I0101 00:00:00.000000 1 server.cc:592] 
> +------------------+---------+--------+
> | Model            | Version | Status |
> +------------------+---------+--------+
> | resnet50_int8    | 1       | READY  |
> +------------------+---------+--------+
> ...
> I0101 00:00:01.000000 1 grpc_server.cc:2451] Started GRPCInferenceService at 0.0.0.0:8001
> I0101 00:00:01.000000 1 http_server.cc:3558] Started HTTPService at 0.0.0.0:8000
> ```
> If you see `UNAVAILABLE` instead of `READY`, check tensor names in config.pbtxt match engine bindings.

### Intermediate Exit Criteria

**You have completed the Intermediate tier when all are true:**
- [ ] INT8 accuracy validated via Polygraphy diff gate (Level 1 + Level 2 pass)
- [ ] Triton serving works — model loads as `READY`, inference returns correct results
- [ ] Dynamic batching configured with appropriate queue delay for your quantized model
- [ ] Dynamic shapes supported with optimization profiles (if input dims vary)
- [ ] Triton correctness gate passes (local TRT vs Triton output match)

---

### Step 6: Profile with Nsight

NVIDIA provides Nsight Systems and Nsight Compute for profiling:

```bash
# Profile TensorRT engine execution
nsys profile --trace=cuda,nvtx \
    trtexec --loadEngine=model_int8.plan --batch=32 --iterations=100

# Analyze kernel-level performance
ncu --target-processes all --set full \
    trtexec --loadEngine=model_int8.plan --batch=1
```

`trtexec` is TensorRT's built-in benchmarking tool:
```bash
# Quick benchmark
trtexec --onnx=model.onnx --int8 --fp16 --workspace=4096 \
    --calib=calibration.cache --saveEngine=model_int8.plan

# Output: throughput (inferences/sec), latency (min/max/mean/p99)
```

---

### Golden Commands (Copy-Paste Reference)

The most common TensorRT operations in one place. All commands assume you are inside an NGC TensorRT container. For detailed explanation of each command, flags, and expected output, see "trtexec: The Canonical Build and Benchmark Tool" below.

```bash
# ── FP16 engine build ──
trtexec --onnx=model.onnx --saveEngine=model_fp16.plan --fp16

# ── INT8 engine build with calibration cache ──
trtexec --onnx=model.onnx --saveEngine=model_int8.plan --int8 --fp16 \
    --calib=calibration.cache

# ── Dynamic shapes engine build with profiles ──
trtexec --onnx=model.onnx --saveEngine=model.plan --fp16 \
    --minShapes=input:1x3x224x224 \
    --optShapes=input:8x3x224x224 \
    --maxShapes=input:64x3x224x224

# ── Load and benchmark an existing engine ──
trtexec --loadEngine=model.plan --batch=32 --iterations=100 --warmUp=500

# ── Profile per-layer timing (find bottleneck layers) ──
trtexec --loadEngine=model.plan --dumpProfile --separateProfileRun

# ── Compare correctness: ONNXRuntime vs TensorRT ──
polygraphy run model.onnx --onnxrt --trt --atol 1e-3 --rtol 1e-3

# ── Inspect engine binding names (needed for Triton config) ──
trtexec --loadEngine=model.plan --verbose 2>&1 | grep "Binding"
```

### Full Pipeline Artifacts Checklist

At each stage of the pipeline, verify the expected files exist. Missing files indicate a failed step:

```
After ONNX export:
  ✓ model.onnx                  ← Exported model graph + weights

After calibration (if using implicit INT8):
  ✓ calibration.cache           ← Computed INT8 scales (reusable across builds)

After engine build:
  ✓ model.plan                  ← Compiled TensorRT engine for your specific GPU
  ✓ timing.cache (optional)     ← Tactic timing cache (speeds up rebuilds)

After Triton setup:
  ✓ model_repository/
      └── my_model/
          ├── config.pbtxt      ← Triton model configuration
          └── 1/
              └── model.plan    ← Engine file (version 1)

After ModelOpt PTQ:
  ✓ model_qdq.onnx              ← ONNX with explicit Q/DQ nodes (scales baked in)

After TRT-LLM quantization:
  ✓ checkpoint_dir/             ← Quantized checkpoint (consumed by trtllm-build)
  ✓ engine_dir/                 ← TRT-LLM engine directory (consumed by Triton)
```

---

### trtexec: The Canonical Build and Benchmark Tool

`trtexec` is TensorRT's Swiss Army knife — the standard tool for building engines, benchmarking, and debugging. Every TensorRT workflow starts here.

**Canonical dynamic-shape build pattern:**
```bash
# Build a plan with dynamic shapes (one input called "input")
trtexec \
  --onnx=model.onnx \
  --saveEngine=model.plan \
  --minShapes=input:1x3x224x224 \
  --optShapes=input:8x3x224x224 \
  --maxShapes=input:128x3x224x224 \
  --fp16 \
  --workspace=4096 \
  --verbose
```

The `--minShapes/--optShapes/--maxShapes` flags define optimization profiles for dynamic dimensions. TensorRT auto-tunes kernels for the `opt` shape and guarantees correctness across the `min`–`max` range. This is essential for Triton deployment where batch sizes vary.

**Dynamic Shapes First Principles — Micro-Example:**

If your ONNX model has a dynamic batch dimension (common for any model served via Triton):

```bash
# Step 1: Export ONNX with dynamic batch
python -c "
import torch, torchvision
m = torchvision.models.resnet50(pretrained=False).eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(m, dummy, 'resnet50_dynamic.onnx', opset_version=17,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})
"

# Step 2: Build with matching profiles
trtexec --onnx=resnet50_dynamic.onnx --saveEngine=resnet50_dynamic.plan --fp16 \
    --minShapes=input:1x3x224x224 \
    --optShapes=input:16x3x224x224 \
    --maxShapes=input:64x3x224x224

# Step 3: In Triton config.pbtxt:
#   max_batch_size: 64
#   dims: [ 3, 224, 224 ]       ← no batch dim here; Triton handles it
#   dynamic_batching { preferred_batch_size: [8, 16, 32] }

# What happens if profiles are missing in Triton:
#   → Error: "model expected static shape [1,3,224,224], got [8,3,224,224]"
#   → Fix: rebuild engine with --minShapes/--optShapes/--maxShapes
```

**Precision comparison workflow:**
```bash
# FP16 baseline
trtexec --onnx=model.onnx --fp16 --saveEngine=model_fp16.plan

# INT8 (with calibration cache)
trtexec --onnx=model.onnx --int8 --fp16 --calib=calibration.cache --saveEngine=model_int8.plan

# FP8 (Hopper+)
trtexec --onnx=model.onnx --fp8 --saveEngine=model_fp8.plan

# Profile per-layer timing (find bottleneck layers)
trtexec --loadEngine=model_int8.plan --dumpProfile --separateProfileRun
```

> **Expected output excerpt — successful build:**
> ```
> [I] [TRT] Detected 1 input and 1 output network tensors.
> [I] Engine built in 145.234 sec.
> [I] Saved engine to model_int8.plan
> ```

> **Expected output excerpt — `--dumpProfile`:**
> ```
> [I] === Profile (145 layers) ===
> [I] Layer(Convolution):   conv1 + bn1 + relu      Precision: INT8   Time: 0.032ms
> [I] Layer(Convolution):   layer1.0.conv1 + bn + relu   Precision: INT8   Time: 0.028ms
> [I] Layer(Reformat):      Reformat(layer2→layer3)  Precision: FP16→INT8  Time: 0.005ms
> [I] ...
> [I] Total: 1.234ms (87% INT8, 10% FP16, 3% Reformat)
> ```
> Look for: layers running in unexpected precision, high Reformat percentage, and layers that dominate total time.

**Red flags in trtexec output:**
| Signal | What It Means |
|--------|--------------|
| High "Reformat" time | TensorRT is spending time converting tensor layouts between layers — indicates poor fusion |
| Layer not fused | A layer that should be fused is running as a separate kernel — check if it's an unsupported pattern |
| Precision mismatch in profile | A layer you expected to run in INT8 is running in FP16 — silent fallback occurred |
| "No valid tactics" error | No kernel implementation exists for this layer at the requested precision on this GPU |

---

### Deep Profiling: Advanced Engine Inspection

The trtexec red flags above catch common issues. For expert-level diagnosis — understanding *which specific kernel* TensorRT selected for each layer, *why* a layer didn't quantize, and *where* the time is actually going — you need the detailed profiling and layer information export.

**Step 1: Export per-layer kernel selection and timing.**

```bash
# Export detailed layer information (which tactic was selected per layer)
trtexec --onnx=model.onnx --int8 --fp16 \
    --exportLayerInfo=layer_info.json \
    --profilingVerbosity=detailed \
    --saveEngine=model.plan

# The layer_info.json contains:
# - Layer name and type
# - Selected tactic (kernel implementation)
# - Precision used (FP32/FP16/INT8)
# - Input/output formats and dimensions
# - Execution time per layer
```

**Step 2: Interpret the output — what to look for.**

The `layer_info.json` file reveals the builder's internal decisions. Key fields to examine:

```json
{
  "Layers": [
    {
      "Name": "conv1 + relu1",
      "TacticName": "sm80_xmma_fprop_implicit_gemm_indexed_f16f16_...",
      "Precision": "Half",
      "LayerType": "CaskConvolution",
      "Inputs": [{"Format": "NC/2HW2", "Type": "Half"}],
      "Outputs": [{"Format": "NC/2HW2", "Type": "Half"}],
      "AverageMs": 0.0342
    }
  ]
}
```

**Red flags in the tactic names:**

| Tactic Pattern | What It Means | Action |
|----------------|---------------|--------|
| `xmma_*` or `hmma_*` | Tensor Core kernel — this is what you want | Good — hardware accelerated |
| `cudnn_*` or `generic_*` | Generic cuDNN kernel, not fused | Check if fusion was blocked; possibly an unsupported pattern |
| `myelin_*` | Myelin (TensorRT's graph compiler) fused kernel | Good — aggressive fusion happened |
| Precision shows `Half` when you expected `Int8` | INT8 kernel was slower or unavailable for this layer | Silent fallback; check if alignment/dimensions are suboptimal |
| `reformat_*` | Pure data layout conversion, no compute | Overhead; indicates format mismatch between adjacent layers |

**Step 3: Diagnose "why didn't this layer quantize?"**

When a layer stays in FP16 despite `--int8` being set, the reasons are (in order of likelihood):

1. **No INT8 tactic exists** for this layer type on this GPU (e.g., LayerNorm on pre-Hopper GPUs)
2. **The INT8 tactic was slower** than the FP16 tactic during auto-tuning (TensorRT picks the fastest)
3. **Dimensions are poorly aligned** — tile sizes don't map efficiently to INT8 Tensor Core tiles
4. **The layer was explicitly forced to FP16** in the network definition

To force the issue and get an error instead of a silent fallback:
```python
config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)
```

**Step 4: Correlate with Nsight Systems for end-to-end analysis.**

For the most complete picture, combine layer info with an Nsight Systems trace:

```bash
# Capture a system-level trace with per-kernel detail
nsys profile --trace=cuda,nvtx,osrt \
    --output=profile_report \
    trtexec --loadEngine=model.plan --batch=32 --iterations=100

# Open in Nsight Systems UI:
nsys-ui profile_report.nsys-rep
```

In the Nsight Systems timeline, you can see:
- Each kernel's GPU time and whether it's compute-bound or memory-bound
- Gaps between kernels (launch overhead — CUDA graphs can eliminate these)
- Memory transfer overhead (H2D / D2H copies)
- Whether multiple streams are overlapping (pipelining)

This is the NVIDIA equivalent of the diagnostic grep commands in Chapter 20's Qualcomm profiling. The workflow is: `trtexec --exportLayerInfo` tells you *what* TensorRT chose → Nsight tells you *how* the GPU actually executed it → the gap between the two reveals your optimization opportunity.

**What to do when the selected tactic is slow:**

Export layer info might reveal that TensorRT's auto-tuning selected a suboptimal tactic — for example, a `generic_*` kernel for a layer that should be running on Tensor Cores. When this happens, you have three levers:

1. **Adjust tensor dimensions.** Tensor Core kernels require specific alignment (multiples of 8 for FP16, 16 for INT8). If your layer dimensions are slightly off (e.g., hidden_dim=1023 instead of 1024), padding to the aligned size can unlock faster tactics. Check `Inputs.Format` in layer_info.json — `NC/32HW32` indicates a Tensor Core-friendly layout.
2. **Refactor the graph.** If a layer consistently selects a slow kernel, it may be because the surrounding ops prevent fusion. Use `onnx-graphsurgeon` to restructure — for example, splitting a large grouped convolution into separate convolutions can sometimes unlock better tactics.
3. **Pin tactics via timing cache.** After finding a good tactic configuration, save the timing cache (`--timingCacheFile=timing.cache`) and reuse it in subsequent builds. This prevents TensorRT from re-autotuning and potentially selecting a different (worse) tactic on a slightly different run.

```bash
# Save timing cache after a good build
trtexec --onnx=model.onnx --int8 --fp16 \
    --saveEngine=model.plan \
    --timingCacheFile=timing.cache

# Reuse timing cache in subsequent builds (deterministic tactic selection)
trtexec --onnx=model.onnx --int8 --fp16 \
    --saveEngine=model_v2.plan \
    --timingCacheFile=timing.cache
```

> **Rule of thumb:** If `exportLayerInfo` shows more than 10% of total inference time in `reformat_*` layers, you have a fusion problem. If a compute layer shows `generic_*` instead of `xmma_*`/`hmma_*`, you have an alignment or dimension problem.

---

### Polygraphy: The Correctness Debugging Tool

For a complete TensorRT debugging workflow, you need three tools:
1. **trtexec** — performance and build
2. **Polygraphy** — correctness, diffing, ONNX sanity, engine inspection
3. **Nsight** — kernel-level profiling

Your chapter already covers trtexec and Nsight. Polygraphy fills the critical gap: **"is my quantized output correct?"**

**What Polygraphy does:**
- Compares outputs across runtimes (PyTorch → ONNXRuntime → TensorRT)
- Identifies which layers introduce the largest numerical differences
- Inspects ONNX graphs for issues before TensorRT consumption
- Debugs TensorRT engine internals (layer precision, tactics selected)

**Common Polygraphy workflows:**

```bash
# Compare ONNX Runtime vs TensorRT outputs (find accuracy regressions)
polygraphy run model.onnx \
  --onnxrt --trt \
  --atol 1e-3 --rtol 1e-3 \
  --val-range input:[0,1]

# Inspect ONNX model structure and op support
polygraphy inspect model model.onnx --mode=basic

# Debug a TensorRT engine (show layer precisions and tactics)
polygraphy inspect model model.plan --mode=layers

# Bisect to find the first layer that diverges
polygraphy debug precision model.onnx \
  --fp32 --int8 --check polygraphy_check.py
```

**When to use Polygraphy:**
- After INT8 calibration, to verify outputs match FP32 within tolerance
- When task-level accuracy drops but you don't know which layer is responsible
- Before deploying a new model version, as a correctness gate in CI
- When debugging ONNX export issues (unsupported ops, shape mismatches)

---

### Plugins and Unsupported Operations — Escalation Ladder (Advanced)

TensorRT does not support every ONNX operation or pattern. When the ONNX parser encounters an unsupported op, it fails the build. This is the most common real-world blocker when bringing new models to TensorRT.

**How to detect unsupported layers:**
```bash
# Available in: NGC tensorrt container
# Verbose ONNX parsing — shows which ops succeed and which fail
trtexec --onnx=model.onnx --verbose 2>&1 | grep -i "error\|unsupported\|warning"

# Polygraphy ONNX inspection
polygraphy inspect model model.onnx --mode=basic
```

**Escalation ladder — always try in this order:**

| Step | Approach | Time/Cost | When to Use |
|------|----------|-----------|-------------|
| **1. Re-export** | Change PyTorch code to use supported ops, re-export ONNX | ~1 hour | Custom attention → `scaled_dot_product_attention`; custom norm → standard GroupNorm |
| **2. ONNX surgery** | Rewrite ONNX graph with `onnx-graphsurgeon` | ~2–4 hours | Unsupported op has a mathematical equivalent in supported ops |
| **3. Plugin (last resort)** | Write C++/CUDA plugin | ~1–2 weeks (including testing) | Genuinely novel op with no ONNX equivalent; custom quantized kernel |

> **Rule:** Plugins are a last resort. They require C++/CUDA expertise, must be maintained across TensorRT versions, and must be distributed with every engine. Always try steps 1 and 2 first.

**Step 1 — Graph rewrite (preferred):** Modify the PyTorch model or ONNX export to use supported operations. For example, replace a custom attention with `torch.nn.functional.scaled_dot_product_attention`, which has good ONNX/TRT support.

**Step 2 — ONNX surgery:** Use `onnx-graphsurgeon` (NVIDIA's tool) to rewrite the ONNX graph directly — replace unsupported nodes with equivalent supported patterns.

```python
# Available in: NGC tensorrt container (onnx-graphsurgeon pre-installed)
import onnx_graphsurgeon as gs
graph = gs.import_onnx(onnx.load("model.onnx"))
# Find and replace unsupported nodes
for node in graph.nodes:
    if node.op == "UnsupportedOp":
        # Replace with supported equivalent
        ...
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "model_fixed.onnx")
```

**Step 3 — TensorRT plugins:** Write a custom C++/CUDA plugin that implements the unsupported op. This is the most work but handles truly custom operations — for example, a novel quantized attention kernel from a research paper that TensorRT doesn't natively support.

**Plugin Distribution Checklist (Advanced):**
- [ ] Plugin `.so` lives alongside the engine in the model repository (`plugins/` subdirectory)
- [ ] Triton loads it via `LD_PRELOAD` or the model config's `parameters` section
- [ ] `LD_LIBRARY_PATH` includes the plugin directory in the container (set in Dockerfile or `docker run -e`)
- [ ] Plugin is compiled against the **same** TensorRT version as the engine — version mismatches cause silent crashes
- [ ] Plugin `.so` is pinned and versioned in your artifact registry (not "latest" — a TensorRT upgrade requires plugin recompile)
- [ ] Integration test: load engine + plugin in a fresh container and verify output matches reference

**TensorRT Plugin: Complete C++ Boilerplate**

> **Note:** This full C++ boilerplate is included here for Advanced/Expert readers. Beginners should skip to the "Production lesson" note below.

A TensorRT plugin has two components: the **plugin** (does the computation) and the **plugin creator** (factory that TensorRT uses to instantiate the plugin). Here is a minimal but complete skeleton:

```cpp
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <cuda_runtime.h>
#include <cstring>
#include <vector>

using namespace nvinfer1;

// ─── The Plugin ───
class MyQuantizedOp : public IPluginV2DynamicExt {
public:
    MyQuantizedOp(float scale) : mScale(scale) {}

    // Deserialization constructor (loading from engine file)
    MyQuantizedOp(const void* data, size_t length) {
        const char* d = static_cast<const char*>(data);
        mScale = *reinterpret_cast<const float*>(d);
    }

    // ── Shape inference ──
    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs,
                                  int nbInputs, IExprBuilder& builder) noexcept override {
        return inputs[0];  // Output shape == input shape for element-wise ops
    }

    // ── Precision/format support ──
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut,
                                    int nbInputs, int nbOutputs) noexcept override {
        // Support FP16 and INT8 (the quantized types we care about)
        return (inOut[pos].type == DataType::kHALF || inOut[pos].type == DataType::kINT8)
            && inOut[pos].format == TensorFormat::kLINEAR;
    }

    // ── The actual CUDA kernel launch ──
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs,
                void* workspace, cudaStream_t stream) noexcept override {
        const int n = volume(inputDesc[0].dims);
        // Launch your custom CUDA kernel here:
        // myQuantizedKernel<<<blocks, threads, 0, stream>>>(
        //     static_cast<const int8_t*>(inputs[0]),
        //     static_cast<int8_t*>(outputs[0]),
        //     mScale, n);
        return 0;
    }

    // ── Serialization (save plugin state into engine file) ──
    size_t getSerializationSize() const noexcept override { return sizeof(float); }
    void serialize(void* buffer) const noexcept override {
        *static_cast<float*>(buffer) = mScale;
    }

    // ── Metadata ──
    const char* getPluginType() const noexcept override { return "MyQuantizedOp"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    int getNbOutputs() const noexcept override { return 1; }
    void destroy() noexcept override { delete this; }

    IPluginV2DynamicExt* clone() const noexcept override {
        return new MyQuantizedOp(mScale);
    }

    // (Other required methods: configurePlugin, getWorkspaceSize, etc.
    //  — return defaults for simple plugins)
    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs,
                         const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override {}
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs,
                            const PluginTensorDesc* outputs, int nbOutputs) const noexcept override { return 0; }
    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    void setPluginNamespace(const char* ns) noexcept override { mNamespace = ns; }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }
    DataType getOutputDataType(int index, const DataType* inputTypes,
                                int nbInputs) const noexcept override { return inputTypes[0]; }

private:
    float mScale;
    std::string mNamespace;

    int volume(Dims d) const {
        int v = 1; for (int i = 0; i < d.nbDims; i++) v *= d.d[i]; return v;
    }
};

// ─── The Plugin Creator (factory) ───
class MyQuantizedOpCreator : public IPluginCreator {
public:
    const char* getPluginName() const noexcept override { return "MyQuantizedOp"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const PluginFieldCollection* getFieldNames() noexcept override { return &mFC; }

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override {
        float scale = 1.0f;
        for (int i = 0; i < fc->nbFields; i++) {
            if (strcmp(fc->fields[i].name, "scale") == 0)
                scale = *static_cast<const float*>(fc->fields[i].data);
        }
        return new MyQuantizedOp(scale);
    }

    IPluginV2* deserializePlugin(const char* name, const void* data,
                                  size_t length) noexcept override {
        return new MyQuantizedOp(data, length);
    }

    void setPluginNamespace(const char* ns) noexcept override { mNamespace = ns; }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
    PluginFieldCollection mFC{0, nullptr};
    std::string mNamespace;
};

// ─── Register the plugin so TensorRT can find it ───
REGISTER_TENSORRT_PLUGIN(MyQuantizedOpCreator);
```

**Compiling and loading the plugin:**
```bash
# Compile the plugin as a shared library
nvcc -shared -o libmyquantizedop.so my_quantized_op.cpp \
    -I/usr/include/x86_64-linux-gnu \
    -lnvinfer -lcudart

# Load the plugin when building an engine
trtexec --onnx=model.onnx --plugins=libmyquantizedop.so --saveEngine=model.plan
```

> **When to write a plugin vs. when to rewrite your model:** Plugins are a last resort. They require C++/CUDA expertise, have a significant maintenance burden across TensorRT versions, and must be distributed with your engine. Always try graph rewrite or ONNX surgery first. Plugins are justified when: (1) the op is genuinely novel and has no ONNX equivalent, (2) you need a custom quantized kernel that TensorRT's auto-tuning can't match, or (3) you're deploying a research paper's custom kernel in production.

**Plugin deployment with Triton:** Custom plugins must be packaged with the engine and loaded by Triton at startup. Add the plugin .so file to your model repository and configure Triton to load it:
```
model_repository/
└── my_model/
    ├── config.pbtxt
    ├── 1/
    │   └── model.plan
    └── plugins/
        └── libmycustomplugin.so
```

> **The production lesson:** Before committing to a model architecture for TensorRT deployment, verify that all its ops are supported. Run a quick `trtexec --onnx=model.onnx` early in the development cycle. Discovering unsupported ops after training is expensive.

---

## Troubleshooting: When Things Go Wrong

Like Chapter 20's three diagnostic scenarios for Qualcomm, here are the NVIDIA-specific failure modes and how to fix them.

### The 5 Most Common Errors (and Exact Fixes)

Before diving into the deep diagnostics, here are the errors that hit beginners most often. These are ordered by frequency — search for the exact error string to find your fix:

**1. `Error: no NVIDIA GPU detected` or `nvidia-smi` fails**
```
Symptom:  nvidia-smi returns "NVIDIA-SMI has failed" or "command not found"
Cause:    NVIDIA driver not installed, or driver/kernel mismatch after OS update
Fix:      Install the latest NVIDIA driver from nvidia.com. On Linux, verify with
          `lsmod | grep nvidia`. After a kernel update, you may need to reinstall
          the driver or reboot.
```

**2. `docker: Error response from daemon: could not select device driver`**
```
Symptom:  docker run --gpus all fails with device driver error
Cause:    nvidia-container-toolkit not installed or Docker daemon not restarted
Fix:      Install nvidia-container-toolkit (see NVIDIA's container toolkit docs),
          then restart Docker: sudo systemctl restart docker
          Verify: docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

**3. `unauthorized: authentication required` when pulling NGC containers**
```
Symptom:  docker pull nvcr.io/nvidia/tensorrt:... fails with auth error
Cause:    Missing NGC registry login
Fix:      docker login nvcr.io
          Username: $oauthtoken    (literally this string, not your email)
          Password: <NGC API key from https://ngc.nvidia.com/setup/api-key>
```

**4. `ONNX parse failed` or `Unsupported ONNX opset` during engine build**
```
Symptom:  trtexec --onnx=model.onnx fails with parser errors
Cause:    ONNX opset version higher than TensorRT supports, or model uses
          unsupported operations
Fix:      Re-export with a lower opset: torch.onnx.export(..., opset_version=17)
          For unsupported ops: see "Plugins and Unsupported Operations" section.
          Quick check: polygraphy inspect model model.onnx --mode=basic
```

**5. `Engine builds, outputs are all wrong / random`**
```
Symptom:  Engine loads and runs without errors, but predictions are garbage
Cause:    Almost always preprocessing mismatch — the inference preprocessing
          differs from training preprocessing (different normalization, resize,
          channel order RGB vs BGR, or wrong input dtype)
Fix:      Verify preprocessing is byte-for-byte identical to training.
          Compare: polygraphy run model.onnx --onnxrt --trt --atol 1e-3
          If ONNXRuntime output is also wrong, the problem is ONNX export, not TRT.
```

---

### Scenario A: "Engine Builds, but Output Is Wrong"

This is the accuracy failure — the engine runs but produces incorrect predictions. Systematic diagnosis:

**Step 1: Isolate where the error enters.**
```bash
# Compare FP32 PyTorch → ONNXRuntime → TensorRT FP16 → TensorRT INT8
polygraphy run model.onnx --onnxrt --trt --atol 1e-3 --rtol 1e-3
```

If ONNXRuntime and PyTorch disagree, the problem is in ONNX export (not TensorRT). If TRT FP16 and ONNXRuntime agree but TRT INT8 diverges, the problem is quantization.

**Step 2: Check preprocessing.**
The most common cause of "wrong output" is different preprocessing between training and inference. Verify:
- Same normalization (mean/std)
- Same resize/crop
- Same channel order (RGB vs BGR)
- Same dtype (float32 input expected)

This is identical to the Qualcomm "input mismatch" scenario in Chapter 20.

**Step 3: Check quantization scales.**
- If using **explicit Q/DQ**: verify Q/DQ nodes have reasonable scale values (not 0, not infinity)
- If using **implicit calibration**: verify the calibration cache corresponds to the same model and preprocessing
- If calibration used non-representative data: re-calibrate with production-like data → **Calibration Mismatch** (failure pattern from Chapter 13)

**Step 4: Check for outlier-induced resolution collapse.**
Transformer activations with extreme outliers (Chapter 14) cause **Resolution Collapse** under INT8 — the scale is set by the outlier, crushing resolution for normal values. Fix: use SmoothQuant (Chapter 15) or switch to FP8.

> **Rule:** If INT8 accuracy drops >1% for CNNs or >0.5 perplexity for LLMs, suspect calibration or outliers before blaming TensorRT.

---

### Scenario B: "Engine Builds, but It's Slower Than Expected"

You expected 2× speedup from INT8, but got only 1.1×. Diagnosis:

**Step 1: Are you actually using Tensor Cores?**
```bash
trtexec --loadEngine=model.plan --dumpProfile --separateProfileRun
```
Check the profile output for:
- Layers running in unexpected precision (FP16 fallback instead of INT8)
- High "Reformat" overhead (tensor layout conversions between layers)

**Step 2: Is the workload memory-bound?**
If batch=1 and your model is large, you are memory-bandwidth-bound, not compute-bound. INT8 compute throughput doesn't help — you need weight-only quantization (W4A16) to reduce memory traffic. This connects directly to the memory-vs-compute model from Chapter 16.

**Step 3: Is workspace too small?**
Insufficient workspace forces TensorRT to fall back to slower kernel tactics:
```python
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
```

**Step 4: Are dynamic shapes causing suboptimal profile selection?**
If your optimization profile has a wide min-max range (e.g., batch 1–256), TensorRT optimizes for the `opt` shape. Inputs far from `opt` may run suboptimally. Create multiple profiles for different operating points.

**Step 5: Are Tensor Core alignment requirements met?**
Matrix dimensions that aren't multiples of 8 (INT8) or 16 (FP8) cause padding overhead. This is usually handled by TensorRT, but custom layers or unusual shapes can defeat alignment.

### Reformat Overhead Remediation Playbook (Advanced)

If `trtexec --dumpProfile` shows high "Reformat" time, follow this ordered playbook:

**1. Identify the reformat layers.**
```bash
# Available in: NGC tensorrt container
trtexec --loadEngine=model.plan --dumpProfile --separateProfileRun 2>&1 | grep -i "reformat"
# Each reformat line shows: source format → destination format, and time in ms
```

**2. Understand the causes.**

| Cause | Example | Fix |
|-------|---------|-----|
| **Mixed precision boundaries** | FP16 layer → INT8 layer | Reduce precision crossings: keep contiguous blocks in the same precision via ModelOpt layer-level config |
| **Layout mismatch** | NCHW layer → NHWC layer | Re-export ONNX with consistent layout; or accept the cost if TensorRT chose NHWC for Tensor Core efficiency |
| **Non-fused Q/DQ** | Q/DQ nodes not absorbed into compute kernel | Check that Q/DQ nodes are placed on fusable boundaries (Conv/MatMul inputs/outputs). Misplaced Q/DQ = extra reformat |
| **Plugin output format** | Plugin outputs NCHW but next layer expects NHWC | Make plugin output match the expected layout (implement `supportsFormatCombination` correctly) |

**3. Measure the impact.** Reformat overhead of <5% total inference time is typically acceptable. Above 10%, it's worth investigating.

**4. When you cannot eliminate the reformat:** If TensorRT chose NHWC for Tensor Core efficiency but your pre/post-processing expects NCHW, the reformat at the boundary is unavoidable and correct. The Tensor Core speedup usually outweighs the reformat cost.

---

### Scenario C: "Build Succeeds on One Machine, Fails in Production"

The "works on my machine" problem. Most common causes:

| Symptom | Cause | Fix |
|---------|-------|-----|
| `deserialize_cuda_engine` fails | TensorRT version mismatch | Rebuild engine in production container, or use `--versionCompatible` flag |
| Engine loads but crashes | GPU architecture mismatch | Rebuild for target GPU, or use hardware compatibility mode (perf tradeoff) |
| Plugin not found | Plugin .so not in LD_LIBRARY_PATH | Package plugins with engine, set `LD_LIBRARY_PATH` in container |
| CUDA error on launch | Driver too old for CUDA version | Update driver or use NGC container with matching driver requirements |
| Different outputs | Different cuDNN/cuBLAS version | Pin container version; same container for build and deploy |

**The golden rule:** Build engines inside the same container image used in production. Use NGC containers as the known-good environment.

---

## Common Pitfalls (Quick Reference)

**1. Engine not portable across GPUs:**
Build separate engines per GPU type, or use hardware compatibility mode (with performance cost). See "Portability Decision Box" in the Engine Lifecycle section.

**2. ONNX opset mismatches:**
TensorRT supports specific ONNX opsets. If your ONNX model uses opset 18 but your TensorRT version only supports up to opset 17, the parser will fail. Check compatibility before exporting.

**3. Dynamic shapes without profiles:**
TensorRT requires **optimization profiles** for dynamic shapes:
```python
profile = builder.create_optimization_profile()
profile.set_shape("input",
    min=(1, 3, 224, 224),    # Minimum shape
    opt=(8, 3, 224, 224),    # Optimal shape (auto-tuning target)
    max=(32, 3, 224, 224)    # Maximum shape
)
config.add_optimization_profile(profile)
```
Without profiles, TensorRT assumes static shapes and errors on dynamic inputs.

**4. Calibration data not representative:**
Non-representative calibration data → wrong scales → **Calibration Mismatch** (Chapter 13). Use production-like data.

**5. FP16 accumulation causing NaN:**
Some models produce intermediate values that overflow FP16 range (±65504). Fix: keep specific layers in FP32, or use BF16 (wider range) if available.

**6. Workspace memory too small:**
```python
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
```
When in doubt, give it more workspace.

**7. INT8 calibration on wrong GPU:**
Calibration should ideally be done on the same GPU type as deployment.

---

## Performance Tuning: Getting the Last 10%

**1. Use trtexec for quick experiments:**
```bash
# Compare FP16 vs INT8 vs FP8
trtexec --onnx=model.onnx --fp16 --saveEngine=model_fp16.plan
trtexec --onnx=model.onnx --int8 --fp16 --saveEngine=model_int8.plan
trtexec --onnx=model.onnx --fp8 --saveEngine=model_fp8.plan
```

**2. Profile kernel-level bottlenecks:**
If a model is not hitting expected throughput, profile individual kernels:
```bash
trtexec --loadEngine=model_int8.plan --dumpProfile --separateProfileRun
```
This shows per-layer timing. Look for layers that are disproportionately slow — they may be running in FP32 fallback.

**3. Use CUDA graphs:**
For latency-sensitive workloads, CUDA graphs capture the entire inference as a single GPU submission:
```python
# Capture CUDA graph
stream = cuda.Stream()
with torch.cuda.graph(graph, stream=stream):
    output = run_engine(input_buffer)

# Replay (much faster than individual kernel launches)
graph.replay()
```

**4. Optimize batch size:**
INT8 models have higher throughput, which means they can efficiently process larger batches. Profile throughput vs. latency at different batch sizes to find the sweet spot:

> **Illustrative numbers** — measured on a specific GPU SKU, TensorRT version, and model. Your results will vary with hardware, model architecture, and TRT version. Always measure on your own setup.

| Batch Size | Latency (ms) | Throughput (img/s) |
|-----------|-------------|-------------------|
| 1 | 0.8 | 1250 |
| 8 | 1.2 | 6667 |
| 32 | 2.5 | 12800 |
| 64 | 4.8 | 13333 |
| 128 | 9.5 | 13474 |

> **Measurement knobs:** batch size, sequence length (for LLMs), GPU SKU, TRT/TRT-LLM version. Change any one and the numbers change.

Throughput plateaus around batch 64–128 for INT8. Beyond that, you are limited by memory bandwidth.

**5. Multi-instance GPU (MIG) on A100/H100:**
For serving multiple models, use MIG to partition the GPU:
```bash
nvidia-smi mig -cgi 9,9,9,9,9,9,9 -C  # 7 instances on A100
```
Each instance gets a slice of compute and memory. Useful when serving many small quantized models.

**6. Sparsity on Ampere+:**
If using structured sparsity (2:4), ensure TensorRT is building with sparsity-aware kernels:
```python
config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
```

**7. Quantization increases fusion; fusion reduces launches; graphs reduce launch overhead further.**
This is the compounding effect: INT8 quantization enables more aggressive layer fusion (quant/dequant absorbed into compute kernels), fusion reduces the number of kernel launches, and CUDA graphs reduce the per-launch overhead. The three optimizations multiply.

---

## Decision Tables

### TensorRT (Non-LLM) Quantization Decision Table

| Model Type | Default Precision | Calibration | Fallback Strategy |
|-----------|------------------|-------------|-------------------|
| CNN classification/detection | FP16 first, then INT8 PTQ | Entropy (default) | If INT8 drops >1%, try percentile calibration or per-channel Q/DQ |
| Sensitive regression models | INT8 with per-channel Q/DQ | Percentile or MinMax | Keep LayerNorm/output layers in FP32 |
| Transformer encoder (BERT) | INT8 Q/DQ or FP16 | Entropy; careful with softmax/LayerNorm | SmoothQuant if outliers present |
| Generative model (diffusion) | FP16 (INT8 risky) | Not recommended without QAT | Use ModelOpt QAT if INT8 needed |
| Real-time (latency-critical) | INT8 + CUDA graphs | Entropy | Profile; ensure no reformat overhead |

> **Rule:** Implicit quantization is deprecated. Use explicit Q/DQ for portability and reproducibility in all new projects.

---

## Toolbox — Everything You Need in One Place

| Tool | Purpose | Key Command |
|------|---------|-------------|
| **trtexec** | Build engines, benchmark, profile | `trtexec --onnx=model.onnx --fp16 --int8` |
| **Polygraphy** | Correctness debugging, runtime comparison | `polygraphy run model.onnx --onnxrt --trt` |
| **Nsight Systems** | System-level trace (kernel launches, memory) | `nsys profile trtexec --loadEngine=m.plan` |
| **Nsight Compute** | Kernel-level analysis (occupancy, throughput) | `ncu trtexec --loadEngine=m.plan --batch=1` |
| **onnx-graphsurgeon** | ONNX graph rewriting | `import onnx_graphsurgeon as gs` |
| **ModelOpt** | PyTorch PTQ/QAT/sparsity | `mtq.quantize(model, cfg, calibrate)` |
| **Torch-TensorRT** | PyTorch-first TRT compilation | `torch_tensorrt.compile(model, ...)` |
| **TRT-LLM quantize.py** | LLM quantization scripts | `python quantize.py --qformat fp8` |
| **trtllm-build** | Build TRT-LLM engines | `trtllm-build --checkpoint_dir ./ckpt` |
| **Triton** | Model serving | `tritonserver --model-repository=/models` |

**Canonical references:**
- TensorRT Support Matrix — the definitive source for driver/CUDA/cuDNN/GPU compatibility
- TensorRT Quantization Documentation — precision support and quantized type details
- TRT-LLM GitHub — quantization examples and model support matrix

---

## Engine Lifecycle in Production

### Portability Decision Box (Advanced)

| Strategy | When to Use | Performance Cost | Operational Complexity | File Size Impact |
|----------|-------------|-----------------|----------------------|-----------------|
| **Rebuild per GPU** (default) | Homogeneous fleet, CI/CD can build per target | None (optimal) | Must maintain build pipelines per GPU | Normal |
| **`--hardwareCompatibilityLevel=ampere+`** | Mixed Ampere/Ada/Hopper fleet | 5–15% slower (generic kernels) | Single build, deploy anywhere ≥ Ampere | ~Same |
| **`--versionCompatible`** | Rolling TensorRT upgrades without rebuild | 0–5% (version-specific optimizations lost) | Requires trusted-plan governance | Larger (10–30%) |
| **Both flags combined** | Maximum flexibility (fleet heterogeneity + rolling upgrades) | Compounded: 5–20% | Requires trusted plans + version tracking | Largest |

> **Rule:** Start with "rebuild per GPU." Only add portability flags when your fleet or upgrade cadence demands it, and always benchmark the performance cost on your target workload.

### CI/CD Integration

Treat calibration cache + timing cache as **build artifacts** alongside your `.plan` files:

```bash
# CI pipeline: build → cache → store → deploy
trtexec --onnx=model.onnx --int8 --fp16 \
    --calib=calibration.cache \
    --timingCacheFile=timing.cache \
    --saveEngine=model.plan

# Store as CI artifacts:
# - model.plan            → deploy to Triton model repository
# - calibration.cache     → reuse across builds for same model
# - timing.cache          → reuse across builds on same GPU type
```

**Rebuild triggers:** Any of these should trigger a fresh engine build in CI:
- Model architecture or weight change
- TensorRT version upgrade (new container tag)
- Target GPU change (e.g., migrating from A100 to H100)
- Calibration data update

**Pin container versions** in your CI config (e.g., `nvcr.io/nvidia/tensorrt:24.05-py3`, not `:latest`). An unpinned container means unpinned TensorRT version, which means non-reproducible engines.

### Build-Time Caching

TensorRT build is expensive. Cache aggressively:

1. **Calibration cache** — stores computed INT8 scales. Reused across engine builds for the same model.
2. **Tactic timing cache** — stores kernel auto-tuning results. Dramatically speeds up rebuilds on the same GPU.

```bash
# Build with timing cache
trtexec --onnx=model.onnx --int8 --fp16 \
    --timingCacheFile=timing.cache --saveEngine=model.plan

# Rebuild (reuses timing cache — much faster)
trtexec --onnx=model_v2.onnx --int8 --fp16 \
    --timingCacheFile=timing.cache --saveEngine=model_v2.plan
```

### Version-Compatible Engines

For fleet upgrades where you can't rebuild every engine simultaneously:
```bash
trtexec --onnx=model.onnx --saveEngine=model.plan --versionCompatible
```

**Caveats:**
- Increases plan file size
- Requires "trusted plan" security mechanism (engines must be marked trusted)
- Performance may differ from version-specific builds
- Only forward-compatible (old engine on newer TRT), not backward

### Hardware Compatibility Mode

For heterogeneous GPU fleets:
```bash
trtexec --onnx=model.onnx --saveEngine=model.plan --hardwareCompatibilityLevel=ampere+
```

This produces an engine that runs on any Ampere or newer GPU, but uses generic kernels instead of architecture-specific optimized ones. Expect 5–15% performance cost.

### Weight Stripping and Refit

For large models, you can strip weights from the engine and refit them at load time:
```bash
# Build with weight stripping
trtexec --onnx=model.onnx --saveEngine=model_stripped.plan --stripWeights

# Refit weights at load time (smaller engine file for distribution)
```

This is useful for CI/CD pipelines where you want to distribute engine structure separately from weights.

### Advanced Exit Criteria

**You have completed the Advanced tier when all are true:**
- [ ] You can read `trtexec --dumpProfile` output and identify bottleneck layers
- [ ] You can diagnose and explain reformat overhead (cause + fix)
- [ ] You have a portability strategy for your fleet (rebuild per GPU vs. hardware compatibility)
- [ ] Unsupported ops are resolved via the escalation ladder (rewrite → surgery → plugin)
- [ ] For LLMs: you can select the right recipe for your workload phase (prefill vs. decode vs. long-context)
- [ ] Engine lifecycle is integrated into CI/CD (calibration cache + timing cache as artifacts)

---

# ── EXPERT TIER ──

> **Expert Goals:** SLO-driven serving + fleet heterogeneity + deep diagnosis. After this tier, you can set SLOs (TTFT, inter-token latency, P99), pick batching policy, and maintain reproducibility across container upgrades and GPU fleets.
>
> **Skip guidance:** Most teams don't need this tier for a first deployment. Come here when you have multi-GPU requirements, need to hit specific latency SLOs, or are debugging production performance regressions.

## Expert Checklist

### CNN Deployment Checklist

- [ ] Verify all ONNX ops supported (`trtexec --onnx=model.onnx --verbose`)
- [ ] Build with FP16 first — establish baseline performance and accuracy
- [ ] Enable INT8 with entropy calibration using production-representative data
- [ ] Use explicit Q/DQ (ModelOpt PTQ) instead of implicit calibration
- [ ] Compare FP32 → FP16 → INT8 accuracy (`polygraphy run --onnxrt --trt`)
- [ ] Profile per-layer timing (`trtexec --dumpProfile --separateProfileRun`)
- [ ] Check for silent FP16 fallback in INT8 layers
- [ ] Set dynamic shape profiles matching production batch sizes
- [ ] Build engine on target GPU inside production container
- [ ] Cache calibration data and tactic timing for reproducible rebuilds
- [ ] Deploy via Triton with tuned dynamic batching parameters
- [ ] Monitor P99 latency and throughput in production

### LLM Deployment Checklist

- [ ] Choose quantization recipe from TRT-LLM recipe matrix based on GPU and workload
- [ ] For memory-bound decode: W4A16 (AWQ or GPTQ)
- [ ] For compute-bound prefill: W8A8 SmoothQuant
- [ ] For Hopper+: default to FP8
- [ ] Quantize KV cache (FP8 or NVFP4) for long-context workloads
- [ ] Run perplexity evaluation on held-out data before deployment
- [ ] Configure TRT-LLM in-flight batching and paged attention
- [ ] Deploy via Triton with `tensorrtllm` backend and decoupled mode
- [ ] Set SLO targets (time-to-first-token, inter-token latency, throughput)
- [ ] Monitor KV cache utilization and batch scheduling efficiency
- [ ] For multi-GPU: verify tensor parallelism topology and NVLink/NCCL config

### SLO Tuning Loop (Expert)

Production serving requires hitting specific Service Level Objectives. The tuning process differs for CNNs and LLMs:

**CNN SLO tuning — measure → change one knob → remeasure:**

```
1. Baseline:     trtexec --loadEngine=model.plan --batch=32 --iterations=1000
                 → record: P50 latency, P99 latency, throughput

2. Knob: Triton batching window
   max_queue_delay_microseconds: 1000 → 2000 → 5000
   → P99 latency vs throughput tradeoff

3. Knob: preferred_batch_size
   [8, 16] → [16, 32] → [32, 64]
   → larger = higher throughput, higher P99

4. Knob: instance_group count
   count: 1 → 2 → 4    (multiple engine instances per GPU)
   → more instances = more concurrent requests, more GPU memory

5. Knob: CUDA graphs (reduces kernel launch overhead)
   parameters { key: "enable_cuda_graph" value: { string_value: "true" } }
   → reduces P99 tail latency
```

**LLM SLO tuning — different knobs, different metrics:**

| Knob | Affects | Trade-off |
|------|---------|-----------|
| `batch_scheduler_policy: inflight_fused_batching` | Throughput + TTFT | Default for TRT-LLM; always enable |
| `max_tokens_in_paged_kv_cache` | Max concurrent sequences | Higher = more sequences but more memory; lower = OOM protection |
| `kv_cache_free_gpu_mem_fraction` | Memory split between model + KV | 0.9 = 90% for KV cache (good for long context) |
| TP degree (tensor parallelism) | Latency vs. throughput | More GPUs = lower latency per token, but higher total cost |
| `max_num_tokens` | Inflight batching token budget | Higher = more throughput but higher P99 |

> **SLO targets to set (LLM):**
> - **TTFT (Time to First Token):** typically <500ms for interactive, <2s for batch
> - **Inter-token latency:** typically <50ms for chat, <100ms for batch
> - **P99 latency:** set at 2–3× median; alert if exceeded
> - **Throughput:** tokens/second/GPU as the fleet-level metric

### Observability Must-Haves (Expert)

For production serving, instrument these metrics. Without them, you are debugging blind:

| Metric | Source | Why |
|--------|--------|-----|
| **Latency histogram** (P50/P95/P99) | Triton metrics endpoint (`:8002/metrics`) | Detect tail latency regressions |
| **GPU utilization** | `nvidia-smi --query-gpu=utilization.gpu` or DCGM | Low utilization = batching misconfigured or memory-bound |
| **GPU memory usage** | `nvidia-smi --query-gpu=memory.used` | Detect KV cache OOM before it crashes |
| **KV cache utilization** (LLM) | TRT-LLM metrics | High utilization = nearing capacity; evictions imminent |
| **Queue time vs. compute time** | Triton `nv_inference_queue_duration_us` vs `nv_inference_compute_infer_duration_us` | If queue >> compute, add instances or GPUs. If compute >> queue, optimize the engine |
| **Request success rate** | Triton `nv_inference_request_success/failure` | Detect silent failures (OOM, timeout, shape mismatch) |

```bash
# Quick check: Triton Prometheus metrics
curl localhost:8002/metrics | grep nv_inference
```

### Multi-GPU Deployment Rules (Expert)

The earlier multi-GPU section stated facts. Here are the deployment rules:

**Tensor Parallelism (TP) constraints:**
1. TP degree must evenly divide the number of attention heads. For LLaMA-2-70B (64 heads): TP=2, 4, 8 work. TP=3, 5, 6 do not.
2. NVLink topology is mandatory for TP efficiency. NVLink provides ~900 GB/s (H100 SXM); PCIe provides ~64 GB/s. TP across PCIe-only connections is 10–15× slower — effectively unusable for real-time serving.
3. All GPUs in a TP group must be the same architecture and SKU. Mixing A100 and H100 in one TP group is not supported.

**Pipeline Parallelism (PP) rules:**
- PP adds latency (pipeline bubbles) but reduces per-GPU memory
- Useful only when TP alone can't fit the model on available GPUs
- PP=2, TP=4 on an 8-GPU node serves a 70B model with good utilization

**When to use which:**

| Model Size | Available GPUs (H100 80GB) | Strategy |
|-----------|---------------------------|----------|
| 7B | 1 GPU | No parallelism needed (FP8: ~7GB) |
| 13B | 1 GPU | W4A16: fits in ~7GB; FP8: ~13GB, tight but works |
| 70B | 2–4 GPUs | TP=2 (FP8) or TP=4 (FP16); prefer TP=2 with FP8 |
| 70B | 8 GPUs | TP=8 for minimum latency; or TP=4 for cost efficiency with 2 replicas |
| 405B | 8 GPUs | TP=8, PP=2 across 2 nodes; or TP=8 with NVFP4 on Blackwell |

### Expert Failure Scenario: "TP Works but Is Slower Than Expected"

**Triage path:**

1. **Check topology:** `nvidia-smi topo -m` — verify NVLink connections. If you see `PHB` (PCIe Host Bridge) or `SYS` between GPUs instead of `NV#`, you're running TP over PCIe. Fix: place TP group on NVLink-connected GPUs.

2. **Check NCCL config:** Set `NCCL_DEBUG=INFO` and look for "Using network PCIe" — this confirms PCIe-only transport. For NVLink, you should see "Using network IB" or "NVLink."

3. **Check PCIe saturation:** If the GPUs are on different NUMA nodes, PCIe traffic crosses the CPU socket boundary, adding latency. Use `numactl --hardware` to verify GPU-to-CPU affinity.

4. **Check micro-batching mismatch:** If `max_batch_size` is too small for the TP degree, each GPU gets very little work per step, and communication overhead dominates. Rule: batch size per GPU should be ≥4 for TP to be worthwhile.

### Trusted Engine Governance (Expert)

Version-compatible engines use TensorRT's "trusted plan" mechanism. In production, this creates a governance requirement:

**The problem:** A version-compatible engine can run on newer TensorRT versions without rebuild. But this also means a *malicious or corrupted* `.plan` file could be loaded by your inference server. The trusted plan mechanism exists to prevent this.

**Governance rules:**
1. **Build engines only in CI/CD** — never accept manually-built engines for production
2. **Sign/hash `.plan` files** — store SHA-256 hashes of approved engines; verify before loading
3. **Store approved engines in a versioned artifact registry** (e.g., S3, GCS, or Artifactory) — not on local disks
4. **Triton model repository should be read-only** — mount as `:ro` in Docker to prevent runtime engine replacement
5. **Audit trail:** log which engine version is loaded at Triton startup, with container version and GPU info

```bash
# Compute hash of approved engine
sha256sum model.plan > model.plan.sha256

# Verify before deployment
sha256sum -c model.plan.sha256
# Expected: "model.plan: OK"
```

### Deep Profiling: 3-Artifact Triage Bundle (Expert)

For any performance or accuracy bug, attach these three artifacts to the investigation:

```
1. layer_info.json    — per-layer kernel selection, precision, format, timing
   → trtexec --onnx=model.onnx --int8 --fp16 \
       --exportLayerInfo=layer_info.json --profilingVerbosity=detailed

2. nsight_trace.qdrep — Nsight Systems trace (GPU timeline, kernel launches, memory copies)
   → nsys profile --trace=cuda,nvtx -o nsight_trace \
       trtexec --loadEngine=model.plan --batch=32 --iterations=100

3. polygraphy_debug/   — Polygraphy debug/bisect output (which layer introduced the error)
   → polygraphy debug reduce model.onnx --mode=bisect \
       --check polygraphy run polygraphy_debug/reduced.onnx --trt --atol 1e-3
```

> **Rule:** Attach these three to every perf/accuracy bug report. Without them, diagnosis is guesswork.

### Expert Exit Criteria

**You have completed the Expert tier when all are true:**
- [ ] SLOs are set and monitored (TTFT, inter-token latency, P99, throughput)
- [ ] Batching policy is tuned (Triton dynamic batching for CNN, inflight batching for LLM)
- [ ] Multi-GPU topology is validated (NVLink confirmed, TP degree correct, NCCL configured)
- [ ] Engine governance is in place (CI/CD builds, hashed artifacts, read-only model repo)
- [ ] Observability dashboards are live (latency histograms, GPU utilization, KV cache usage, queue time)
- [ ] Reproducibility: container versions pinned, calibration + timing caches archived, rebuild triggers documented

---

## Checklist: Before Production

Before declaring a model production-ready, verify every item:

- [ ] **Engine built in production container:** Same NGC image tag for build and deploy
- [ ] **Container version pinned:** No `:latest` tags; explicit `nvcr.io/nvidia/tensorrt:24.05-py3`
- [ ] **Explicit Q/DQ nodes verified:** Inspected in Netron or onnx-graphsurgeon (no implicit calibration in production)
- [ ] **`trtexec` confirms target precision:** INT8/FP8 kernels executing (no silent FP16 fallback — check `--dumpProfile`)
- [ ] **Polygraphy correctness gate passed:** Level 1 (ONNX→TRT FP16) and Level 2 (FP16→INT8) both pass
- [ ] **Task metric evaluated:** Top-1 / mAP / perplexity acceptable (not just numerical diff)
- [ ] **Timing cache saved:** `--timingCacheFile` stored as CI artifact for reproducible rebuilds
- [ ] **Calibration cache saved:** `calibration.cache` archived alongside the engine
- [ ] **Triton config matches engine bindings:** Tensor names, data types, and dims verified
- [ ] **Dynamic shapes + profiles:** Engine built with `--minShapes/--optShapes/--maxShapes` matching Triton's `max_batch_size`
- [ ] **KV-cache paging enabled (LLM):** `--paged_kv_cache enable` for long-context workloads
- [ ] **Triton dynamic batching configured:** `preferred_batch_size` and `max_queue_delay_microseconds` tuned
- [ ] **Triton correctness gate passed:** Local TRT output matches Triton endpoint output

---

## Consolidation: The Life of a Quantized Tensor on NVIDIA

This chapter completes the "deployment stack" arc that began with Chapter 20 (Qualcomm). Where Qualcomm targets the edge, NVIDIA targets the data center — but the quantization principles from earlier chapters apply identically. Here is how they manifest in the NVIDIA stack:

### The Pipeline

1. **ModelOpt** quantizes your model (PTQ or QAT) and exports quantized ONNX with explicit Q/DQ nodes — or produces quantized checkpoints for TRT-LLM.
2. **TensorRT** compiles the quantized model into a GPU-specific engine, auto-tuning kernels and fusing Q/DQ operations into compute kernels.
3. **TensorRT-LLM** extends this for LLMs with KV cache management, paged attention, in-flight batching, and the full quantization recipe matrix (W4A16/W8A8/FP8/FP4).
4. **Triton** serves the compiled engine in production — dynamic batching for CNNs, decoupled streaming for LLMs.
5. **cuDNN/cuBLAS** provide the low-level INT8/FP8 Tensor Core kernels that execute the "Life of a Tensor" (Chapter 4) at hardware speed.

### Canonical Failure Patterns on NVIDIA

Every failure pattern from the book's theory chapters has a concrete manifestation here:

| Book Concept | NVIDIA Manifestation | Fix |
|-------------|---------------------|-----|
| **Calibration Mismatch** (Ch. 13) | INT8 calibration cache from wrong data or model version | Re-calibrate with production data; use explicit Q/DQ for reproducibility |
| **Resolution Collapse** (Ch. 5, 14) | Transformer activation outliers crush INT8 resolution | SmoothQuant (Ch. 15) or FP8 |
| **Tail Clipping** (Ch. 5) | Entropy calibration clips rare but important values | Switch to percentile or MinMax calibration |
| **Weight-only escape** (Ch. 16) | Memory-bound decode doesn't benefit from W8A8 | W4A16 (AWQ/GPTQ) for bandwidth reduction |
| **KV Cache Memory Wall** (Ch. 18) | Long-context KV cache exceeds weight memory | KV cache FP8 or NVFP4 quantization |
| **Hardware dictates the grid** (Ch. 4) | SM capability determines available quantization types | Check compute capability table; validate via TensorRT Support Matrix |

### Critical Success Factors

- **Choose the right precision for your GPU** — INT8 for Turing/Ampere, FP8 for Hopper+, FP4 for Blackwell.
- **Use explicit Q/DQ** — implicit quantization is deprecated; explicit is portable, reproducible, and future-proof.
- **Calibrate with representative data** — bad calibration = bad accuracy (Chapter 9 applies directly).
- **Build engines on the target GPU** — engines are GPU-specific by default. Use version/hardware compatibility only when necessary.
- **Profile, profile, profile** — use trtexec, Polygraphy, and Nsight to find bottlenecks and verify correctness.
- **Use NGC containers** — build and deploy in the same container to avoid driver/CUDA/cuDNN mismatches.

When the pipeline is working correctly, INT8 models on Tensor Cores achieve 1.5–2× throughput improvement over FP16, and FP8 on Hopper achieves the same speedup with less effort. For data-center inference at scale — serving millions of requests per hour — this stack is the standard.

> In the NVIDIA stack, the "Life of a Tensor" is a series of fusions; the goal of quantization is to ensure those fusions never break.
