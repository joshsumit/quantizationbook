# Appendix A: The Quantization Toolkit Map

## From Theory to Code

This book teaches quantization as a system — grids, scales, boundaries, error, calibration, and strategies. But deploying a quantized model requires choosing a tool. The ecosystem is fragmented: different tools target different hardware, support different algorithms, and optimize for different workloads.

This appendix maps the landscape so you know where to start.

---

## The Decision Matrix

Before consulting the matrix: **identify your primary bottleneck.** Is inference memory-bound (LLM decode, batch=1) or compute-bound (CNN, large batch)? Chapter 16's table determines the answer. The bottleneck dictates which row matters.

| Your Workload | Your Hardware | Start With | Algorithm | Format |
|---|---|---|---|---|
| CNN inference (edge) | Mobile NPU / DSP | TensorFlow Lite, ONNX Runtime | PTQ (int8) | TFLite FlatBuffer, ONNX |
| CNN inference (server) | NVIDIA GPU | TensorRT | PTQ or QAT (int8) | TensorRT engine |
| LLM serving (single GPU) | NVIDIA GPU (Ampere+) | vLLM, TensorRT-LLM | GPTQ, AWQ (int4) | safetensors, GGUF |
| LLM serving (multi-GPU) | NVIDIA GPU (H100+) | TensorRT-LLM, vLLM | FP8, AWQ | safetensors (with FP8 scales) |
| LLM local / desktop | CPU or consumer GPU | llama.cpp, ollama | GGUF K-quants (weight-only schemes) | GGUF |
| Research / prototyping | Any GPU | PyTorch (torchao) | PTQ, GPTQ, AWQ | safetensors / PyTorch checkpoints |
| Fine-tuning with quant | NVIDIA GPU | bitsandbytes + PEFT | QLoRA (NF4) | bitsandbytes format |
| Training in FP8 | H100+ | Transformer Engine | FP8 mixed precision | framework checkpoint (with FP8 scales) |

---

## Tool Profiles

### PyTorch Native Quantization (torchao)

**What it does:** PTQ and QAT for PyTorch models. Supports int8 dynamic, int8 static (with calibration), and int4 weight-only quantization. The `torchao` library is the modern replacement for the older `torch.quantization` API.

**When to use:** Prototyping quantization, research, or when you need fine-grained control over the quantization graph. Best starting point for understanding what calibration and observers do in practice.

**Limitations:** Primarily a quantization toolkit, not an inference runtime. Produces quantized PyTorch models that can be exported to other runtimes (ONNX, TensorRT); deployment-grade performance usually requires exporting to a dedicated runtime.

**Decision guidance:** Choose torchao when you are *developing* a quantization strategy, not deploying one. It is the right tool for answering "which layers are sensitive?" and "does PTQ work for my model?" before committing to a production runtime. If you find yourself writing custom observers or debugging per-layer accuracy drops, torchao gives you the visibility that compiled runtimes hide. Once you have a working quantization recipe, export to TensorRT or ONNX Runtime for production speed.

### TensorRT / TensorRT-LLM

**What it does:** NVIDIA's inference optimizer. Takes a trained model, applies graph optimizations (including fusion from Chapter 8), and compiles it for specific NVIDIA GPU architectures. TensorRT-LLM extends this for large language models with paged attention, KV-cache quantization, and multi-GPU tensor parallelism.

**When to use:** Production deployment on NVIDIA GPUs. Delivers the highest throughput for int8, int4, and FP8 on NVIDIA hardware.

**Limitations:** NVIDIA-only. Model conversion can be opaque. Debugging quantization issues requires profiling tools (Nsight Systems).

**Decision guidance:** TensorRT is the right choice when latency and throughput are the primary metrics and you are deploying exclusively on NVIDIA GPUs. It applies aggressive graph optimizations (fusion, kernel selection, memory planning) that other runtimes cannot match on NVIDIA silicon. The trade-off is opacity: when quantization accuracy drops, TensorRT's internal decisions about fusion patterns and precision selection are difficult to inspect. Use torchao or ONNX Runtime to debug, then deploy with TensorRT. For LLMs specifically, TensorRT-LLM adds paged KV-cache management and multi-GPU tensor parallelism — features that vLLM also provides but with different performance characteristics. Choose TensorRT-LLM over vLLM when you need maximum single-request latency; choose vLLM when you need maximum multi-request throughput with simpler deployment.

*Common failure patterns: Silent Fallback (unsupported ops fall to float), Fusion Loss (unexpected boundary materialization). Profile with Nsight Systems to detect both.*

### vLLM

**What it does:** High-throughput LLM serving framework. Supports GPTQ, AWQ, and FP8 quantized models. Implements paged attention for KV-cache memory management and continuous batching for throughput.

**When to use:** Serving LLMs in production with high concurrency. Supports quantized models from multiple sources (GPTQ, AWQ, bitsandbytes).

**Limitations:** Primarily targets NVIDIA GPUs (AMD ROCm support varies by version). Not designed for edge deployment or CNNs.

**Decision guidance:** vLLM excels at *throughput under concurrency* — many users hitting the same model simultaneously. Its continuous batching engine dynamically groups requests, which matters more than raw single-request latency for most serving scenarios. Choose vLLM when you serve quantized LLMs to many concurrent users and want broad algorithm compatibility (GPTQ, AWQ, FP8 models all load without conversion). Avoid vLLM for edge deployment, CNN models, or scenarios where you need to customize the quantization pipeline — it is an inference server, not a quantization toolkit.

*Common failure patterns: Dequantization Bottleneck (int4 unpack overhead on some kernels), KV-cache bottleneck at long context lengths (Chapter 18).*

### llama.cpp / GGUF

**What it does:** CPU and GPU inference for LLMs using custom quantization formats. The GGUF format supports a range of quantization types — from Q2_K (2-bit with super-blocks) to Q8_0 (8-bit) — with "K-quants" that quantize the quantization scales themselves (super-blocks).

**When to use:** Running LLMs on consumer hardware — laptops, desktops, Mac M-series. The K-quant formats (Q4_K_M, Q5_K_S) provide excellent accuracy-per-bit through importance-aware mixed quantization.

**Limitations:** Custom formats not interoperable with other runtimes. Optimized for batch-size-1 generation, not high-throughput serving.

**Decision guidance:** llama.cpp is the right tool when your deployment target is a consumer device — a laptop CPU, a Mac with Metal, or a desktop GPU with limited VRAM. Its K-quant formats (Q4_K_M is the most popular) apply mixed quantization automatically: important weight groups get more bits, less important groups get fewer. This produces better accuracy-per-bit than uniform int4. The GGUF format is self-contained — one file holds weights, scales, metadata, and tokenizer config. The trade-off: GGUF models are not portable to other runtimes. A model quantized for llama.cpp cannot be served by vLLM without re-quantization from the original weights. Choose llama.cpp for local/offline inference; choose vLLM or TensorRT-LLM for datacenter serving.

*Common failure patterns: memory bandwidth saturation (CPU inference is purely bandwidth-bound), CPU vectorization limits on older hardware.*

### bitsandbytes

**What it does:** GPU-accelerated NF4 and int8 quantization for PyTorch, designed for QLoRA fine-tuning. Loads models in 4-bit NF4 (Chapter 17) with per-block double quantization.

**When to use:** Fine-tuning large models on limited GPU memory. A 70B model in NF4 fits on a single 48GB GPU for LoRA fine-tuning.

**Limitations:** Not an inference optimizer. Inference throughput is lower than dedicated serving frameworks.

**Decision guidance:** bitsandbytes solves one problem well: fitting a model into GPU memory for fine-tuning. It is not a serving solution and should not be used as one. If you are deploying a model for inference, use GPTQ or AWQ to quantize the weights and serve with vLLM or TensorRT-LLM. If you are fine-tuning a model that does not fit in GPU memory, bitsandbytes + LoRA is the standard approach. After fine-tuning, export the LoRA adapter, merge it with the base model, and re-quantize with a production-grade algorithm for serving.

### ONNX Runtime

**What it does:** Cross-platform inference runtime. Supports int8 quantization (PTQ and QAT) with CPU and GPU execution providers. Exports quantized models in the ONNX format.

**When to use:** Cross-platform deployment where the same model must run on different hardware (CPU, NVIDIA GPU, Intel, ARM). Good for CNN and vision model deployment.

**Limitations:** LLM support is less mature than TensorRT-LLM or vLLM. Quantization options are more limited.

**Decision guidance:** ONNX Runtime is the right choice when the same quantized model must run on multiple hardware targets without re-quantization. A single ONNX model can execute on Intel CPUs, ARM devices, NVIDIA GPUs, and Apple Silicon through different execution providers. This portability comes at a cost: the quantization is less aggressively optimized than hardware-specific tools (TensorRT on NVIDIA, CoreML on Apple). For CNNs and vision models deployed across heterogeneous infrastructure, ONNX Runtime is often the pragmatic choice. For LLMs, prefer vLLM or TensorRT-LLM — ONNX Runtime's LLM quantization support lacks paged attention, continuous batching, and other serving optimizations.

*Common failure patterns: Silent Fallback (execution provider may not cover all quantized ops), Fusion Loss (less aggressive fusion than vendor-specific runtimes).*

### NVIDIA Transformer Engine

**What it does:** Enables FP8 mixed-precision training and inference for transformers. Automatically manages FP8 scaling factors and loss scaling.

**When to use:** Training large transformers in FP8 on H100+ hardware (Chapter 19). Inference with FP8 precision.

**Limitations:** H100+ only. Training-focused — for inference-only FP8, TensorRT-LLM is more optimized.

**Decision guidance:** Transformer Engine is specialized: FP8 training and inference on Hopper+ GPUs. If you are training a model from scratch or doing full fine-tuning (not LoRA) on H100s, Transformer Engine's automatic FP8 scaling can substantially reduce training time (often reported as 30–50%, depending on model and recipe). For inference, it integrates with TensorRT-LLM for FP8 serving. Do not use it for int4/int8 quantization — it does not support integer formats. Do not use it on pre-Hopper hardware — FP8 tensor core instructions do not exist before H100.

---

## Other Ecosystem Tools (Brief Mentions)

- **OpenVINO** (Intel): Quantization toolkit + inference runtime for Intel CPUs, iGPUs, and VPUs. Strong int8 PTQ support with post-training optimization toolkit (POT/NNCF).
- **Core ML** (Apple): On-device deployment for iOS/macOS. Supports int8 and palettization (weight clustering). Primary path for quantized models on Apple hardware.
- **ExecuTorch** (Meta): Mobile deployment for PyTorch models. Successor to PyTorch Mobile with quantization-aware export.
- **Triton Inference Server** (NVIDIA): Serving orchestrator — not a quantization tool itself, but hosts TensorRT-LLM and vLLM backends.
- **TGI** (Hugging Face): Text Generation Inference server. Supports GPTQ/AWQ/bitsandbytes models with a serving API.

---

## The Format Confusion

The same quantized model can be described by three independent labels (Chapter 17):

| Label | Examples | What It Describes |
|---|---|---|
| **Algorithm** | GPTQ, AWQ, RTN, SmoothQuant | How the integer values were chosen |
| **Format** | GGUF, safetensors, GPTQ-format, AWQ-format | How the data is stored on disk |
| **Runtime** | llama.cpp, vLLM, TensorRT-LLM, ExLlamaV2 | What executes the model |

Common valid combinations:
- AWQ (algorithm) → safetensors (format) → vLLM (runtime)
- GPTQ (algorithm) → GPTQ-format (format) → ExLlamaV2 (runtime)
- GGML k-quants (algorithm) → GGUF (format) → llama.cpp (runtime)

Invalid assumption: "I'm using GPTQ" does not tell you the format or the runtime. A model "quantized with GPTQ" can be served by vLLM, ExLlamaV2, or TensorRT-LLM — each using different kernels with different performance characteristics.

---

## A Practical Starting Workflow

For a practitioner quantizing a model for the first time:

1. **Identify the bottleneck.** Is inference memory-bound (LLM decode, batch=1) or compute-bound (CNN, large batch)? Chapter 16's table determines the answer.

2. **Check hardware support.** What formats does your target hardware accelerate? Int8? FP8? Int4 with dequantize kernels? Chapter 4's capability envelope applies.

3. **Start with PTQ.** Try post-training quantization (Chapter 10) first. It costs minutes. If accuracy is acceptable, deploy.

4. **If PTQ fails, diagnose.** Use Chapter 13's diagnostic order. Is it outlier explosion? Try SmoothQuant (Chapter 15) or FP8 (Chapter 19). Is it calibration drift? Fix the calibration data (Chapter 9). Is it per-layer sensitivity? Try mixed precision (Chapter 12).

5. **For LLMs, default to weight-only int4.** Use GPTQ or AWQ (Chapter 17) with group size 128. Compare accuracy against the float16 baseline.

6. **Profile before celebrating.** A model that is "quantized" but hits silent fallback (Chapter 13) or decompression bottlenecks (Chapter 17) may be slower than float16. Measure latency, not just accuracy.

---

## Conceptual Consolidation

The quantization ecosystem is fragmented but navigable. The key is separating algorithm, format, and runtime — and choosing based on your workload, your hardware, and your accuracy requirements. No single tool covers all scenarios. The decision matrix above is a starting point; the chapters in this book provide the understanding to diagnose when the chosen tool produces unexpected results.

**If you only remember one thing:** pick runtime first (what hardware you deploy on), then algorithm (what accuracy you need), then format (what interoperability you require).
