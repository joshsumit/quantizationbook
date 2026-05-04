# Preface

Quantization is often treated as a mathematical abstraction—a set of formulas designed to bound rounding errors. While this theoretical foundation is essential, it frequently overlooks the practical complexities of deploying models on physical silicon. In a production environment, quantization is a systems engineering challenge that spans number formats, memory hierarchies, and vendor-specific hardware constraints.

This book provides a path from first principles to hardware-aware implementation. The objective is to explain the underlying mechanics of these algorithms, why they fail in specific scenarios, and how hardware actually processes low-precision data. By addressing quantization as a primary architectural constraint rather than a post-processing step, we can develop models that are both efficient and functionally robust on real-world accelerators.

The idea for this book came from a recurring frustration. I kept watching teams quantize models, observe a mysterious accuracy collapse, and then either abandon the effort or brute-force their way through it with no real understanding of what went wrong. The tooling was there. The papers were there. What was missing was a single, coherent path from first principles to deployed model — one that explained not just *what* to do, but *why* it works, *where* it breaks, and *what the hardware actually does* with the numbers you give it.

## What This Book Covers

The book is organized in six parts:

**Part I (Chapters 1–4)** builds the foundations: what quantization is, how scale and zero-point define the mapping, and why hardware constraints — not mathematical elegance — dictate the rules of the game.

**Part II (Chapters 5–8)** traces the anatomy of quantization error: where it is born, how it propagates through the operator graph, how requantization amplifies it, and how operator fusion reshapes the entire computation.

**Part III (Chapters 9–12)** covers the practitioner's core toolkit: calibration, post-training quantization, quantization-aware training, and mixed-precision strategies.

**Part IV (Chapters 13–15)** is the diagnostic section. It catalogs failure patterns, explains why transformer architectures are uniquely hostile to quantization, and walks through techniques like SmoothQuant that tame outlier distributions.

**Part V (Chapters 16–19)** covers the modern frontier: weight-only and group-wise quantization, GPTQ, AWQ, the KV-cache bottleneck in large language models, and the FP8 format that is reshaping data-center inference.

**Part VI (Chapters 20–21)** walks two real vendor stacks end to end — Qualcomm for on-device deployment, NVIDIA for data-center inference — because quantization is never finished until the model runs on actual silicon.

Two appendices close the book: a toolkit map linking every major framework to its quantization capabilities, and a full numeric walkthrough that traces a single model from floating-point training to quantized deployment.

## Who This Book Is For

This book is for engineers who deploy models. If you train models but never think about what happens after `model.eval()`, this book will change how you think about that boundary. If you already work in deployment and have been bitten by quantization failures you couldn't diagnose, this book will give you the vocabulary and the mental model to fix them.

No prior knowledge of quantization is assumed. Familiarity with deep learning fundamentals — tensors, layers, loss functions, gradient descent — is expected.

## A Note on Style

This book leads with hardware, not with math. It measures cost in picojoules and milliseconds before it ever measures cost in loss points. Every claim is grounded in specific numbers, specific formats, specific silicon. Where there is a choice between elegance and utility, utility wins.

---

*Sumit Joshi*
*2026*
