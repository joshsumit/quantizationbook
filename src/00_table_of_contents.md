# Table of Contents

---

## Front Matter

- [Cover](00_cover.md)
- [Title Page](00_title_page.md)
- [Copyright and License](00_copyright.md)
- [Table of Contents](00_table_of_contents.md)
- [Preface](00_preface.md)

---

## Part I — Foundations

1. [Why Quantization Exists](chapter01_why_quantization_exists.md)
	1.1 [Floating-Point Fundamentals for Quantization](chapter01_a_floating_points.md)

2. [Quantization as a Representational Constraint](chapter02_quantization_as_representational_constraint.md)
3. [Scale and Zero-Point](chapter03_scale_and_zero_point.md)
4. [Hardware Dictates the Rules](chapter04_hardware_dictates_the_rules.md)

## Part II — The Error Model

5. [Where Error Is Born](chapter05_where_error_is_born.md)
6. [The Quantized Graph](chapter06_the_quantized_graph.md)
7. [Requantization](chapter07_requantization.md)
8. [Operator Fusion](chapter08_operator_fusion.md)

## Part III — Calibration and Training

9. [Calibration and Observers](chapter09_calibration_and_observers.md)
10. [Post-Training Quantization](chapter10_post_training_quantization.md)
11. [Quantization-Aware Training](chapter11_quantization_aware_training.md)
12. [Dynamic Quantization and Mixed Precision](chapter12_dynamic_and_mixed_precision.md)

## Part IV — Failure and Recovery

13. [Failure Patterns](chapter13_failure_patterns.md)
14. [Why Transformers Break Quantization](chapter14_why_transformers_break_quantization.md)
15. [SmoothQuant and Outlier Control](chapter15_smoothquant_and_outlier_control.md)

## Part V — Modern Techniques

16. [Weight-Only and Group-Wise Quantization](chapter16_weight_only_and_groupwise_quantization.md)
17. [GPTQ, AWQ, and Offline Calibration](chapter17_gptq_awq_and_offline_calibration.md)
18. [The FP8 Revolution](chapter19_the_fp8_revolution.md)
19. [The KV-Cache Bottleneck](chapter18_the_kv_cache_bottleneck.md)

## Part VI — Hardware Stacks

20. [The Qualcomm Stack — From Training to On-Device Inference](chapter20_qualcomm_stack.md)
21. [The NVIDIA Stack — From Training to Data-Center Inference](chapter21_nvidia_stack.md)

---

## Appendices

- [Appendix A: The Quantization Toolkit Map](appendix_quantization_toolkit_map.md)
- [Appendix B: End-to-End Numeric Walkthrough](appendix_end_to_end_walkthrough.md)
- [Appendix C: Floating-Point Bit Architecture](appendix_floating_point_bit_architecture.md)
