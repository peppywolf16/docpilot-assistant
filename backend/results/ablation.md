# DocPilot: Ablation Table

| Day | Version / Technique | Hit@4 (Recall) | MRR | Correct Refusal | Answer Rate | p50 Latency | Notes |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Day 3** | Baseline (Dense + Naive 1000c Chunking) | 52.9% | 0.417 | 100.0% | 88.2% | 2.47s | Cold baseline with disk cache & naive slicing. |
| **Day 4** | **Structure-Aware Markdown Chunker (800c)** | **58.8%** | **0.441** | **100.0%** | **94.1%** | **1.01s** | **+5.9% recall! Heading paths prepended & code blocks kept intact.** |