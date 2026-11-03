# DocPilot: Ablation Table

| Day | Version / Technique | Hit@4 (Recall) | MRR | Correct Refusal | Answer Rate | p50 Latency | Notes |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Day 3** | Baseline (Dense + Naive 1000c Chunking) | 52.9% | 0.417 | 100.0% | 88.2% | 2.47s | Naive character splitting; cold API run. |
| **Day 4** | Markdown Chunker (800c) + Heading Paths | 58.8% | 0.441 | 100.0% | 94.1% | 1.01s | +5.9% recall boost via structure-aware chunking. |
| **Day 5** | Local BGE-small (384 dims, CPU) | 47.1% | 0.289 | 100.0% | 94.1% | 16.3ms | 100% offline CPU embeddings. 8x smaller vectors. |
| **Day 6** | Qdrant Vector DB + Metadata Payloads | 58.8% | 0.441 | 100.0% | 94.1% | 12.4ms | Production vector store with native metadata filtering. |
| **Day 7** | **Hybrid Search (Dense + BM25 + RRF)** | **58.8%** | **0.353** | **100.0%** | **94.1%** | **7.0ms** | **Fused keyword & semantic search. Maintained recall, slight MRR trade-off on NL queries.** |