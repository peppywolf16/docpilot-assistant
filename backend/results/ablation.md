# DocPilot: Ablation Table

| Day | Version / Technique | Hit@4 (Recall) | MRR | Correct Refusal | Answer Rate | p50 Latency | Notes |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Day 3** | Baseline (Dense + Naive 1000c Chunking) | 52.9% | 0.417 | 100.0% | 88.2% | 2.47s | Naive character splitting; cold API run. |
| **Day 4** | Markdown Chunker (800c) + Heading Paths | 58.8% | 0.441 | 100.0% | 94.1% | 1.01s | +5.9% recall boost via structure-aware chunking. |
| **Day 5** | Local BGE-small (384 dims, CPU) | 47.1% | 0.289 | 100.0% | 94.1% | 16.3ms | 100% offline CPU embeddings. 8x smaller vectors. |
| **Day 6** | Qdrant Vector DB + Metadata Payloads | 58.8% | 0.441 | 100.0% | 94.1% | 12.4ms | Production vector store with native metadata filtering. |
| **Day 7** | Hybrid Search (Dense + BM25 + RRF) | 58.8% | 0.353 | 100.0% | 94.1% | 5.8ms | Fused keyword & semantic search. |
| **Day 8** | Two-Stage Pipeline (Hybrid 20 -> Rerank 4) | 52.9% | 0.407 | 100.0% | 94.1% | 686ms | Cross-Encoder neural re-ranking + Lost-in-the-Middle. |
| **Day 9** | **Query Rewriting Shootout (Multi-Query / HyDE)** | **52.9%** | **0.446** | **100.0%** | **94.1%** | **3.3s - 4.0s** | **MRR reached all-time high (0.446) and Source-Specific MRR hit 0.667. Trade-off: 5x higher latency.** |