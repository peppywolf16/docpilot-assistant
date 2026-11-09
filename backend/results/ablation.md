# DocPilot: Ablation Table

| Day | Version / Technique | Hit@4 (Recall) | MRR | Correct Refusal | Answer Rate | p50 Latency | Notes |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Day 3** | Baseline (Dense + Naive 1000c Chunking) | 52.9% | 0.417 | 100.0% | 88.2% | 2.47s | Naive character splitting; cold API run. |
| **Day 4** | Markdown Chunker (800c) + Heading Paths | 58.8% | 0.441 | 100.0% | 94.1% | 1.01s | +5.9% recall boost via structure-aware chunking. |
| **Day 5** | Local BGE-small (384 dims, CPU) | 47.1% | 0.289 | 100.0% | 94.1% | 16.3ms | 100% offline CPU embeddings. 8x smaller vectors. |
| **Day 6** | Qdrant Vector DB + Metadata Payloads | 58.8% | 0.441 | 100.0% | 94.1% | 12.4ms | Production vector store with native metadata filtering. |
| **Day 7** | Hybrid Search (Dense + BM25 + RRF) | 58.8% | 0.353 | 100.0% | 94.1% | 5.8ms | Fused keyword & semantic search. |
| **Day 8** | Two-Stage Pipeline (Hybrid 20 -> Rerank 4) | 52.9% | 0.407 | 100.0% | 94.1% | 686ms | Cross-Encoder neural re-ranking + Lost-in-the-Middle. |
| **Day 9** | Query Rewriting Shootout (Multi-Query / HyDE) | 52.9% | 0.446 | 100.0% | 94.1% | 3.3s - 4.0s | MRR reached all-time high (0.446) with 5x latency cost. |
| **Day 10** | Decomposition & Filter Extraction | 47.1% | 0.402 | 100.0% | 94.1% | 2.75s | Multi-doc MRR hit 0.750, Source-specific hit 100%. |
| **Day 11** | Parent-Child Retrieval (1200c / 250c) | 41.2% | 0.324 | 100.0% | 100.0% | 292.7ms | 239 child vectors. Ultra-fast search, but 250c chunks fractured code. |
| **Day 12** | Verified Citations & Abstention (PROMPT_V2) | 41.2% | 0.324 | 100.0% | 94.1% | 1.48s | Pydantic structured output. 100% citation validity rate & fast refusal. |
| **Day 13** | Streaming & Polished UI v1.0 | 41.2% | 0.324 | 100.0% | 94.1% | Real-time SSE | Server-Sent Events token streaming to Streamlit. |
| **Day 14** | LangGraph Agent with Self-Correction (CRAG) | -- | -- | 100.0% | 100.0% | Stateful | State machine with retrieval grading and query rewrite loop. |
| **Day 15** | Memory, Follow-Up Condensation & Self-RAG | -- | -- | 100.0% | 100.0% | Stateful | Multi-turn conversation memory + Self-RAG answer verification. |
| **Day 16** | Security: JWT RBAC, PII Masking & Injection Shield | -- | -- | 100.0% | 100.0% | Fast | 100% security test pass rate (5/5 in pytest). |
| **Day 17** | **Production Hardening: Caching & Rate Limits** | **--** | **--** | **100.0%** | **100.0%** | **Instant** | **Tenant-safe disk cache (passed cache-leak test) + slowapi rate limiting (20/min).** |