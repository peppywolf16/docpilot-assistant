# Framework Comparison: Custom Architecture vs. LlamaIndex

## Executive Summary
On Day 19, we ported the DocPilot pipeline to **LlamaIndex** (`VectorStoreIndex` with Gemini embeddings and chat models) inside `backend/experiments/llamaindex/` and evaluated both architectures against our frozen 20-question benchmark.

---

## 1. Quantitative Benchmark Results

| Metric | DocPilot Custom Engine | LlamaIndex Framework Build | Delta |
|:---|:---:|:---:|:---:|
| **Hit@4 (Recall)** | 58.8% | **70.6%** | +11.8% (LlamaIndex) |
| **MRR (Mean Reciprocal Rank)** | 0.441 | **0.510** | +0.069 (LlamaIndex) |
| **Median Retrieval Latency (p50)** | 686.0ms | **586.5ms** | ~100ms faster |
| **How-To Bucket Hit@4** | 40.0% | **80.0%** | +40.0% (LlamaIndex) |
| **Multi-Doc Bucket Hit@4** | 100.0% | **100.0%** | Tie |
| **Source-Specific Hit@4** | 100.0% | **100.0%** | Tie |
| **Security Test Pass Rate** | **100% (6/6 in pytest)** | N/A (No built-in ACLs) | DocPilot wins |
| **Citation Verification Rate** | **100% in code** | Unverified text brackets | DocPilot wins |

---

## 2. Qualitative Architectural Analysis

### A. Developer Velocity & Code Volume
- **LlamaIndex:** Required only ~30 lines of code (`SimpleDirectoryReader` -> `VectorStoreIndex` -> `as_query_engine`). Ideal for rapid prototyping and zero-to-one builds.
- **Custom Engine:** Required ~180 lines across `loader.py`, `chunking/`, `store_qdrant.py`, and `retriever.py`.

### B. Transparency & Upstream Fragility
- **LlamaIndex:** Heavily abstracted. Running the pipeline threw deprecation warnings (`Google GenerativeAI SDK deprecated`, `Gemini class deprecated`). Debugging a failed query requires stepping through 10+ internal stack frames.
- **Custom Engine:** Zero magic. Every mathematical dot product, BM25 score, and RRF formula is visible, unit-testable, and directly modifiable in native Python/NumPy.

### C. Enterprise Customization (Security & Control)
- **LlamaIndex:** Adding multi-tenancy ACL filters (`tenant == user.tenant`), deterministic Python citation quote verification, and prompt injection fences requires writing custom query transforms or subclassing base classes.
- **Custom Engine:** Built natively from Day 1. Security filters are first-class parameters in Qdrant and BM25 search calls.

---

## 3. Engineering Conclusion
- **When to use LlamaIndex:** For early-stage POCs, generic knowledge bases, and standard document search where developer velocity is the #1 priority.
- **When to use Custom Architecture (DocPilot):** For enterprise applications requiring strict multi-tenant access control (RBAC), deterministic citation verification, real-time Server-Sent Events streaming, and autonomous multi-agent state machines (LangGraph).