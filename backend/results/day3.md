# Day 3: Advanced Retrieval – Embedding Shootout, Qdrant Vector DB & Hybrid Search (RRF)

## Executive Summary
Completed a massive architectural sprint covering three advanced retrieval milestones:
1. **Embedding Model Shootout (Day 5):** Built a unified `Embedder` interface and benchmarked commercial API embeddings (`gemini-embedding-001`, 3072 dims) against local open-source CPU embeddings (`BAAI/bge-small-en-v1.5`, 384 dims), discovering the optimal trade-off between semantic recall and offline zero-cost execution.
2. **Qdrant Vector Database & Metadata Filtering (Day 6):** Upgraded storage from NumPy/JSON to **Qdrant** running in local persistent mode, attached rich metadata payloads (`doc_name`, `version`, `heading_path`), implemented metadata filtering, and added a document selector dropdown to the Streamlit UI.
3. **Hybrid Search & RRF (Day 7):** Implemented **Hybrid Search** combining Qdrant dense vector search with sparse **BM25** keyword matching via **Reciprocal Rank Fusion (RRF)**, and added per-bucket evaluation telemetry (`Lookup`, `How_to`, `Multi_doc`, `Source_specific`).

---

## 1. Embedding Model Shootout (Day 5)

### A. The Unified `Embedder` Interface (`app/embeddings/`)
- Created an abstract base class `Embedder` (`base.py`) enforcing standardized `.embed_documents()` and `.embed_query()` methods.
- **Gemini Embedder (`gemini.py`):** Wraps Google’s API with optional Matryoshka dimension truncation.
- **BGE Embedder (`bge.py`):** Runs `BAAI/bge-small-en-v1.5` locally on CPU via `sentence-transformers`, automatically applying the required query instruction prefix (`"Represent this sentence for searching..."`).
- **Registry (`registry.py`):** Factory allowing instant model switching via `EMBEDDER_TYPE` in `config.py`.

### B. Empirical Benchmark Comparison
We indexed all 85 chunks under model-specific index folders (`data/index/gemini_full/` vs. `data/index/bge-small_full/`) and evaluated them using `--retrieval-only`:

| Embedder | Vector Dimension | Vector Size / Chunk | Hit@4 (Recall) | MRR | Latency (p50) | Key Trade-Off |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Gemini** | 3,072 dims | ~12.3 KB | **58.8%** | **0.441** | 0.0ms (cached) | Higher semantic accuracy, requires API key & internet quota. |
| **BGE-small** | 384 dims | ~1.5 KB | **47.1%** | **0.289** | 16.3ms | **100% local CPU, offline, free, 8x smaller vector storage.** |

*Decision:* Retained **Gemini** as our primary production embedder for maximum semantic recall on technical documentation.

---

## 2. Production Vector Store & Metadata Filtering (Day 6)

### A. Qdrant Integration (`app/store_qdrant.py`)
- Replaced custom NumPy arrays and JSON files with **Qdrant** running in local persistent mode (`data/index/.../qdrant_storage`).
- Used stable UUIDs for point IDs to prevent collision bugs during document re-ingestion/deletion.
- Added clean lifecycle shutdown handling via `atexit.register(_close_client)` to prevent SQLite file locks.

### B. Metadata Payload Filtering
- Stored rich metadata payloads alongside vectors: `doc_name`, `chunk_id`, `heading_path`, `position`, and `version`.
- Implemented payload filtering (`doc_filter`, `version_filter`) using Qdrant’s native `models.Filter` and `FieldCondition`.

### C. UI Filter Dropdown (`frontend/app.py`)
- Added a **"🔍 Search Filters"** selectbox in the Streamlit sidebar, allowing users to optionally restrict searches to a single document (e.g., searching only within `path-params.md`).

---

## 3. Hybrid Search & Reciprocal Rank Fusion (Day 7)

### A. Hybrid Retriever (`app/retriever.py`)
- **Dense Search:** Queries semantic similarity via Qdrant.
- **Sparse Search:** Tokenizes corpus and queries exact keyword matches via **BM25Okapi** (`rank-bm25`).
- **Reciprocal Rank Fusion (RRF):** Fuses both ranked lists mathematically without needing score normalization:
  $$\text{RRF Score} = \sum \frac{1}{60 + \text{rank}}$$

### B. Per-Bucket Telemetry (`eval/run_eval.py`)
Upgraded the evaluation runner to output a per-bucket performance breakdown across question archetypes:

*Finding:* While Hybrid Search is essential for exact identifiers and error codes, pure dense vector search with Gemini achieved slightly higher top-1 ranking (MRR 0.441 vs 0.353) on natural language technical questions.

---

## 4. Cumulative Ablation Table (Day 1 to Today)

| Day / Sprint | Version / Technique | Hit@4 (Recall) | MRR | Correct Refusal | Answer Rate | p50 Latency | Key Takeaway |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Day 1** | Naive Baseline (1000c + NumPy) | 52.9% | 0.417 | 100.0% | 88.2% | 2.47s | Initial working prototype; exposed chunking and quota flaws. |
| **Day 2** | Markdown Chunker (800c) + Heading Paths | 58.8% | 0.441 | 100.0% | 94.1% | 1.01s | +5.9% recall boost via structure-aware chunking & persistence. |
| **Day 3** | **Embeddings, Qdrant & Hybrid Search** | **58.8%** | **0.353** | **100.0%** | **94.1%** | **5.8ms** | **Production vector store (Qdrant), metadata filtering & BM25 hybrid search.** |

---

## 5. Current Repository Layout

```text
docpilot/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py           # Centralized configuration & embedder settings
│   │   ├── llm.py              # Gemini embedding/generation with diskcache & backoff
│   │   ├── loader.py           # Multi-format text extractor (.md, .txt, .pdf)
│   │   ├── store.py            # Store adapter routing to Qdrant
│   │   ├── store_qdrant.py     # Qdrant client, metadata payloads & payload filtering
│   │   ├── retriever.py        # Dense + BM25 Hybrid Search with RRF
│   │   ├── main.py             # FastAPI backend with filters & doc management
│   │   └── chunking/           # Modular chunking package (Base, Cleaner, Markdown, Registry)
│   ├── data/
│   │   ├── index/              # Model-specific Qdrant persistent storage folders
│   │   └── sample_docs/        # 5 official FastAPI Markdown tutorial files
│   ├── eval/
│   │   ├── questions.jsonl     # 20 gold evaluation benchmark questions
│   │   ├── metrics.py          # Hit@k, MRR, and Abstention accuracy formulas
│   │   └── run_eval.py         # Automated evaluation runner with per-bucket telemetry
│   ├── results/
│   │   ├── day1.md             # Day 1 logbook
│   │   ├── day2.md             # Day 2 logbook (Foundations & Chunking)
│   │   ├── day3.md             # This Day 3 logbook (Embeddings, Qdrant & Hybrid)
│   │   ├── ablation.md         # Official cumulative ablation table
│   │   └── runs/               # Timestamped JSON evaluation audit reports
│   ├── scripts/
│   │   └── ingest_folder.py    # CLI tool to bulk-index document folders
│   ├── .cache/                 # Persistent disk cache for embeddings and prompts
│   ├── .env                    # Protected GEMINI_API_KEY & EMBEDDER_TYPE
│   └── requirements.txt
└── frontend/
    └── app.py                  # Streamlit UI with doc management, filters & citations