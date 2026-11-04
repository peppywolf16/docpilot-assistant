# Day 4 Sprint: Advanced Ranking, Query Rewriting & Intent-Driven Retrieval (Roadmap Days 8, 9, 10)

## Executive Summary
On Calendar Day 4, DocPilot made the leap from a basic search engine to an intelligent, multi-stage retrieval and routing system. This sprint covered three core milestones:
1. **Neural Re-ranking (Roadmap Day 8 - v0.5):** Implemented an industry-standard two-stage retrieval pipeline: broad candidate retrieval (top 20 via Hybrid Search) $\rightarrow$ deep neural re-ranking (top 4 via Cross-Encoder `ms-marco-MiniLM-L-6-v2`), combined with chunk deduplication, **Lost-in-the-Middle** context curation, and a failure analysis diagnostic suite.
2. **Pre-Retrieval Query Rewriting (Roadmap Day 9):** Built the `QueryTransform` modular architecture implementing **Multi-Query** (4 perspectives + RRF fusion), **HyDE** (Hypothetical Document Embeddings), and **Step-Back Prompting**, empirically benchmarking their quality vs. latency trade-offs.
3. **Decomposition, Filter Extraction & Routing (Roadmap Day 10):** Engineered query decomposition for multi-part questions, Pydantic-based automatic metadata filter extraction with unfiltered fallback, and an intent router that short-circuits casual greetings directly to the LLM without querying Qdrant.

---

## 1. Two-Stage Retrieval & Neural Re-ranking (Roadmap Day 8 - v0.5)

### A. The Two-Stage Pattern (`app/reranker.py` & `app/retriever.py`)
- **Stage 1 (Candidate Generation):** Hybrid Search retrieves the top 20 candidate passages fast using Qdrant and BM25.
- **Stage 2 (Neural Scoring):** A local Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) performs deep cross-attention on all 20 `(query, passage)` pairs on CPU, scoring semantic interaction and pruning down to the top 4 hyper-relevant passages.

### B. Context Curation & Lost-in-the-Middle
- **Chunk Deduplication (`deduplicate_passages`):** Jaccard word-overlap filtering (threshold: 0.85) eliminates redundant overlapping chunk windows.
- **Lost-in-the-Middle Reordering:** Position-aware reordering placing the #1 best passage first, #2 best passage last, and weaker passages in the middle to optimize LLM U-shaped attention curves during generation.

### C. Failure Analysis Diagnostics (`eval/failure_analysis.py`)
- Built an automated diagnostic script analyzing why queries missed.
- **Key Finding:** For queries like `[q03]`, the retriever successfully placed the correct document in 100% of top slots; failures were due to strict text-snippet verification rather than retrieval misses.

---

## 2. Query Rewriting Shootout (Roadmap Day 9)

### A. Modular Transformation Engine (`app/query/`)
- Abstract base class `QueryTransform` (`base.py`) and dynamic registry (`registry.py`).
- **Multi-Query (`multi_query.py`):** Generates 3 alternative query phrasings, searches across all 4 variants, merges candidates with RRF, and re-ranks against the original query.
- **HyDE (`hyde.py`):** Drafts a hypothetical documentation passage and embeds it in document space.
- **Step-Back Prompting (`step_back.py`):** Generates a high-level conceptual question to pull broader architectural context.

### B. Empirical Benchmark Shootout

| Strategy | Hit@4 (Recall) | MRR | Source-Specific MRR | p50 Latency | Extra LLM Calls | Engineering Decision |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Direct Search (No Transform)** | 52.9% | 0.407 | 0.333 | **686ms** | **0** | **Production Fast Path:** Default for low-latency queries. |
| **Multi-Query** | 52.9% | **0.446 🟢** | **0.667 🟢** | 4,033ms | 1 | **Conditional / On-Demand:** Highest ranking quality; used when initial retrieval score is low. |
| **HyDE** | 52.9% | **0.446 🟢** | **0.667 🟢** | 3,311ms | 1 | **Conditional:** Excellent conceptual grounding; reserved for abstract queries. |
| **Step-Back** | 52.9% | 0.407 | 0.333 | 2,722ms | 1 | **Drop:** Added latency without measurable gain on technical documentation. |

---

## 3. Decomposition, Filter Extraction & Routing (Roadmap Day 10)

### A. Query Decomposition (`app/query/decomposition.py`)
- Uses a few-shot prompt with Pydantic validation (`DecomposedQueries`) to break complex multi-part questions into 2–3 independent sub-questions.
- Retrieves candidate pools for each sub-question independently, merges them with RRF, and re-ranks against the original query.
- **Result:** Multi-doc bucket Hit@4 reached **75.0%** and MRR reached **0.750**!

### B. Pydantic Metadata Filter Extraction (`app/query/filter_extractor.py`)
- Analyzes natural language questions for explicit document references (e.g., *"In extra-models.md, how do I hide passwords?"*).
- Extracts `doc_filter="extra-models.md"` and cleans the query for Qdrant.
- **Safety Fallback:** If a filtered search returns fewer than 2 chunks, it automatically retries an unfiltered search so the user never gets an empty result.
- **Result:** Source-specific bucket Hit@4 reached **100.0%**!

### C. Intent Router (`app/query/router.py`)
- Classifies user inputs into:
  - `direct_chat`: Casual greetings ("Hi", "Who are you?") $\rightarrow$ answered directly by Gemini without searching Qdrant.
  - `search`: Standard technical query $\rightarrow$ fast two-stage search.
  - `decompose`: Multi-part technical query $\rightarrow$ sub-query decomposition.

---

## 4. Cumulative Ablation Table (Full Progress to Date)

| Sprint / Day | Version / Milestone | Hit@4 (Recall) | MRR | Correct Refusal | Answer Rate | p50 Latency | Core Takeaway |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **Day 1** | Naive Baseline (1000c + NumPy) | 52.9% | 0.417 | 100.0% | 88.2% | 2.47s | Initial working prototype; exposed quota and chunking bugs. |
| **Day 2** | Markdown Chunker (800c) + Disk Persistence | 58.8% | 0.441 | 100.0% | 94.1% | 1.01s | +5.9% recall boost via structure-aware heading injection. |
| **Day 3** | BGE Embeddings & Qdrant Vector DB | 58.8% | 0.441 | 100.0% | 94.1% | 12.4ms | Production vector database with metadata payload filtering. |
| **Day 4** | **Re-ranking, Query Rewriting & Decomposition (v0.5)** | **47.1% - 58.8%** | **0.446** | **100.0%** | **94.1%** | **686ms - 2.7s** | **Two-Stage retrieval + Cross-Encoder. Multi-doc MRR hit 0.750, Source-specific hit 100%.** |

---

## 5. Current Repository Layout

```text
docpilot/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py           # Config (QUERY_TRANSFORM_STRATEGY, EMBEDDER_TYPE, etc.)
│   │   ├── llm.py              # Gemini embedding/generation with diskcache & backoff
│   │   ├── loader.py           # Multi-format text extractor (.md, .txt, .pdf)
│   │   ├── store.py            # Adapter routing to Qdrant & Smart Search
│   │   ├── store_qdrant.py     # Qdrant client, metadata payloads & payload filtering
│   │   ├── retriever.py        # Smart Two-Stage Search: Hybrid + Decomposition + Re-ranking
│   │   ├── reranker.py         # Cross-Encoder neural re-ranker, deduplication & Lost-in-the-Middle
│   │   ├── main.py             # FastAPI backend with intent routing, doc management & filters
│   │   ├── chunking/           # Modular chunking package (Base, Cleaner, Markdown, Registry)
│   │   ├── embeddings/         # Modular embeddings package (Base, Gemini, BGE, Registry)
│   │   └── query/              # Pre-retrieval intelligence package
│   │       ├── __init__.py
│   │       ├── base.py         # QueryTransform abstract base class
│   │       ├── multi_query.py  # Multi-Query rephrasings
│   │       ├── hyde.py         # Hypothetical Document Embeddings
│   │       ├── step_back.py    # Step-Back conceptual prompting
│   │       ├── decomposition.py# Multi-part query decomposition
│   │       ├── filter_extractor.py # Pydantic automatic filter extractor & fallback
│   │       ├── router.py       # Intent router (direct_chat, search, decompose)
│   │       └── registry.py     # Query transform factory
│   ├── data/
│   │   ├── index/              # Model-specific Qdrant persistent storage folders
│   │   └── sample_docs/        # 5 official FastAPI Markdown tutorial files
│   ├── eval/
│   │   ├── questions.jsonl     # 20 gold evaluation benchmark questions
│   │   ├── metrics.py          # Hit@k, MRR, and Abstention accuracy formulas
│   │   ├── run_eval.py         # Automated evaluation runner with pacing & per-bucket breakdown
│   │   └── failure_analysis.py # Diagnostic tool isolating retrieval vs. ranking failures
│   ├── results/
│   │   ├── day1.md             # Day 1 logbook
│   │   ├── day2.md             # Day 2 logbook (Foundations & Chunking)
│   │   ├── day3.md             # Day 3 logbook (Embeddings & Qdrant)
│   │   ├── day4.md             # This Day 4 logbook (Re-ranking, Query Rewriting & Decomposition)
│   │   ├── ablation.md         # Official cumulative ablation table
│   │   └── runs/               # Timestamped JSON evaluation audit reports
│   ├── scripts/
│   │   └── ingest_folder.py    # CLI tool to bulk-index document folders
│   ├── .cache/                 # Persistent disk cache for embeddings and prompts
│   ├── .env                    # Protected API keys & model configs
│   └── requirements.txt
└── frontend/
    └── app.py                  # Streamlit UI with doc management, filters & citations