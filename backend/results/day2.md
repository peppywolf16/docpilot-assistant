# Day 2: The Core RAG Engine – Persistence, Evaluation Harness & Structure-Aware Chunking

## Executive Summary
On Day 2, the project evolved from a volatile in-memory prototype into a modular, persistent, scientifically measured, and structure-aware RAG platform. Today's sprint covered three major engineering milestones:
1. **Solid Foundations:** Replaced in-memory state with on-disk binary vector persistence (`vectors.npy` / `chunks.json`), added native PDF parsing, implemented document replacement/deduplication, and built a document management UI with deletion controls.
2. **Evaluation Harness & Zero-Cost Caching:** Eliminated "vibe-checking" by building a persistent disk cache (slashing repeated query latency from 2.09s to 0.0000s), constructing a 20-question gold benchmark suite, and measuring the mathematical baseline (**52.9% Hit@4, 0.417 MRR**).
3. **Structure-Aware Chunking:** Slashed retrieval failure by building an atomic markdown chunker that preserves code blocks and prepends heading hierarchies (`[Section: H1 > H2]`), **measurably boosting Hit@4 recall from 52.9% to 58.8%**.

---

## 1. Modularization & Persistence (Restart-Safety)

### A. Modular Codebase Architecture
Refactored the Day 1 monolithic `rag.py` into single-responsibility modules:
- **`app/config.py`**: Centralized configuration reading from `.env` (paths, models, chunk size, top-k, chunker strategy).
- **`app/loader.py`**: Universal text extraction supporting `.md`, `.txt`, and `.pdf` (via `pypdf`), capable of reading from local file paths or raw HTTP upload bytes.
- **`app/store.py`**: High-performance persistent storage engine using native NumPy binary matrices (`vectors.npy`) and JSON metadata (`chunks.json`).

### B. Persistent On-Disk Vector Store (`app/store.py`)
- **Zero-Cost Restart-Safety:** Index state is saved to `data/index/` after every mutation. On backend startup, vectors and chunks load from disk in under **10ms** with **0 API calls**, completely eliminating the 148 RPM quota spikes observed yesterday.
- **Deduplication / Replacement:** Re-uploading an existing file automatically deletes its stale chunks and vector rows before inserting the new ones.
- **Document Deletion:** Added `delete_document(doc_name)` which removes chunks from the JSON list and slices the corresponding rows out of the NumPy matrix.

### C. Backend API & Ingestion Scripts
- Added `GET /documents` to list all indexed document filenames.
- Added `DELETE /documents/{name}` to delete a document and its vectors from memory and disk.
- Upgraded `POST /ingest` to handle multi-file uploads including `.pdf`.
- Built `scripts/ingest_folder.py` to bulk-load documentation folders in a single command.

### D. Streamlit UI v0.2 (`frontend/app.py`)
- **Indexed Document Drawer:** Sidebar displaying all currently indexed files with one-click trash (`🗑️`) delete buttons.
- **PDF Uploading:** Enabled `.pdf` file uploads directly from the UI.
- **Dynamic Metrics:** Live chip showing indexed chunks and document count.

---

## 2. Evaluation Harness & Disk Caching (Eliminating Guesswork)

### A. Persistent Disk Caching (`app/llm.py`)
- Integrated `diskcache` around `embed()` and `generate()`.
- Caches requests on disk (`.cache/`) keyed by model, mode/task_type, and input text.
- **Result:** Repeated queries and test runs execute in **0.0000s** for **$0 cost**, completely protecting Google's 15 RPM free-tier quota.

### B. Gold Evaluation Dataset (`eval/questions.jsonl`)
Created 20 hand-crafted test cases targeting the FastAPI technical corpus across 5 distinct buckets:
1. **Lookup (6 questions):** Direct factual questions (`fastapi dev`, HTTP methods, URL query separators).
2. **How-To (5 questions):** Implementation questions with code (Pydantic bodies, Enums, path converters).
3. **Multi-Doc (4 questions):** Questions requiring facts from multiple files (parameter precedence, OpenAPI generation).
4. **Source-Specific (2 questions):** Queries targeting specific files (`extra-models.md`, `first-steps.md`).
5. **Unanswerable (3 questions):** Topics outside the documentation (WebSockets with Redis, Quantum computing, Kubernetes Helm charts) to verify hallucination resistance.
- Split: **70% Dev** (14 questions) and **30% Test** (6 questions).

### C. Search & Retrieval Metrics Engine (`eval/metrics.py`)
- **`Hit@k`:** Measures whether the required gold document/passage was present in the top-k retrieved list.
- **`MRR` (Mean Reciprocal Rank):** Measures ranking quality ($\frac{1}{\text{rank}}$ of the first relevant passage).
- **`Abstention Accuracy`:** Assesses whether the model strictly refused unanswerables without hallucinating, and answered valid queries without false refusal.

### D. Automated Benchmark Runner (`eval/run_eval.py`)
- Iterates through the evaluation dataset, performs retrieval and generation, records latencies, and computes aggregate metrics.
- Built-in **3.5s rate-limit pacing** for uncached calls to remain safely below the 15 RPM free-tier quota.
- Exports timestamped JSON audit reports to `results/runs/run_<timestamp>.json`.

---

## 3. Structure-Aware Chunking (Heading Injection & Code Protection)

### A. The Common `Chunker` Interface (`app/chunking/base.py`)
- Created an abstract base class `Chunker` with a standardized `.split(text, metadata) -> list[dict]` contract.
- Allows hot-swapping chunking strategies via configuration (`CHUNKER_STRATEGY = "markdown"`).

### B. Pre-Chunking Document Cleaner (`app/chunking/cleaner.py`)
- Normalizes Unicode characters (NFKC).
- Strips YAML frontmatter, HTML comments, and framework-specific macros (`{* ... *}`, `/// note`).
- Collapses excessive whitespace while preserving paragraph breaks.

### C. Structure-Aware Markdown Chunker (`app/chunking/markdown.py`)
- **Heading Path Tracking:** Maintains a stack of headers to record full hierarchy (e.g., `Path Parameters > Predefined values`).
- **Context Injection:** Prepends `[Section: H1 > H2]` directly at the start of every chunk. When embedded, the vector represents both the code snippet and its overarching topic.
- **Atomic Code Blocks:** Tracks ```` ``` ```` delimiters and prevents splitting code blocks across chunk boundaries.

---

## 4. Empirical Ablation Table & Benchmark Comparison

We ran `eval/run_eval.py` across both the naive baseline and the optimized structure-aware index:

| Run / Version | Hit@4 (Recall) | MRR | Correct Refusal | Answer Rate | p50 Latency | Key Changes |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Baseline (Naive 1000c)** | 52.9% | 0.417 | 100.0% | 88.2% | 2.47s | Naive character splitting; cold API run with disk cache setup. |
| **Optimized (Markdown 800c)** | **58.8%** | **0.441** | **100.0%** | **94.1%** | **1.01s** | **+5.9% recall! Heading paths prepended & code blocks kept intact.** |

### Key Scientific Findings:
1. **Contextual Grounding:** Chunks that previously contained only bare Python code now contain their section title (`[Section: Path Parameters]`), allowing semantic vector search to match conceptual queries like *"How do I define predefined values?"*.
2. **Reduced False Refusals:** With intact code blocks, the generator found complete code examples, reducing false refusals from 2 down to 1 (Answer Rate rose from 88.2% to 94.1%).
3. **Purity of Signal:** Hit@4 jumped nearly 6% purely from data preprocessing and chunk representation, without modifying the underlying embedding model or prompt.

---

## 5. Bugs Discovered & Resolved Today

1. **Quota Throttling (429 Rate Limit):**
   - *Problem:* Google Gemini free tier limits `generateContent` to 15 RPM. Firing 20 questions in a fast loop caused a `429 RESOURCE_EXHAUSTED` error.
   - *Solution:* Added a 3.5s sleep between uncached queries in `run_eval.py` and leveraged disk caching so cached queries return in 0.0000s without hitting the API.
2. **Doc Identifier Type Mismatch:**
   - *Problem:* `search()` returned internal chunk numbers (`[22, 21, 19]`) rather than filenames (`first-steps.md`), causing `is_passage_relevant` to crash with `AttributeError: 'int' object has no attribute 'lower'` and report 0.0% Hit Rate.
   - *Solution:* Fixed `search()` in `store.py` to assign `chunk_data["doc_name"]` and wrapped strings defensively in `metrics.py`.
3. **Contraction Mismatch in Abstention:**
   - *Problem:* Exact string matching on `"i don't have enough information"` failed when the LLM generated `"I do not have enough information"`.
   - *Solution:* Updated `check_abstention` to check for multiple variations (`"don't"`, `"do not"`).
4. **Module vs. Package Import Conflict:**
   - *Problem:* Having both `app/chunking.py` (file) and `app/chunking/` (folder) caused `ModuleNotFoundError: 'app.chunking' is not a package`.
   - *Solution:* Removed the legacy file and created `app/chunking/__init__.py`.

---

## 6. Current Repository Structure

```text
docpilot/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py           # Centralized configuration & settings
│   │   ├── llm.py              # Gemini embedding/generation with diskcache & backoff
│   │   ├── loader.py           # Multi-format text extractor (.md, .txt, .pdf)
│   │   ├── store.py            # Persistent NumPy vector store & deduplication
│   │   ├── main.py             # FastAPI backend with doc management endpoints
│   │   └── chunking/           # Modular chunking package
│   │       ├── __init__.py     # Registry exports
│   │       ├── base.py         # Chunker abstract base class
│   │       ├── cleaner.py      # Unicode, frontmatter & macro cleaner
│   │       ├── markdown.py     # Structure-aware heading & code chunker
│   │       ├── recursive.py    # Paragraph/sentence fallback chunker
│   │       └── registry.py     # Chunker factory registry
│   ├── data/
│   │   ├── index/              # Persistent on-disk vector database
│   │   │   ├── chunks.json     # 85 structured chunk records with metadata
│   │   │   └── vectors.npy     # 85 x 3072 float32 NumPy matrix
│   │   └── sample_docs/        # 5 official FastAPI Markdown files
│   ├── eval/
│   │   ├── questions.jsonl     # 20 gold evaluation benchmark questions
│   │   ├── metrics.py          # Hit@k, MRR, and Abstention accuracy formulas
│   │   └── run_eval.py         # Automated evaluation runner with pacing
│   ├── results/
│   │   ├── day1.md             # Day 1 logbook
│   │   ├── day2.md             # This comprehensive Day 2 logbook
│   │   ├── ablation.md         # Official ablation tracking table
│   │   └── runs/               # Timestamped JSON evaluation audit reports
│   ├── scripts/
│   │   └── ingest_folder.py    # CLI tool to bulk-index document folders
│   ├── .cache/                 # Persistent disk cache for embeddings and prompts
│   ├── .env                    # Protected GEMINI_API_KEY
│   └── requirements.txt
└── frontend/
    └── app.py                  # Streamlit UI with doc management & citations