# Day 1: Basic In-Memory RAG Application (v0.1)

## Overview
Built a complete, framework-free Retrieval-Augmented Generation (RAG) prototype from scratch in a single day. The system ingests technical documentation, creates normalized vector embeddings via Google Gemini, performs similarity searches using linear algebra in NumPy, serves responses via FastAPI, and provides an interactive web chat interface with source citations via Streamlit.

---

## Architecture & Implementation

### 1. Environment & Technical Corpus
- Set up a clean virtual environment using Python, FastAPI, NumPy, and the new Google GenAI SDK.
- Secured API credentials using `.env` and `.gitignore`.
- Sourced real-world documentation by cloning official Markdown tutorial files from the **FastAPI** repository:
  - `first-steps.md`
  - `path-params.md`
  - `query-params.md`
  - `body.md`
  - `extra-models.md`

### 2. The Gemini Bridge (`app/llm.py`)
- Created a lightweight, framework-free bridge to Google AI Studio:
  - **Embedding Model:** `gemini-embedding-001` (outputs 3,072-dimensional dense vectors).
  - **Generation Model:** `gemini-3.5-flash-lite` (fast, lightweight inference).
- Handled task-type optimization:
  - `task_type="RETRIEVAL_DOCUMENT"` for indexing chunks.
  - `task_type="RETRIEVAL_QUERY"` for user search queries.
- Applied **L2-normalization** to all vectors so Cosine Similarity is equivalent to a fast dot product (`A @ B`).
- Added an **exponential backoff retry wrapper** to catch and recover from rate limits (`429`) and server errors (`503`).

### 3. The In-Memory RAG Engine (`app/rag.py`)
- Zero-dependency vector retrieval pipeline using native Python and NumPy:
  - **Chunking:** Sliding character window (1,000 characters, 150 overlap).
  - **In-Memory Store:** Chunks stored in a standard Python list (`CHUNKS`); vectors stored in a 2D NumPy array (`VECTORS`).
  - **Vector Search:** Dot-product matrix multiplication (`VECTORS @ query_vector`) sorted with `np.argsort`.
  - **Strict Grounding Prompt:** Instructs the LLM to answer using *only* the provided passages, cite bracketed numbers (`[1]`, `[2]`), and strictly refuse with *"I don't have enough information in the provided documents"* if missing.

### 4. The FastAPI Service (`app/main.py`)
- Exposed the RAG pipeline over HTTP with interactive Swagger documentation (`/docs`):
  - `GET /health`: Returns health status and total count of indexed chunks.
  - `POST /ingest`: Multi-file upload endpoint to read, chunk, embed, and index new files dynamically.
  - `POST /ask`: Accepts a question and returns a grounded answer with ranked source passages.
- Includes a startup event that automatically preloads all `.md` files present in `data/sample_docs/`.

### 5. The Streamlit Frontend (`frontend/app.py`)
- Developed a two-column web interface:
  - **Sidebar:** Live backend health status, top-k slider, and multi-file uploader.
  - **Chat Area:** Preserves chat history and displays model answers.
  - **Expandable Citation Drawers:** Renders an accordion below each response showing exact chunk text, source filename, chunk ID, and cosine relevance score.

---

## Empirical Observations & Test Results

### What Succeeded
1. **Multi-Source Synthesis & Citation:**
   - **Query:** *"Can I pass both body and path parameters in the same function?"*
   - **Output:** Accurately synthesized facts across `body.md` and `query-params.md`, explaining parameter detection with inline citations (`[1]`, `[2]`).
2. **Zero-Hallucination Abstention:**
   - **Query:** *"What is quantum computing?"*
   - **Output:** Exactly responded with *"I don't have enough information in the provided documents."* The LLM did not invent facts or leak pre-training knowledge.

### Failures Observed (The Day 1 Baseline)
1. **False Refusal on Documented Information:**
   - **Query:** *"How do I declare optional query parameters?"*
   - **Output:** Refused to answer (*"I don't have enough information"*), even though the concept exists in `query-params.md`.
   - **Root Cause:** Naive fixed 1,000-character chunking cut the code example mid-block, or dense vector search ranked the relevant passage outside the top-4 (`top_k=4`).
   - **Action Plan:** Needs structure-aware chunking (Day 4) and Hybrid Search (BM25 + Re-ranking) (Days 7–8).
2. **Rate Limit / Quota Spike:**
   - Google AI Studio dashboard showed `Gemini Embedding 1` reaching **148 / 100 RPM** during batch indexing.
   - **Root Cause:** In-memory storage forces a full re-embedding of all documents on every application startup.
   - **Action Plan:** Needs persistent on-disk index storage (Day 2) and disk caching (Day 3).

---

## Repository Layout

```text
docpilot/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── llm.py          # Gemini embedding, generation & retry backoff
│   │   ├── rag.py          # In-memory chunking, numpy dot-product search, grounding
│   │   └── main.py         # FastAPI endpoints (/health, /ingest, /ask)
│   ├── data/
│   │   └── sample_docs/    # 5 official FastAPI Markdown tutorial files
│   ├── results/
│   │   └── notes.md        # Empirical failure log and notes
│   ├── .env                # GEMINI_API_KEY
│   └── requirements.txt
└── frontend/
    └── app.py              # Streamlit chat interface with citation expanders
```

**Git Milestone Tag:** `v0.1`