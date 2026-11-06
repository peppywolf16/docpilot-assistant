# Day 5 Sprint: Trust, Streaming, Agentic LangGraph & Enterprise Security (Roadmap Days 11–16)

## Executive Summary
Calendar Day 5 was the most comprehensive development sprint of the project, elevating DocPilot from an advanced search pipeline into an autonomous, trustworthy, real-time streaming, and enterprise-hardened AI system:
1. **Parent-Document & Contextual Retrieval (Day 11):** Decoupled search pieces (250c child chunks) from context pieces (1200c parent sections) with deduplication, and tested Anthropic-style contextual prefix anchors.
2. **Verified Citations & Strict Abstention (Day 12):** Enforced Pydantic structured output (`GroundedResponse`), built deterministic Python quote verification (`Citation Validity Rate: 1.0`), and implemented Cross-Encoder score thresholds for instant refusal without LLM calls.
3. **Real-Time Token Streaming & UI v1.0 (Day 13):** Upgraded the backend with Server-Sent Events (`POST /ask/stream`) and connected Streamlit using `st.write_stream()` to deliver a live, demo-ready conversational experience.
4. **LangGraph Agent with Self-Correction (Day 14 - CRAG):** Wrapped the pipeline in a stateful cyclic graph that evaluates retrieved document relevance using Cross-Encoder logits and LLM scoring, conditionally rewriting queries and retrying when relevance is poor.
5. **Conversation Memory & Self-RAG (Day 15):** Implemented multi-turn conversation memory with `MemorySaver` checkpointer and `thread_id`, pronoun resolution (`"Can it also accept an Enum?"` $\rightarrow$ `"Can FastAPI path parameters accept an Enum?"`), an independent `check_support_node` hallucination checker, and exported a visual Mermaid state machine diagram.
6. **Enterprise Security & Multi-Tenancy (Day 16):** Built cryptographically verified JWT authentication (`/login`), mandatory Qdrant multi-tenancy ACL filters (`tenant` and `allowed_roles`), automated PII/credential scrubbing at ingestion (`mask_pii`), prompt injection detection, untrusted context boundary framing, and achieved a **100% pass rate (5/5)** on our automated pytest security test suite.

---

## 1. Smarter Indexing & Verified Citations (Roadmap Days 11 & 12)

### A. Parent-Document Retrieval (`app/chunking/parent_child.py`)
- Indexed small child chunks (~250 chars) in Qdrant for razor-sharp vector similarity while storing larger parent sections (~1,200 chars) in payloads.
- Added `seen_parents` deduplication in `store_qdrant.py` so multiple child hits from the same section emit only one clean parent block to the LLM.
- **Empirical Finding:** On technical code documentation, Structure-Aware Markdown Chunking (800c) gave higher recall (**58.8%** vs. 41.2%) and required $\frac{1}{3}$ of the embedding API calls compared to 250c child slices.

### B. Verified Citations & Fast Refusal (`app/generation/verifier.py`)
- **Pydantic Schema:** Structured output enforcing `answerable: bool`, `answer: str`, and `citations: list[Citation(passage_number, exact_quote)]`.
- **Deterministic Python Verification:** Scans every cited quote against the retrieved passage text in code. Hallucinated quotes are stripped out, tracking a **Citation Validity Rate of 1.0 (100%)**.
- **Score Threshold Fast Refusal:** If top-1 passage score is below `-4.0`, refuses immediately with zero LLM generation calls.
- **Prompt Versioning:** Managed via `PROMPT_V2_STRUCTURED` in `app/generation/prompts.py`.

---

## 2. Real-Time Token Streaming & UI v1.0 (Roadmap Day 13)

### A. Server-Sent Events (`app/main.py`)
- Built `POST /ask/stream` using FastAPI's `StreamingResponse(media_type="text/event-stream")`.
- Implemented `generate_stream()` in `app/llm.py` streaming tokens directly from Google Gemini.

### B. Streamlit UI v1.0 (`frontend/app.py`)
- Consumes incoming token streams in real-time via `st.write_stream()`.
- Interactive sidebar document manager with live chunk counters and delete buttons.

---

## 3. Agentic LangGraph State Machine (Roadmap Days 14 & 15)

### A. Corrective RAG (CRAG) Workflow (`app/agent/graph.py`)
[User Input] ──> [condense_query] ──> [retrieve] ──> [grade_documents]
│
┌─── score < 0.4 ───────┴─────── score >= 0.4 ───┐
▼ ▼
[rewrite_query] ──(retry)──> [retrieve] [generate]
│
▼
[check_support]
│
▼
[END]

### B. Conversation Memory & Pronoun Resolution
- **LangGraph Checkpointer:** Uses `MemorySaver` configured with `thread_id`.
- **Query Condensation:** Resolves follow-up pronouns based on chat history:
  - *Turn 1:* `"How do I declare path parameters in FastAPI?"`
  - *Turn 2:* `"Can it also accept an Enum?"`
  - *Condensed Query:* `"Can FastAPI path parameters accept an Enum?"`

### C. Self-RAG Answer Groundedness Check
- `check_support_node` evaluates whether factual claims in the draft answer are strictly supported by retrieved documentation passages before delivering the response.

---

## 4. Enterprise Security & Multi-Tenancy (Roadmap Day 16)

### A. Cryptographic Auth & RBAC (`app/security/auth.py`)
- Implemented `POST /login` issuing signed JWT access tokens (`HS256`).
- User `tenant` and `role` are cryptographically verified from token payload, never trusted from client request bodies.

### B. Multi-Tenancy Retrieval Access Control (`app/store_qdrant.py`)
- Injected mandatory filters into Qdrant:
  - `FieldCondition(key="tenant", match=MatchValue(value=user.tenant))`
  - `FieldCondition(key="allowed_roles", match=MatchAny(any=[user.role]))`
- Restricted documents are physically invisible to unauthorized roles at the index level.

### C. Ingestion Sanitization & PII Masking (`app/security/pii.py`)
- Automatic redaction of leaked API keys (`AIza...`, `sk-...`), Bearer tokens, emails, and IPv4 addresses before chunking and embedding.

### D. Prompt Injection Shield (`app/security/guardrails.py`)
- Regex scanner intercepting system prompt overrides and jailbreaks.
- Untrusted context boundary fencing (`<untrusted_documentation_context>`) instructing the LLM never to execute instructions embedded inside retrieved text.

### E. Pytest Security Suite (`tests/test_security.py`)
tests/test_security.py::test_tampered_jwt_rejected PASSED [ 20%]
tests/test_security.py::test_pii_masking PASSED [ 40%]
tests/test_security.py::test_prompt_injection_detection PASSED [ 60%]
tests/test_security.py::test_untrusted_context_framing PASSED [ 80%]
tests/test_security.py::test_access_control_filter_simulation PASSED [100%]
============================== 5 passed in 1.50s ===============================
---

## 5. Cumulative Ablation Table (Full Progress)

| Sprint / Day | Version / Milestone | Hit@4 (Recall) | MRR | Correct Refusal | Answer Rate | Latency | Core Innovation |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Day 1** | Naive Baseline (1000c + NumPy) | 52.9% | 0.417 | 100.0% | 88.2% | 2.47s | In-memory proof of concept. |
| **Day 2** | Markdown Chunker (800c) + Disk Persistence | 58.8% | 0.441 | 100.0% | 94.1% | 1.01s | Structure-aware heading injection. |
| **Day 3** | BGE Embeddings & Qdrant Vector DB | 58.8% | 0.441 | 100.0% | 94.1% | 12.4ms | Production vector database + payload filtering. |
| **Day 4** | Two-Stage Re-ranking & Query Rewriting (v0.5) | 52.9% | 0.446 | 100.0% | 94.1% | 686ms | Neural Cross-Encoder & Multi-Query shootout. |
| **Day 5** | **Agentic LangGraph, Streaming & Security** | **58.8%** | **0.441** | **100.0%** | **100.0%** | **Real-Time** | **LangGraph CRAG state machine, memory, verified quotes (1.0 validity), and 100% security test pass.** |

---

## 6. Current Repository Layout

```text
docpilot/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration & feature flags
│   │   ├── llm.py              # Gemini embedding/generation, streaming & diskcache
│   │   ├── loader.py           # Multi-format text extractor (.md, .txt, .pdf)
│   │   ├── store.py            # Store adapter routing to Qdrant & Smart Search
│   │   ├── store_qdrant.py     # Qdrant client, metadata payloads & ACL filters
│   │   ├── retriever.py        # Smart Two-Stage Search: Hybrid + Decomposition + Re-ranking
│   │   ├── reranker.py         # Cross-Encoder neural re-ranker, deduplication & Lost-in-the-Middle
│   │   ├── main.py             # FastAPI backend with JWT auth, streaming, doc management & filters
│   │   ├── agent/              # LangGraph state machine package
│   │   │   ├── __init__.py
│   │   │   └── graph.py        # CRAG loop, memory condensation, checkpointer & Self-RAG
│   │   ├── chunking/           # Modular chunking package (Base, Cleaner, Markdown, Parent-Child)
│   │   ├── embeddings/         # Modular embeddings package (Base, Gemini, BGE, Registry)
│   │   ├── generation/         # Trustworthy generation package
│   │   │   ├── __init__.py
│   │   │   ├── schemas.py      # Pydantic schemas (Citation, GroundedResponse)
│   │   │   ├── prompts.py      # Prompt versioning (PROMPT_V1, PROMPT_V2)
│   │   │   └── verifier.py     # Quote verification & threshold fast refusal
│   │   ├── query/              # Pre-retrieval intelligence package
│   │   │   ├── __init__.py
│   │   │   ├── base.py         # QueryTransform abstract base class
│   │   │   ├── multi_query.py  # Multi-Query rephrasings
│   │   │   ├── hyde.py         # Hypothetical Document Embeddings
│   │   │   ├── step_back.py    # Step-Back conceptual prompting
│   │   │   ├── decomposition.py# Multi-part query decomposition
│   │   │   ├── filter_extractor.py # Pydantic automatic filter extractor & fallback
│   │   │   └── router.py       # Intent router (direct_chat, search, decompose)
│   │   └── security/           # Enterprise security package
│   │       ├── __init__.py
│   │       ├── auth.py         # JWT token creation & cryptographic verification
│   │       ├── pii.py          # PII detection & credential masking
│   │       └── guardrails.py   # Prompt injection defense & untrusted context framing
│   ├── data/
│   │   ├── index/              # Model-specific Qdrant persistent storage folders
│   │   └── sample_docs/        # Official FastAPI Markdown tutorial files
│   ├── eval/
│   │   ├── questions.jsonl     # 20 gold evaluation benchmark questions
│   │   ├── metrics.py          # Hit@k, MRR, and Abstention accuracy formulas
│   │   ├── run_eval.py         # Automated evaluation runner with per-bucket telemetry
│   │   └── failure_analysis.py # Diagnostic tool isolating retrieval vs. ranking failures
│   ├── results/
│   │   ├── day1.md             # Day 1 logbook
│   │   ├── day2.md             # Day 2 logbook (Foundations & Chunking)
│   │   ├── day3.md             # Day 3 logbook (Embeddings & Qdrant)
│   │   ├── day4.md             # Day 4 logbook (Hybrid, Re-ranking & Query Rewriting)
│   │   ├── day5.md             # This Day 5 logbook (Streaming, LangGraph, Memory & Security)
│   │   ├── ablation.md         # Official cumulative ablation table
│   │   └── runs/               # Timestamped JSON evaluation audit reports
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_security.py    # Automated security test suite (5 passed in 1.50s)
│   ├── scripts/
│   │   └── ingest_folder.py    # CLI tool to bulk-index document folders
│   ├── .cache/                 # Persistent disk cache for embeddings and prompts
│   ├── .env                    # Protected API keys & model configs
│   └── requirements.txt
└── frontend/
    └── app.py                  # Streamlit UI with SSE streaming, doc management, filters & citations
