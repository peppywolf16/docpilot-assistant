# DocPilot

**DocPilot** is a version-aware Retrieval-Augmented Generation (RAG)
assistant for exploring the documentation of an open-source project. It
lets users ask questions about project documentation and receive answers
grounded in retrieved passages, with source references and an explicit
fallback when the available documents do not contain enough information.

The project is designed as both a practical documentation assistant and
an end-to-end learning project for building, evaluating, and improving a
RAG system. Its central idea is to improve retrieval and answer quality
in measurable stages rather than adding complexity without evidence.

> **Project status:** The supplied project plan describes a phased
> build, beginning with a basic upload-and-query RAG application and
> progressing toward hybrid retrieval, reranking, query transformation,
> agentic correction, security, observability, and deployment. Treat
> advanced capabilities as roadmap items unless they have been
> implemented in the current codebase.

------------------------------------------------------------------------

## Table of contents

-   [Why DocPilot?](#why-docpilot)
-   [Problem it addresses](#problem-it-addresses)
-   [Core workflow](#core-workflow)
-   [Key capabilities](#key-capabilities)
-   [Architecture](#architecture)
-   [Technology stack](#technology-stack)
-   [Document sources and ingestion](#document-sources-and-ingestion)
-   [Retrieval and answer generation](#retrieval-and-answer-generation)
-   [Evaluation approach](#evaluation-approach)
-   [Security and reliability](#security-and-reliability)
-   [Repository structure](#repository-structure)
-   [Getting started](#getting-started)
-   [Development roadmap](#development-roadmap)
-   [Design principles](#design-principles)
-   [Future extensions](#future-extensions)

## Why DocPilot?

Technical documentation changes frequently. A function may be renamed,
an API may be deprecated, or behavior may differ between releases. A
general-purpose language model can produce a plausible answer based on
outdated knowledge, even when the answer is incorrect for the version a
developer is using.

DocPilot addresses this by grounding responses in a selected
documentation corpus. Documents carry metadata such as source, version,
document type, and URL so that retrieval can take the requested version
and source into account.

The intended corpus can include:

1.  Official documentation (Markdown or HTML).
2.  Release notes and changelogs.
3.  GitHub issues and discussions.
4.  Optionally, source code for code-aware retrieval.

The project plan uses fast-changing developer documentation---such as
DuckDB or Polars---as example corpora. The corpus is configurable and
should be pinned to a version or commit when running evaluations.

## Problem it addresses

A useful documentation assistant must do more than retrieve text that
looks similar to a question. It should:

-   Find the correct information for the requested documentation
    version.
-   Handle exact identifiers, function names, flags, and error messages.
-   Preserve code blocks, tables, headings, and document structure
    during ingestion.
-   Combine semantic and lexical retrieval where appropriate.
-   Provide citations that point to the passages used.
-   Admit when the indexed sources do not support an answer.
-   Avoid exposing documents that a user is not authorized to access.

## Core workflow

At a high level, the system follows this pipeline:

``` text
Documentation sources
        |
        v
Ingestion and parsing
        |
        v
Cleaning and metadata extraction
        |
        v
Chunking
        |
        v
Embeddings and indexing
        |
        v
Question processing
        |
        v
Retrieval and reranking
        |
        v
Context construction
        |
        v
LLM answer generation
        |
        v
Grounded answer with citations
```

In the more advanced design, a query layer can rewrite or decompose
questions, route them to suitable sources, and apply metadata filters.
An agent loop can grade retrieved passages, retry retrieval, or use a
fallback source when the evidence is insufficient.

## Key capabilities

The following describes the capabilities targeted by the project plan.
Availability in a particular checkout depends on which phases have been
implemented.

### 1. Document ingestion

-   Load supported documentation files, including text, Markdown, and
    PDF in the planned application.
-   Clean extracted content and retain useful structure.
-   Extract metadata such as document name, source type, version, URL,
    heading, and position.
-   Re-ingest changed documents without creating duplicate entries.

### 2. Structure-aware chunking

The plan compares multiple chunking strategies:

-   Fixed-size chunking.
-   Recursive chunking.
-   Markdown-heading-aware chunking.
-   Semantic chunking.
-   Optional AST-aware chunking for source code.

The goal is to preserve meaningful units such as code blocks, tables,
and section headings instead of splitting them arbitrarily.

### 3. Embeddings and vector indexing

-   Convert document chunks and user questions into vectors.
-   Compare local embedding models with an API-based embedding option.
-   Evaluate embedding quality, retrieval quality, latency, and memory
    usage.
-   Explore vector compression techniques such as Matryoshka truncation
    and int8/binary quantization.

The initial plan uses local sentence-transformer models as an option and
Gemini embeddings as a comparison. Chroma is the initial vector-store
choice, with Qdrant or PostgreSQL/pgvector considered as alternatives.

### 4. Retrieval

The roadmap progressively introduces:

-   Dense semantic retrieval.
-   BM25 lexical retrieval.
-   Hybrid retrieval using Reciprocal Rank Fusion (RRF).
-   Metadata filtering by version and source type.
-   Cross-encoder reranking.
-   Parent-document retrieval.
-   Contextual retrieval.

These methods are intended to address different query types. For
example, semantic retrieval can help with conceptual questions, while
lexical retrieval can be valuable for exact function names, flags, and
error codes.

### 5. Query processing

Advanced query strategies in the plan include:

-   Multi-query retrieval: generate alternative phrasings and combine
    their results.
-   HyDE: generate a hypothetical answer passage and retrieve against
    it.
-   Step-back prompting: retrieve broader context for a specific
    question.
-   Query decomposition: split multi-part questions into smaller
    sub-questions.
-   Metadata extraction and filtering.
-   Query routing to documentation, release notes, or issues.

Each strategy should be evaluated independently to determine when it
helps and when it adds unnecessary latency or cost.

### 6. Grounded answer generation

The answer-generation layer is designed to:

-   Use retrieved passages as the evidence for its response.
-   Include structured citations with document and passage information.
-   Return an explicit uncertainty or abstention response when evidence
    is insufficient.
-   Avoid presenting unsupported statements as facts.
-   Support streaming responses in the user interface.

### 7. Evaluation and observability

DocPilot is intended to measure changes rather than rely on subjective
impressions. Its evaluation harness tracks retrieval and generation
quality, latency, and cost. Tracing tools such as Langfuse or Arize
Phoenix are included in the plan for inspecting pipeline execution.

### 8. Agentic RAG

A later phase introduces a LangGraph-based workflow that can:

1.  Retrieve relevant documents.
2.  Grade the relevance of retrieved passages.
3.  Rewrite the query or retry retrieval when evidence is weak.
4.  Use a fallback source where configured.
5.  Check whether the generated answer is supported.
6.  Return an answer or abstain.

The agentic pipeline should be compared against the non-agentic pipeline
using the same evaluation set.

------------------------------------------------------------------------

## Architecture

The planned end-state architecture separates the RAG pipeline into
components with common interfaces. This makes it possible to switch
chunking, retrieval, embedding, and generation strategies without
rewriting the entire application.

``` mermaid
flowchart TD
    A[Documentation sources] --> B[Ingestion]
    B --> C[Parsing, cleaning and metadata]
    C --> D[Chunking]
    D --> E[Embedding]
    E --> F[Vector index]
    E --> G[Lexical index]

    H[User question] --> I[Query layer]
    I --> J[Rewrite, route or decompose]
    J --> K[Dense retrieval]
    J --> L[Lexical retrieval]
    F --> K
    G --> L
    K --> M[Merge and rerank]
    L --> M
    M --> N[Context construction]
    N --> O[LLM generation]
    O --> P[Grounded answer and citations]

    M --> Q[Agent evaluation]
    Q -->|Insufficient evidence| I
    Q -->|Relevant evidence| N

    R[Evaluation and tracing] -.-> B
    R -.-> I
    R -.-> M
    R -.-> O
```

### Component responsibilities

  -----------------------------------------------------------------------
  Component                           Responsibility
  ----------------------------------- -----------------------------------
  Ingestion                           Load documents from configured
                                      sources

  Parsing and cleaning                Extract text while preserving
                                      useful structure

  Metadata extraction                 Attach version, source, URL,
                                      heading, and other attributes

  Chunking                            Split documents into
                                      retrieval-friendly passages

  Embedding                           Convert chunks and questions into
                                      vectors

  Vector index                        Store vectors and support
                                      similarity search

  Lexical index                       Support exact-term and keyword
                                      retrieval

  Query layer                         Rewrite, decompose, filter, or
                                      route questions

  Retriever                           Fetch candidate passages

  Reranker                            Reorder candidates by relevance

  Generator                           Produce an answer grounded in the
                                      selected context

  Agent workflow                      Grade evidence and control retries
                                      or fallbacks

  Evaluation and tracing              Measure quality and inspect
                                      execution
  -----------------------------------------------------------------------

## Technology stack

The project plan identifies the following technologies. These are
planned or candidate choices; the actual dependencies should be
confirmed from the repository's dependency files.

  -----------------------------------------------------------------------
  Layer                               Technology / options
  ----------------------------------- -----------------------------------
  Language                            Python 3.11+

  API backend                         FastAPI, Uvicorn

  Frontend                            Streamlit initially; Next.js as a
                                      possible later option

  LLM                                 Gemini API through a swappable
                                      model interface

  Embeddings                          Sentence Transformers (for example,
                                      BGE/E5) and Gemini embeddings

  Vector store                        Chroma initially; Qdrant or
                                      pgvector as alternatives

  Lexical search                      BM25; OpenSearch or PostgreSQL FTS
                                      as possible extensions

  Reranking                           Local BGE reranker or Cohere Rerank

  RAG framework                       Initial implementation without a
                                      framework; LangChain/LlamaIndex and
                                      LangGraph in later phases

  Evaluation                          Ragas and custom
                                      retrieval/generation metrics

  Tracing                             Langfuse or Arize Phoenix

  Caching                             Disk cache initially; Redis in the
                                      production-oriented plan

  Packaging                           Docker and Docker Compose

  Validation                          Pydantic
  -----------------------------------------------------------------------

## Document sources and ingestion

The project plan favors downloading a fixed snapshot of documentation
instead of fetching live web pages during every evaluation.

  -----------------------------------------------------------------------
  Source                              Planned ingestion method
  ----------------------------------- -----------------------------------
  Official documentation              Clone the documentation repository
                                      and check out a pinned version or
                                      commit

  Release notes and changelogs        Read from the repository or use the
                                      GitHub Releases API

  GitHub issues and discussions       Fetch through the GitHub REST API

  Source code (optional)              Clone the main repository for
                                      AST-aware chunking

  Research PDFs (side experiment)     Download or upload papers into a
                                      separate corpus
  -----------------------------------------------------------------------

Keeping a pinned corpus makes experiments reproducible: a change in the
evaluation score can be attributed to a pipeline change rather than a
silently changed document set.

## Retrieval and answer generation

A typical question-answering request is intended to work as follows:

1.  The user submits a question and, where supported, selects a
    documentation version or source.
2.  The query layer prepares the question and applies relevant metadata
    filters.
3.  The retriever obtains candidate passages from the configured
    indexes.
4.  Hybrid retrieval merges semantic and lexical results when enabled.
5.  A reranker selects the most relevant passages.
6.  The generator receives the question and selected context.
7.  The response includes an answer and references to the supporting
    passages.
8.  If the retrieved evidence is insufficient, the system returns an
    abstention response rather than inventing an answer.

Example of the intended response shape:

``` json
{
  "answer": "The answer supported by the retrieved documentation.",
  "citations": [
    {
      "document": "documentation-file.md",
      "chunk_id": "chunk-001",
      "source": "official_docs",
      "version": "selected-version",
      "quote": "Relevant supporting passage."
    }
  ],
  "abstained": false
}
```

This is an illustrative response format, not a guarantee of the current
API schema.

## Evaluation approach

Evaluation is a central part of DocPilot. The plan calls for a manually
reviewed question set, with questions grouped into categories such as:

-   Lookup questions.
-   How-to questions.
-   Multi-document questions.
-   Version-specific questions.
-   Unanswerable questions.

Each answerable question should have a reference answer and a gold
source passage. The set should be split into development and test
portions, and the test split should remain fixed while experiments are
compared.

### Metrics

  Area          Metrics
  ------------- --------------------------------------------------
  Retrieval     Hit Rate@K, Mean Reciprocal Rank (MRR), Recall@K
  Generation    Faithfulness, answer relevance
  Abstention    Correct abstention rate
  Citations     Citation validity and support
  Performance   p50 latency, latency by stage
  Cost          Cost per query and token usage
  System        Cache hit rate and error rate

### Ablation and comparison

The project plan proposes recording each experiment in an ablation
table, including the configuration, retrieval metrics, answer quality,
latency, and cost. Example comparisons include:

-   Naive RAG baseline.
-   Hybrid retrieval with reranking.
-   Contextual retrieval.
-   Query rewriting and decomposition.
-   Agentic RAG with self-correction.

Change one major factor per experiment, preserve the test set, and cache
model calls where practical. This makes results easier to interpret and
helps identify techniques that do not justify their extra complexity.

## Security and reliability

The roadmap includes a security layer that can be tested using synthetic
internal documents and multiple users.

Planned safeguards include:

-   **Authentication:** token-based authentication for API access.
-   **Role-based access:** use verified identity and role information.
-   **Document-level access control:** filter retrieval results using
    tenant, role, and confidentiality metadata.
-   **PII masking:** detect and mask sensitive values such as emails,
    tokens, and API keys during ingestion.
-   **Prompt-injection resistance:** treat retrieved document content as
    untrusted input and test against malicious instructions embedded in
    documents.
-   **Rate limiting:** enforce per-user or per-tenant request and token
    budgets.
-   **Cache isolation:** include relevant access-control and version
    information in cache keys to prevent cross-user data leakage.
-   **Security tests:** test unauthorized access, cross-tenant
    retrieval, tampered tokens, injection attempts, and filter bypass
    attempts.

Access checks must be applied during retrieval, not merely after the LLM
has generated an answer.

## Repository structure

The following is the logical structure described in the project plan.
Adapt it to the actual folders in the repository.

``` text
docpilot/
├── README.md
├── PLAN.md
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── ingest/
│   │   ├── chunking/
│   │   ├── embed/
│   │   ├── index/
│   │   ├── retrieve/
│   │   ├── query/
│   │   ├── generate/
│   │   ├── agent/
│   │   ├── security/
│   │   └── eval/
│   ├── data/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── eval/
│   ├── experiments/
│   ├── results/
│   └── scripts/
├── frontend/
│   └── app.py
├── docker-compose.yml
└── .env.example
```

The project plan also describes fetch scripts for downloading pinned
documentation, release notes, and GitHub issues into a local raw-data
directory. Raw downloaded data and secrets should not be committed to
version control.

## Getting started

The exact installation and run commands depend on the current repository
implementation. The project plan's initial application uses a FastAPI
backend and a Streamlit frontend.

### Prerequisites

-   Python 3.11 or later.
-   Git.
-   A Gemini API key for the planned Gemini-backed generation/embedding
    setup.
-   The dependencies listed in the repository's `requirements.txt` or
    project configuration.

### 1. Clone the repository

``` bash
git clone <your-repository-url>
cd docpilot
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

``` bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Use the dependency file present in the repository. For example:

``` bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the location expected by the application and add
the required credentials.

``` env
GEMINI_API_KEY=your_gemini_api_key
```

The variable name above is illustrative; use the exact name read by the
current application. Keep `.env` out of Git and commit an `.env.example`
containing placeholders instead.

### 5. Start the backend

For a FastAPI application exposed through `app.main:app`, the planned
development command is:

``` bash
uvicorn app.main:app --reload
```

The actual module path and port may differ in the current repository.

### 6. Start the frontend

For the planned Streamlit interface:

``` bash
streamlit run frontend/app.py
```

Open the local URL printed by Streamlit. Confirm the backend URL and any
required frontend environment variables match your local setup.

> **Note:** These commands describe the planned FastAPI/Streamlit setup.
> Check the current source tree and dependency files before using them;
> the supplied planning documents do not establish the exact commands or
> environment-variable names in every project version.

## Development roadmap

The project is intended to evolve in measurable phases.

  -----------------------------------------------------------------------
  Phase                   Focus                   Intended outcome
  ----------------------- ----------------------- -----------------------
  0                       Corpus and evaluation   Pinned documents and
                          set                     reviewed questions

  1                       Naive RAG               End-to-end baseline

  2                       Evaluation and tracing  Repeatable metrics and
                                                  baseline results

  3                       Ingestion and chunking  Clean, structure-aware
                                                  chunks

  4                       Embeddings and index    Evidence-based
                          tuning                  embedding/index choice

  5                       Hybrid search and       Better retrieval for
                          reranking               semantic and
                                                  exact-match queries

  6                       Query layer             Rewriting,
                                                  decomposition,
                                                  filtering, and routing

  7                       Smarter indexing        Contextual and
                                                  parent-document
                                                  retrieval

  8                       Generation, citations,  Grounded answers,
                          and UI                  abstention, and
                                                  streaming interface

  9                       Agentic RAG             Retrieval grading,
                                                  retries, and fallback
                                                  behavior

  10                      Production hardening    Access control, PII
                          and security            handling, injection
                                                  tests, and quotas

  11                      Fine-tuning             Compare a fine-tuned
                                                  embedding model against
                                                  the baseline

  12                      Framework comparison    Compare a second RAG
                                                  framework and prompt
                                                  optimization
  -----------------------------------------------------------------------

A shorter 20-day build plan in the supplied material prioritizes a
working basic application first, followed by persistence, evaluation,
chunking, embeddings, Qdrant, hybrid retrieval, reranking, query
processing, citations, streaming, LangGraph self-correction, security,
caching, and final evaluation.

## Design principles

1.  **Measure before optimizing.** Establish a baseline before adding
    advanced retrieval techniques.
2.  **Keep the corpus reproducible.** Pin documentation versions and
    commits for fair comparisons.
3.  **Change one thing at a time.** Log configuration, quality, latency,
    and cost for every experiment.
4.  **Use common interfaces.** Make chunkers, embedders, retrievers, and
    generators replaceable.
5.  **Ground every answer.** Cite retrieved evidence and abstain when
    the corpus cannot answer.
6.  **Treat retrieved content as untrusted.** Documents can contain
    irrelevant or malicious instructions.
7.  **Keep the system practical.** Prefer techniques that demonstrate
    measurable value over unnecessary complexity.

## Future extensions

Potential extensions described in the project material include:

-   Comparing vector RAG with GraphRAG, RAPTOR, and ColPali on different
    question types.
-   Parsing research papers with tools such as PyMuPDF, MinerU, or
    LlamaParse.
-   Extracting entities and relationships into a graph using
    GraphRAG-style processing.
-   Comparing a second framework such as LlamaIndex or Haystack.
-   Testing DSPy for prompt optimization.
-   Deploying the application using Docker and a hosted demo platform.

These are optional experiments, separate from the core documentation
assistant.

------------------------------------------------------------------------

## A note on project scope

DocPilot is not intended to be just a chat interface over uploaded
files. Its broader objective is to demonstrate the engineering decisions
behind a reliable RAG application: reproducible ingestion, version-aware
retrieval, measurable evaluation, grounded generation, observability,
and security. The phased approach makes it possible to show which
improvements actually help and what trade-offs they introduce.
