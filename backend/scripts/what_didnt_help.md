# DocPilot Engineering Post-Mortem: What Didn't Help (And Why)

Every technique added to DocPilot was measured against a frozen 20-question evaluation benchmark (`Hit@4`, `MRR`, `Latency`). While techniques like Structure-Aware Chunking (+5.9% recall) and Neural Re-ranking (+39% local MRR) succeeded, several popular industry techniques actively degraded performance or added unjustified cost.

Here is the empirical record of what was dropped and why.

---

## 1. Parent-Document Retrieval (Tiny 250c Child Slices)
- **The Textbook Promise:** Index small child chunks (~250 characters) for high vector similarity, and return larger parent sections (~1200 characters) for LLM context.
- **The Empirical Reality:** Hit@4 Recall plummeted from **58.8% down to 41.2%** (-17.6% drop).
- **Root Cause Analysis:** Technical programming documentation is dominated by code blocks, decorators, and function signatures. Arbitrarily slicing text at 250 characters cut multi-line Python functions and Pydantic schemas in half. Even though child chunks pointed to parent sections, the initial child embedding vector was so fragmented that Qdrant never surfaced it in the top candidates.
- **Decision:** **DROPPED.** Reverted to `MarkdownChunker` (800 chars) with atomic code preservation.

---

## 2. Step-Back Prompting for Technical Lookups
- **The Textbook Promise:** Have an LLM generate a broader, high-level conceptual question to retrieve overarching background context.
- **The Empirical Reality:** Hit@4 remained identical (52.9%), MRR did not budge (0.407), while latency increased by **+2,036ms** per query.
- **Root Cause Analysis:** Technical questions in software documentation are concrete and specific (e.g., *"What is the command to run the dev server?"* or *"How to define optional query params?"*). Asking a broader question like *"How does FastAPI handle server execution?"* pulled generic introductory fluff that diluted the ranking of the exact CLI command.
- **Decision:** **DROPPED.** Do not use Step-Back on concrete API/code documentation.

---

## 3. Naive Reciprocal Rank Fusion (Dense + BM25) Without Re-ranking
- **The Textbook Promise:** Combine dense semantic search and sparse BM25 keyword matching via RRF to capture both semantics and exact identifiers.
- **The Empirical Reality:** On natural language queries, naive RRF lowered MRR from **0.441 down to 0.353** compared to pure Gemini dense search.
- **Root Cause Analysis:** For natural language questions without exact code identifiers (e.g., *"What happens if a path parameter is declared as an int but the user sends a string?"*), BM25 pulled chunks containing the words "path", "parameter", and "string" from unrelated chapters, pushing the truly relevant semantic chunk down past Rank 4.
- **Decision:** **MODIFIED.** BM25 is only beneficial when coupled with a downstream Cross-Encoder re-ranker that evaluates semantic interaction between candidate pairs.

---

## 4. Multi-Query Rewriting on the Default Search Path
- **The Textbook Promise:** Generate 3 alternative query phrasings and fuse all results via RRF to maximize search coverage.
- **The Empirical Reality:** Increased MRR slightly (0.407 to 0.446), but spiked median latency from **686ms to 4,033ms (5.8x slower!)**.
- **Decision:** **MADE CONDITIONAL.** Multi-Query is too slow for the default fast path. It should only be triggered conditionally when the initial retrieval confidence score is below an acceptable threshold.