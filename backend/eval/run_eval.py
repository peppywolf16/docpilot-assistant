import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.config import EMBEDDER_TYPE, EMBEDDING_DIM
from app.store import search, CHUNKS
from app.llm import generate
from eval.metrics import calculate_hit_at_k, calculate_mrr, check_abstention

QUESTIONS_FILE = backend_dir / "eval" / "questions.jsonl"
RUNS_DIR = backend_dir / "results" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def run_evaluation(top_k: int = 4, retrieval_only: bool = False):
    if not QUESTIONS_FILE.exists():
        print(f"Error: {QUESTIONS_FILE} not found!")
        return

    questions = [json.loads(line) for line in QUESTIONS_FILE.read_text(encoding="utf-8").strip().split("\n")]
    model_info = f"{EMBEDDER_TYPE}"
    mode_text = "[RETRIEVAL-ONLY MODE]" if retrieval_only else "[FULL RAG MODE]"

    print(f"\n{mode_text} Evaluating {len(questions)} questions with {model_info} across {len(CHUNKS)} chunks...\n")

    results = []
    latencies = []

    for i, q in enumerate(questions, start=1):
        q_id = q["id"]
        q_text = q["question"]
        gold_doc = q.get("gold_doc")
        gold_snippet = q.get("gold_passage")
        is_unanswerable = (q["type"] == "unanswerable")

        # 0. Print progress so you can watch it evaluate
        print(f"[{i:02d}/{len(questions)}] {q_id} ({q['type']}): {q_text[:45]}...")

        t0 = time.time()

        # 1. Retrieve
        passages = search(q_text, top_k=top_k)

        # 2. Generate (optional)
        answer = ""
        abstain_ok, abstain_reason = True, "Skipped (retrieval-only)"

        if not retrieval_only:
            context = "\n\n".join([f"[{p['number']}] (Source: {p['doc']})\n{p['text']}" for p in passages])
            prompt = f"""You are a helpful technical documentation assistant. Answer the user's question using ONLY the provided passages below.

Rules:
1. Cite the passages you use with numbers in brackets, like [1], [2].
2. If the answer cannot be found in the provided passages, reply EXACTLY with:
"I don't have enough information in the provided documents."
Do not make anything up or use outside knowledge.

Passages:
{context}

Question: {q_text}
Answer:"""
            answer = generate(prompt)
            abstain_ok, abstain_reason = check_abstention(answer, is_unanswerable)

        elapsed = time.time() - t0
        latencies.append(elapsed)

        # 3. Calculate metrics
        hit = calculate_hit_at_k(passages, gold_doc, gold_snippet, k=top_k)
        mrr = calculate_mrr(passages, gold_doc, gold_snippet, k=top_k)

        results.append({
            "id": q_id,
            "type": q["type"],
            "split": q["split"],
            "question": q_text,
            "gold_doc": gold_doc,
            "hit@k": hit,
            "mrr": mrr,
            "abstention_ok": abstain_ok,
            "abstention_reason": abstain_reason,
            "retrieved_docs": [p["doc"] for p in passages],
            "latency_sec": round(elapsed, 4),
            "answer": answer.strip(),
        })

        # --- Sleep 3.5s to stay under 15 requests per minute! ---
        if i < len(questions):
            time.sleep(3.5)

    answerable = [r for r in results if r["type"] != "unanswerable"]
    unanswerable = [r for r in results if r["type"] == "unanswerable"]

    mean_hit = np.mean([r["hit@k"] for r in answerable]) if answerable else 0.0
    mean_mrr = np.mean([r["mrr"] for r in answerable]) if answerable else 0.0
    correct_refusals = np.mean([1.0 if r["abstention_ok"] else 0.0 for r in unanswerable]) if unanswerable else 0.0
    p50_latency = float(np.median(latencies))

    # Print Summary Table
    print("=" * 65)
    print(f"       DOCPILOT EVALUATION SUMMARY ({model_info})")
    print("=" * 65)
    print(f"Total Questions Evaluated: {len(results)} ({len(answerable)} answerable)")
    print("-" * 65)
    print(f"Hit@{top_k} (Recall):          {mean_hit * 100:.1f}%")
    print(f"MRR (Mean Reciprocal Rank):  {mean_mrr:.3f}")
    if not retrieval_only:
        print(f"Correct Refusal Rate:        {correct_refusals * 100:.1f}%")
    print(f"Median Search Latency (p50): {p50_latency * 1000:.1f}ms")
    print("=" * 65)

    # Per-bucket breakdown
    print("\n----------------- PER-BUCKET PERFORMANCE --------------------")
    buckets = set(r["type"] for r in results)
    for b in sorted(buckets):
        b_items = [r for r in results if r["type"] == b]
        if b == "unanswerable":
            b_acc = np.mean([1.0 if r["abstention_ok"] else 0.0 for r in b_items])
            print(f"  - {b.capitalize():15s} Count: {len(b_items):2d} | Correct Refusal: {b_acc * 100:.1f}%")
        else:
            b_hit = np.mean([r["hit@k"] for r in b_items])
            b_mrr = np.mean([r["mrr"] for r in b_items])
            print(f"  - {b.capitalize():15s} Count: {len(b_items):2d} | Hit@{top_k}: {b_hit * 100:.1f}% | MRR: {b_mrr:.3f}")
    print("=================================================================")

    return mean_hit, mean_mrr, p50_latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--top-k", type=int, default=4, help="Top-K to retrieve")
    parser.add_argument("-r", "--retrieval-only", action="store_true", help="Skip LLM generation, test search only")
    args = parser.parse_args()

    run_evaluation(top_k=args.top_k, retrieval_only=args.retrieval_only)