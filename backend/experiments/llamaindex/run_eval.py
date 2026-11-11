import sys
import json
import time
from pathlib import Path
import numpy as np

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from experiments.llamaindex.pipeline import build_llamaindex_pipeline
from eval.metrics import calculate_hit_at_k, calculate_mrr, check_abstention

QUESTIONS_FILE = backend_dir / "eval" / "questions.jsonl"


def run_llamaindex_eval(top_k: int = 4):
    print("Building LlamaIndex retriever...")
    _, index = build_llamaindex_pipeline()
    retriever = index.as_retriever(similarity_top_k=top_k)

    questions = [json.loads(line) for line in QUESTIONS_FILE.read_text(encoding="utf-8").strip().split("\n")]
    print(f"\nRunning evaluation on {len(questions)} questions using LlamaIndex...\n")

    results = []
    latencies = []

    for i, q in enumerate(questions, start=1):
        q_id = q["id"]
        q_text = q["question"]
        gold_doc = q.get("gold_doc")
        gold_snippet = q.get("gold_passage")
        is_unanswerable = (q["type"] == "unanswerable")

        print(f"[{i:02d}/{len(questions)}] {q_id} ({q['type']}): {q_text[:45]}...")

        t0 = time.time()
        # Retrieve nodes via LlamaIndex
        source_nodes = retriever.retrieve(q_text)
        elapsed = time.time() - t0
        latencies.append(elapsed)

        # Convert LlamaIndex nodes to standard DocPilot passage format
        passages = []
        for rank, n in enumerate(source_nodes, start=1):
            file_name = Path(n.metadata.get("file_name", "")).name
            passages.append({
                "number": rank,
                "doc": file_name,
                "score": round(float(n.score or 0.0), 4),
                "text": n.text,
            })

        hit = calculate_hit_at_k(passages, gold_doc, gold_snippet, k=top_k)
        mrr = calculate_mrr(passages, gold_doc, gold_snippet, k=top_k)

        results.append({
            "id": q_id,
            "type": q["type"],
            "question": q_text,
            "gold_doc": gold_doc,
            "hit@k": hit,
            "mrr": mrr,
            "retrieved_docs": [p["doc"] for p in passages],
            "latency_sec": round(elapsed, 4),
        })

    answerable = [r for r in results if r["type"] != "unanswerable"]
    unanswerable = [r for r in results if r["type"] == "unanswerable"]

    mean_hit = np.mean([r["hit@k"] for r in answerable]) if answerable else 0.0
    mean_mrr = np.mean([r["mrr"] for r in answerable]) if answerable else 0.0
    p50_latency = float(np.median(latencies))

    print("\n" + "=" * 65)
    print("            LLAMAINDEX EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Total Questions Evaluated: {len(results)} ({len(answerable)} answerable)")
    print("-" * 65)
    print(f"Hit@{top_k} (Recall):          {mean_hit * 100:.1f}%")
    print(f"MRR (Mean Reciprocal Rank):  {mean_mrr:.3f}")
    print(f"Median Search Latency (p50): {p50_latency * 1000:.1f}ms")
    print("=" * 65)

    print("\n----------------- PER-BUCKET PERFORMANCE --------------------")
    buckets = set(r["type"] for r in results)
    for b in sorted(buckets):
        b_items = [r for r in results if r["type"] == b]
        if b != "unanswerable":
            b_hit = np.mean([r["hit@k"] for r in b_items])
            b_mrr = np.mean([r["mrr"] for r in b_items])
            print(f"  - {b.capitalize():15s} Count: {len(b_items):2d} | Hit@{top_k}: {b_hit * 100:.1f}% | MRR: {b_mrr:.3f}")
    print("=================================================================\n")

    return mean_hit, mean_mrr, p50_latency


if __name__ == "__main__":
    run_llamaindex_eval(top_k=4)