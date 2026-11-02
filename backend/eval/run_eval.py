import sys
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np

# Ensure backend root is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.store import search, CHUNKS
from app.llm import generate
from eval.metrics import calculate_hit_at_k, calculate_mrr, check_abstention

QUESTIONS_FILE = backend_dir / "eval" / "questions.jsonl"
RUNS_DIR = backend_dir / "results" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def run_evaluation(top_k: int = 4):
    if not QUESTIONS_FILE.exists():
        print(f"Error: {QUESTIONS_FILE} not found!")
        return

    questions = [json.loads(line) for line in QUESTIONS_FILE.read_text(encoding="utf-8").strip().split("\n")]
    print(f"Running evaluation on {len(questions)} questions against {len(CHUNKS)} indexed chunks...")
    print("Pacing: Sleeping 3.5s between uncached questions to respect the 15 RPM free-tier limit.\n")

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

        # 1. Retrieve
        passages = search(q_text, top_k=top_k)

        # 2. Build prompt & Generate
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
        elapsed = time.time() - t0
        latencies.append(elapsed)

        # 3. Calculate metrics
        hit = calculate_hit_at_k(passages, gold_doc, gold_snippet, k=top_k)
        mrr = calculate_mrr(passages, gold_doc, gold_snippet, k=top_k)
        abstain_ok, abstain_reason = check_abstention(answer, is_unanswerable)

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
            "latency_sec": round(elapsed, 3),
            "answer": answer.strip(),
        })

        # If it was an actual API call (took > 0.5s), sleep to respect 15 RPM quota
        if elapsed > 0.5 and i < len(questions):
            time.sleep(3.5)

    # Summary Metrics Calculation
    answerable = [r for r in results if r["type"] != "unanswerable"]
    unanswerable = [r for r in results if r["type"] == "unanswerable"]

    mean_hit = np.mean([r["hit@k"] for r in answerable]) if answerable else 0.0
    mean_mrr = np.mean([r["mrr"] for r in answerable]) if answerable else 0.0
    correct_refusals = np.mean([1.0 if r["abstention_ok"] else 0.0 for r in unanswerable]) if unanswerable else 0.0
    answer_fidelity = np.mean([1.0 if r["abstention_ok"] else 0.0 for r in answerable]) if answerable else 0.0
    p50_latency = float(np.median(latencies))

    # Print Summary Table
    print("\n" + "=" * 65)
    print("                 DOCPILOT EVALUATION SUMMARY                 ")
    print("=" * 65)
    print(f"Total Questions Evaluated: {len(results)}")
    print(f"  - Answerable:            {len(answerable)}")
    print(f"  - Unanswerable:          {len(unanswerable)}")
    print("-" * 65)
    print(f"Hit@{top_k} (Recall):          {mean_hit * 100:.1f}%")
    print(f"MRR (Mean Reciprocal Rank):  {mean_mrr:.3f}")
    print(f"Correct Refusal Rate:        {correct_refusals * 100:.1f}%")
    print(f"Answer Non-Refusal Rate:     {answer_fidelity * 100:.1f}%")
    print(f"Median Latency (p50):        {p50_latency:.3f}s")
    print("=" * 65)

    # Save detailed JSON run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = RUNS_DIR / f"run_{run_timestamp}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": run_timestamp,
            "summary": {
                "hit@k": round(mean_hit, 4),
                "mrr": round(mean_mrr, 4),
                "correct_refusals": round(correct_refusals, 4),
                "answer_fidelity": round(answer_fidelity, 4),
                "p50_latency_sec": round(p50_latency, 3),
            },
            "detailed_results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed report saved to: {report_file.relative_to(backend_dir)}")


if __name__ == "__main__":
    run_evaluation(top_k=4)