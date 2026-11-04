import sys
import json
import glob
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

def analyze_latest_run():
    runs = sorted(glob.glob(str(backend_dir / "results" / "runs" / "*.json")))
    if not runs:
        print("No run reports found in results/runs/")
        return
    
    latest_run = runs[-1]
    data = json.loads(Path(latest_run).read_text(encoding="utf-8"))
    
    print(f"\nAnalyzing failure modes for run: {Path(latest_run).name}\n" + "="*50)
    
    failures = [r for r in data["detailed_results"] if r["hit@k"] == 0.0 and r["type"] != "unanswerable"]
    
    if not failures:
        print("Zero retrieval failures on answerable questions! 🎉")
        return

    print(f"Total Failed Questions: {len(failures)}\n")
    for f in failures:
        print(f"[{f['id']}] {f['question']}")
        print(f"  - Expected Gold Doc: {f['gold_doc']}")
        print(f"  - Retrieved Docs:    {f['retrieved_docs']}")
        print(f"  - Failure Type:      Check if gold doc was in top 20 candidate pool before reranking.\n")

if __name__ == "__main__":
    analyze_latest_run()