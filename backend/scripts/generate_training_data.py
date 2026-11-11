import sys
import json
import random
from pathlib import Path

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.store_qdrant import client, COLLECTION_NAME, _scroll_all
from app.embeddings.bge import BGEEmbedder
from app.llm import generate

EVAL_QUESTIONS_FILE = backend_dir / "eval" / "questions.jsonl"
OUTPUT_DIR = backend_dir / "data" / "finetune"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_excluded_passages() -> set[str]:
    """Collects gold snippets from our 20 eval questions so we NEVER leak test data into training!"""
    excluded = set()
    if EVAL_QUESTIONS_FILE.exists():
        for line in EVAL_QUESTIONS_FILE.read_text(encoding="utf-8").strip().split("\n"):
            data = json.loads(line)
            if data.get("gold_passage"):
                excluded.add(data["gold_passage"].lower())
    return excluded


def generate_questions_for_chunk(chunk_text: str) -> list[str]:
    """Uses Gemini to generate 2 realistic developer search questions for a documentation chunk."""
    prompt = f"""You are a software engineer searching documentation.
Read this documentation passage and generate 2 realistic, distinct search questions that a developer would ask where this passage provides the exact answer.
Do not use generic questions like 'What is this document about?'. Use specific terminology found in the text.

Passage:
{chunk_text[:800]}

Respond ONLY with 2 questions, one per line. Do not number them:"""

    response = generate(prompt).strip()
    questions = [line.strip("- *0123456789. ") for line in response.split("\n") if line.strip()]
    return questions[:2]


def build_triplets():
    print("Fetching indexed chunks from Qdrant...")
    records = _scroll_all(with_payload=True)
    if not records:
        print("No chunks found in Qdrant! Please run scripts/ingest_folder.py first.")
        return

    excluded_snippets = get_excluded_passages()
    base_embedder = BGEEmbedder()

    # Filter out chunks that contain test-set gold passages to prevent data contamination
    trainable_chunks = []
    for r in records:
        text = r.payload.get("text", "")
        # Skip if chunk contains any eval snippet
        if any(snip in text.lower() for snip in excluded_snippets):
            continue
        # Skip very short fragments (< 100 chars)
        if len(text) < 100:
            continue
        trainable_chunks.append({
            "id": r.id,
            "doc_name": r.payload.get("doc_name"),
            "text": text
        })

    print(f"Total candidate chunks for training: {len(trainable_chunks)}")

    # Pre-embed all chunks with base BGE for fast hard-negative mining
    print("Embedding chunks with base BGE-small for hard negative mining...")
    all_texts = [c["text"] for c in trainable_chunks]
    all_vectors = base_embedder.embed_documents(all_texts)

    triplets = []
    print("\nGenerating synthetic queries and mining hard negatives...")

    # Sample up to 35 diverse chunks to keep dataset compact and fast
    sampled_chunks = random.sample(trainable_chunks, min(35, len(trainable_chunks)))

    for idx, chunk in enumerate(sampled_chunks, start=1):
        positive_text = chunk["text"]
        print(f"[{idx:02d}/{len(sampled_chunks)}] Processing chunk from {chunk['doc_name']}...")

        # 1. Generate 2 synthetic queries
        queries = generate_questions_for_chunk(positive_text)

        # 2. Mine hard negatives: find chunks with high cosine similarity that are NOT this chunk
        for query in queries:
            q_vec = base_embedder.embed_query(query)
            scores = all_vectors @ q_vec
            
            # Rank descending
            top_indices = scores.argsort()[::-1]

            # Find a hard negative: a chunk ranked 2 to 10 from a DIFFERENT section
            hard_negative = None
            for rank_idx in top_indices[1:10]:
                candidate = trainable_chunks[rank_idx]
                # Pick a candidate that isn't identical or trivial
                if candidate["text"] != positive_text:
                    hard_negative = candidate["text"]
                    break

            if hard_negative:
                triplets.append({
                    "query": query,
                    "positive": positive_text,
                    "negative": hard_negative
                })

    print(f"\nGenerated {len(triplets)} high-quality triplets!")

    # Split 90% train / 10% validation
    random.shuffle(triplets)
    split_idx = int(len(triplets) * 0.9)
    train_set = triplets[:split_idx]
    val_set = triplets[split_idx:]

    train_file = OUTPUT_DIR / "train_triplets.jsonl"
    val_file = OUTPUT_DIR / "val_triplets.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for t in train_set:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for t in val_set:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"Saved {len(train_set)} training triplets to {train_file.relative_to(backend_dir)}")
    print(f"Saved {len(val_set)} validation triplets to {val_file.relative_to(backend_dir)}")


if __name__ == "__main__":
    build_triplets()