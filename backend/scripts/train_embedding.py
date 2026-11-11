import sys
import json
import time
from pathlib import Path
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator


# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

DATA_DIR = backend_dir / "data" / "finetune"
MODEL_OUTPUT_DIR = backend_dir / "data" / "models" / "bge-small-fastapi-finetuned"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_triplets(file_path: Path) -> list[InputExample]:
    """Loads JSONL triplets into SentenceTransformer InputExamples."""
    examples = []
    if not file_path.exists():
        raise FileNotFoundError(f"Missing {file_path}")

    for line in file_path.read_text(encoding="utf-8").strip().split("\n"):
        if not line.strip():
            continue
        data = json.loads(line)
        # BGE query instruction prefix
        prefixed_query = f"Represent this sentence for searching relevant passages: {data['query']}"
        examples.append(
            InputExample(
                texts=[prefixed_query, data["positive"], data["negative"]]
            )
        )
    return examples


def train_model():
    print(f"Loading base model: {BASE_MODEL_NAME}...")
    model = SentenceTransformer(BASE_MODEL_NAME)

    print("Loading training and validation triplets...")
    train_examples = load_triplets(DATA_DIR / "train_triplets.jsonl")
    val_examples = load_triplets(DATA_DIR / "val_triplets.jsonl")

    print(f"Loaded {len(train_examples)} train examples, {len(val_examples)} validation examples.")

    # Batch size 8 is fast and memory-safe for CPU
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

    ## MultipleNegativesRankingLoss optimizes contrastive separation
    train_loss = MultipleNegativesRankingLoss(model)

    # Evaluator on validation triplets
    val_queries = {f"q_{i}": ex.texts[0] for i, ex in enumerate(val_examples)}
    val_corpus = {f"doc_{i}": ex.texts[1] for i, ex in enumerate(val_examples)}
    val_relevant_docs = {f"q_{i}": {f"doc_{i}"} for i in range(len(val_examples))}

    evaluator = InformationRetrievalEvaluator(
        queries=val_queries,
        corpus=val_corpus,
        relevant_docs=val_relevant_docs,
        name="fastapi_val_evaluator",
        show_progress_bar=False,
    )

    print("\nStarting fine-tuning (2 epochs)...")
    t0 = time.time()

    # Train model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=2,
        evaluation_steps=10,
        warmup_steps=int(len(train_dataloader) * 0.1),
        output_path=str(MODEL_OUTPUT_DIR),
        show_progress_bar=True,
    )

    elapsed = time.time() - t0
    print("\n" + "=" * 55)
    print(f"Training completed in {elapsed:.1f}s ({elapsed / 60:.2f} minutes)!")
    print(f"Fine-tuned model saved to: {MODEL_OUTPUT_DIR.relative_to(backend_dir)}")
    print("=" * 55)


if __name__ == "__main__":
    train_model()