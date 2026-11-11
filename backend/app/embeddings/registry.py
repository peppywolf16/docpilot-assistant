from pathlib import Path
from app.embeddings.base import Embedder
from app.embeddings.gemini import GeminiEmbedder
from app.embeddings.bge import BGEEmbedder

# Path to local fine-tuned model
FINETUNED_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "bge-small-fastapi-finetuned"


def get_embedder(name: str = "gemini", target_dim: int | None = None) -> Embedder:
    """Factory to get the configured embedding model."""
    name_lower = name.lower()
    if name_lower == "gemini":
        return GeminiEmbedder(target_dim=target_dim)
    elif name_lower in ["bge", "bge-small", "local"]:
        return BGEEmbedder(model_name="BAAI/bge-small-en-v1.5")
    elif name_lower in ["bge-finetuned", "finetuned"]:
        if not FINETUNED_PATH.exists():
            raise FileNotFoundError(f"Fine-tuned model not found at {FINETUNED_PATH}. Run scripts/train_embedding.py first!")
        return BGEEmbedder(model_name=str(FINETUNED_PATH))
    else:
        raise ValueError(f"Unknown embedder '{name}'. Choose 'gemini', 'bge-small', or 'bge-finetuned'.")


