from app.embeddings.base import Embedder
from app.embeddings.gemini import GeminiEmbedder
from app.embeddings.bge import BGEEmbedder


def get_embedder(name: str = "gemini", target_dim: int | None = None) -> Embedder:
    """Factory to get the configured embedding model."""
    name_lower = name.lower()
    if name_lower == "gemini":
        return GeminiEmbedder(target_dim=target_dim)
    elif name_lower in ["bge", "bge-small", "local"]:
        return BGEEmbedder()
    else:
        raise ValueError(f"Unknown embedder '{name}'. Choose 'gemini' or 'bge-small'.")