import numpy as np
from app.embeddings.base import Embedder
from app.llm import embed as gemini_embed


class GeminiEmbedder(Embedder):
    """Google Gemini embedding model (default: 3072 dimensions, optionally truncated)."""

    def __init__(self, target_dim: int | None = None):
        self._target_dim = target_dim

    @property
    def dimension(self) -> int:
        return self._target_dim if self._target_dim else 3072

    def _truncate_and_normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Slices vectors down to target_dim and re-normalizes (Matryoshka Representation Learning)."""
        if self._target_dim and self._target_dim < vectors.shape[1]:
            vectors = vectors[:, : self._target_dim]
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            return vectors / norms
        return vectors

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raw_vecs = gemini_embed(texts, mode="document")
        return self._truncate_and_normalize(raw_vecs)

    def embed_query(self, query: str) -> np.ndarray:
        raw_vecs = gemini_embed([query], mode="query")
        return self._truncate_and_normalize(raw_vecs)[0]