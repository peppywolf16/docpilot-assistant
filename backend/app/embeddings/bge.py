import numpy as np
from sentence_transformers import SentenceTransformer
from app.embeddings.base import Embedder


class BGEEmbedder(Embedder):
    """Local open-source BAAI/bge-small-en-v1.5 running on CPU (384 dimensions)."""

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        # Loads to CPU / local cache automatically
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return self.model.get_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        # normalize_embeddings=True automatically L2-normalizes
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        prefixed_query = self.QUERY_PREFIX + query
        vec = self.model.encode([prefixed_query], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.array(vec, dtype=np.float32)