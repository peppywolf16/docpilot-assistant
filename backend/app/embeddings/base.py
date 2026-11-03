from abc import ABC, abstractmethod
import numpy as np


class Embedder(ABC):
    """Abstract Base Class for all embedding models."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embeds a list of document chunks. Returns an L2-normalized 2D float32 array."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single search query. Returns an L2-normalized 1D float32 array."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the vector dimension size (e.g. 384, 768, 3072)."""
        pass