from abc import ABC, abstractmethod


class Chunker(ABC):
    """Abstract Base Class for all chunking strategies."""

    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    @abstractmethod
    def split(self, text: str, metadata: dict | None = None) -> list[dict]:
        """
        Splits raw text into a list of chunk dictionaries.
        Each dictionary should contain:
        {
            "text": str,           # The actual text to embed & search
            "heading_path": str,   # e.g. "Tutorial > Path Parameters"
            "char_count": int      # Length of chunk
        }
        """
        pass