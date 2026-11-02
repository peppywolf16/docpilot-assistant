from app.chunking.base import Chunker
from app.chunking.markdown import MarkdownChunker
from app.chunking.recursive import RecursiveChunker

CHUNKERS: dict[str, type[Chunker]] = {
    "markdown": MarkdownChunker,
    "recursive": RecursiveChunker,
}


def get_chunker(name: str = "markdown", chunk_size: int = 800, overlap: int = 100) -> Chunker:
    """Returns an instantiated chunker by name."""
    chunker_cls = CHUNKERS.get(name.lower())
    if not chunker_cls:
        raise ValueError(f"Unknown chunker '{name}'. Available: {list(CHUNKERS.keys())}")
    return chunker_cls(chunk_size=chunk_size, overlap=overlap)