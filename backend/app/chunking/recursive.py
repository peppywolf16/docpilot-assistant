import re
from app.chunking.base import Chunker
from app.chunking.cleaner import clean_markdown


class RecursiveChunker(Chunker):
    """Splits by paragraphs and sentences with overlap."""

    def split(self, text: str, metadata: dict | None = None) -> list[dict]:
        text = clean_markdown(text)
        if not text:
            return []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = ""
        position = 1

        for p in paragraphs:
            if not current_chunk:
                current_chunk = p
            elif len(current_chunk) + len(p) + 2 <= self.chunk_size:
                current_chunk += "\n\n" + p
            else:
                chunks.append({
                    "text": current_chunk.strip(),
                    "heading_path": "N/A",
                    "char_count": len(current_chunk.strip()),
                    "position": position,
                })
                position += 1
                current_chunk = p

        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "heading_path": "N/A",
                "char_count": len(current_chunk.strip()),
                "position": position,
            })

        return chunks