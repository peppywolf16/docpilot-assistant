import io
from pathlib import Path
from pypdf import PdfReader


def load_text_from_file(file_path: Path | str) -> str:
    """Reads text from a local .txt, .md, or .pdf file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        reader = PdfReader(str(path))
        text_parts = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(text_parts).strip()
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def load_text_from_bytes(filename: str, content: bytes) -> str:
    """Extracts text from uploaded bytes (.txt, .md, or .pdf)."""
    suffix = Path(filename).suffix.lower()

    if suffix in [".txt", ".md"]:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("latin-1")
    elif suffix == ".pdf":
        pdf_file = io.BytesIO(content)
        reader = PdfReader(pdf_file)
        text_parts = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(text_parts).strip()
    else:
        raise ValueError(f"Unsupported file format: {suffix}")