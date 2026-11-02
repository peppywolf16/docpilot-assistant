import sys
from pathlib import Path

# Add backend directory to path so imports work cleanly
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.config import SAMPLE_DOCS_DIR
from app.loader import load_text_from_file
from app.store import add_document, list_documents, CHUNKS


def ingest_folder(folder_path: Path):
    """Scans a directory and indexes all .md, .txt, and .pdf files."""
    valid_extensions = [".md", ".txt", ".pdf"]
    files = [f for f in folder_path.glob("*") if f.suffix.lower() in valid_extensions]

    if not files:
        print(f"No documents found in {folder_path}")
        return

    print(f"Found {len(files)} documents in {folder_path.name}/. Ingesting...")

    total_new_chunks = 0
    for file_path in files:
        try:
            text = load_text_from_file(file_path)
            chunks_added = add_document(file_path.name, text)
            total_new_chunks += chunks_added
            print(f"  ✓ {file_path.name} ({chunks_added} chunks)")
        except Exception as e:
            print(f"  ✗ Failed to ingest {file_path.name}: {e}")

    print("\n" + "=" * 45)
    print(f"Done! Added {total_new_chunks} chunks.")
    print(f"Total chunks in persistent store: {len(CHUNKS)}")
    print(f"Indexed documents: {list_documents()}")
    print("=" * 45)


if __name__ == "__main__":
    ingest_folder(SAMPLE_DOCS_DIR)