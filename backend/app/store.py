import json
from pathlib import Path
import numpy as np
from app.config import INDEX_DIR, DEFAULT_TOP_K, CHUNKER_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP
from app.chunking import get_chunker
from app.llm import embed

VECTORS_FILE = INDEX_DIR / "vectors.npy"
CHUNKS_FILE = INDEX_DIR / "chunks.json"

# In-memory working copies
CHUNKS: list[dict] = []
VECTORS: np.ndarray | None = None

def save_index():
    """
    Persistes in-memory chunks and vectors to data/index/.
    """
    global CHUNKS, VECTORS
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(CHUNKS, f, ensure_ascii=False, indent=2)

    if VECTORS is not None and len(VECTORS) > 0:
        np.save(VECTORS_FILE, VECTORS)
    elif VECTORS_FILE.exists():
        VECTORS_FILE.unlink()  # Remove empty/stale vector file


def load_index():
    """
    Loads saved chunks and vectors from disk on startup.
    """
    global CHUNKS, VECTORS

    if CHUNKS_FILE.exists() and VECTORS_FILE.exists():
        try:
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                CHUNKS = json.load(f)
            VECTORS = np.load(VECTORS_FILE)
            print(f"Loaded index from disk! {len(CHUNKS)} chunks ready.")
            return
        except Exception as e:
            print(f"Warning: Failed to load index from disk ({e}). Starting fresh.")

    CHUNKS = []
    VECTORS = None

# Load index immediately when store.py is imported
load_index()

def list_documents() -> list[str]:
    """Returns a sorted list of unique document names currently in the index."""
    return sorted(list({c["doc_name"] for c in CHUNKS}))

def delete_document(doc_name: str) -> bool:
    """Removes a document's chunks and vector rows from the index. Returns True if deleted."""
    global CHUNKS, VECTORS

    if not CHUNKS or VECTORS is None:
        return False

    keep_indices = [i for i, c in enumerate(CHUNKS) if c["doc_name"] != doc_name]

    # If all chunks are being kept, document wasn't found
    if len(keep_indices) == len(CHUNKS):
        return False

    if len(keep_indices) == 0:
        # All documents deleted
        CHUNKS = []
        VECTORS = None
    else:
        CHUNKS = [CHUNKS[i] for i in keep_indices]
        VECTORS = VECTORS[keep_indices]

    save_index()
    return True

def add_document(name: str, text: str) -> int:
    """
    Splits text using the configured Chunker strategy, embeds chunks,
    stores rich metadata, and saves index to disk.
    """
    global CHUNKS, VECTORS

    # 1. Deduplication: Delete old version if it exists
    delete_document(name)

    # 2. Chunk using the configured strategy
    chunker = get_chunker(CHUNKER_STRATEGY, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    chunk_dicts = chunker.split(text)
    if not chunk_dicts:
        return 0

    new_text_strings = [c["text"] for c in chunk_dicts]

    # 3. Embed the new chunks
    new_vectors = embed(new_text_strings, mode="document")

    # 4. Append to chunk list with rich metadata
    start_id = len(CHUNKS) + 1
    for i, c in enumerate(chunk_dicts):
        CHUNKS.append({
            "doc_name": name,
            "chunk_id": start_id + i,
            "text": c["text"],
            "heading_path": c.get("heading_path", "N/A"),
            "position": c.get("position", i + 1),
        })

    # 5. Stack into NumPy matrix
    if VECTORS is None or len(VECTORS) == 0:
        VECTORS = new_vectors
    else:
        VECTORS = np.vstack([VECTORS, new_vectors])

    # 6. Save changes to disk
    save_index()

    return len(chunk_dicts)

def search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Finds top_k most similar chunks for a query using cosine similarity."""
    global CHUNKS, VECTORS

    if VECTORS is None or len(CHUNKS) == 0:
        return []

    # Embed query
    q_vec = embed([query], mode="query")[0]

    # Cosine similarity (dot product on normalized vectors)
    scores = VECTORS @ q_vec

    actual_k = min(top_k, len(CHUNKS))
    top_indices = np.argsort(scores)[::-1][:actual_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        chunk_data = CHUNKS[idx]
        results.append({
            "number": rank,
            "doc": chunk_data["doc_name"],       # <--- MUST BE chunk_data["doc_name"]!
            "chunk_id": chunk_data["chunk_id"],   # <--- chunk_id goes here!
            "score": round(float(scores[idx]), 4),
            "text": chunk_data["text"],
        })

    return results