import atexit
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import (
    INDEX_DIR,
    DEFAULT_TOP_K,
    EMBEDDER_TYPE,
    EMBEDDING_DIM,
    CHUNKER_STRATEGY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from app.chunking import get_chunker
from app.embeddings import get_embedder

# Qdrant local persistent path (one folder per embedder, since INDEX_DIR is model-specific)
QDRANT_DIR = INDEX_DIR / "qdrant_storage"
QDRANT_DIR.mkdir(parents=True, exist_ok=True)

# Local on-disk client (no Docker required)
client = QdrantClient(path=str(QDRANT_DIR))
COLLECTION_NAME = "docpilot_docs"


def _close_client():
    """Close the local client cleanly before interpreter shutdown."""
    try:
        client.close()
    except Exception:
        pass


atexit.register(_close_client)

# Instantiate embedder
embedder = get_embedder(EMBEDDER_TYPE, target_dim=EMBEDDING_DIM)
VECTOR_SIZE = embedder.dimension


def _ensure_collection():
    """Ensures the Qdrant collection exists with the correct vector size."""
    collections = client.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )


_ensure_collection()


def _doc_filter(doc_name: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="doc_name", match=models.MatchValue(value=doc_name))]
    )


def _scroll_all(scroll_filter: models.Filter | None = None, with_payload=False) -> list:
    """Scrolls through every matching point, following pagination."""
    all_records = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=scroll_filter,
            with_payload=with_payload,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        all_records.extend(records)
        if offset is None:
            break
    return all_records


def list_documents() -> list[str]:
    """Returns a sorted list of unique document names in Qdrant."""
    _ensure_collection()
    records = _scroll_all(with_payload=["doc_name"])
    docs = {
        r.payload["doc_name"]
        for r in records
        if r.payload and "doc_name" in r.payload
    }
    return sorted(docs)


def delete_document(doc_name: str) -> bool:
    """Deletes all chunks belonging to a document. Returns True if anything was deleted."""
    _ensure_collection()

    records = _scroll_all(scroll_filter=_doc_filter(doc_name))
    point_ids = [r.id for r in records]
    if not point_ids:
        return False

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.PointIdsList(points=point_ids),
    )
    return True


def add_document(name: str, text: str, version: str = "v1.0") -> int:
    """Chunks, embeds, and upserts a document into Qdrant with metadata. Returns chunk count."""
    _ensure_collection()

    # 1. Deduplication: remove the old copy of this document if it exists
    delete_document(name)

    # 2. Chunk text
    chunker = get_chunker(CHUNKER_STRATEGY, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    chunk_dicts = chunker.split(text)
    if not chunk_dicts:
        return 0

    texts = [c["text"] for c in chunk_dicts]

    # 3. Embed chunks
    vectors = embedder.embed_documents(texts)

    # 4. Build points with UUID ids (safe against collisions after deletions)
    points = []
    for i, (c, vec) in enumerate(zip(chunk_dicts, vectors)):
        payload = {
            "doc_name": name,
            "chunk_id": i + 1,  # position within this document
            "text": c["text"],
            "heading_path": c.get("heading_path", "N/A"),
            "position": c.get("position", i + 1),
            "version": version,
        }
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec.tolist(),
                payload=payload,
            )
        )

    # 5. Upsert into Qdrant
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    return len(chunk_dicts)


def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    doc_filter: str | None = None,
    version_filter: str | None = None,
) -> list[dict]:
    """
    Vector search with optional metadata filters
    (restrict to one document and/or one version).
    """
    _ensure_collection()

    q_vec = embedder.embed_query(query)

    conditions = []
    if doc_filter:
        conditions.append(
            models.FieldCondition(key="doc_name", match=models.MatchValue(value=doc_filter))
        )
    if version_filter:
        conditions.append(
            models.FieldCondition(key="version", match=models.MatchValue(value=version_filter))
        )
    q_filter = models.Filter(must=conditions) if conditions else None

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec.tolist(),
        query_filter=q_filter,
        limit=top_k,
    ).points

    results = []
    for rank, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        results.append({
            "number": rank,
            "doc": payload.get("doc_name", "unknown"),
            "chunk_id": payload.get("chunk_id", 0),
            "score": round(float(hit.score), 4),
            "text": payload.get("text", ""),
        })

    return results