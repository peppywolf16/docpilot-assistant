import atexit
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import USE_PARENT_CHILD
from app.chunking.parent_child import split_parent_child

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
    """Chunks, embeds, and upserts a document into Qdrant with metadata."""
    _ensure_collection()
    delete_document(name)

    existing_count = client.count(collection_name=COLLECTION_NAME, exact=True).count
    points = []

    if USE_PARENT_CHILD:
        # Parent-Document Mode: embed small child chunks, store parent section in payload
        parents_lookup, child_dicts = split_parent_child(text, name)
        if not child_dicts:
            return 0

        texts_to_embed = [c["text"] for c in child_dicts]
        vectors = embedder.embed_documents(texts_to_embed)

        for i, (c, vec) in enumerate(zip(child_dicts, vectors)):
            point_id = existing_count + i + 1
            payload = {
                "doc_name": name,
                "chunk_id": point_id,
                "text": c["text"],                       # child text (for search/BM25)
                "parent_id": c["parent_id"],
                "parent_text": parents_lookup[c["parent_id"]], # full parent context (for LLM)
                "heading_path": c.get("heading_path", "N/A"),
                "version": version,
            }
            points.append(models.PointStruct(id=str(uuid.uuid4()), vector=vec.tolist(), payload=payload))

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        return len(child_dicts)

    else:
        # Standard Chunking Mode
        chunker = get_chunker(CHUNKER_STRATEGY, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        chunk_dicts = chunker.split(text)
        if not chunk_dicts:
            return 0

        texts = [c["text"] for c in chunk_dicts]
        vectors = embedder.embed_documents(texts)

        for i, (c, vec) in enumerate(zip(chunk_dicts, vectors)):
            point_id = existing_count + i + 1
            payload = {
                "doc_name": name,
                "chunk_id": point_id,
                "text": c["text"],
                "heading_path": c.get("heading_path", "N/A"),
                "position": c.get("position", i + 1),
                "version": version,
            }
            points.append(models.PointStruct(id=str(uuid.uuid4()), vector=vec.tolist(), payload=payload))

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        return len(chunk_dicts)

def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    doc_filter: str | None = None,
    version_filter: str | None = None,
    user_tenant: str | None = None,       # <--- Mandatory ACL filter
    allowed_role: str | None = None,       # <--- Mandatory RBAC filter
) -> list[dict]:
    """
    Vector search with mandatory metadata and multi-tenancy ACL filters.
    """
    _ensure_collection()
    q_vec = embedder.embed_query(query)

    conditions = []
    if doc_filter:
        conditions.append(models.FieldCondition(key="doc_name", match=models.MatchValue(value=doc_filter)))
    if version_filter:
        conditions.append(models.FieldCondition(key="version", match=models.MatchValue(value=version_filter)))
    
    # Enterprise Access Control (Multi-Tenancy & RBAC)
    if user_tenant:
        conditions.append(models.FieldCondition(key="tenant", match=models.MatchValue(value=user_tenant)))
    if allowed_role:
        # User role must be present in the document's allowed_roles list
        conditions.append(models.FieldCondition(key="allowed_roles", match=models.MatchAny(any=[allowed_role])))

    q_filter = models.Filter(must=conditions) if conditions else None

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec.tolist(),
        query_filter=q_filter,
        limit=top_k * 3 if USE_PARENT_CHILD else top_k,
    ).points

    results = []
    seen_parents = set()

    for hit in hits:
        payload = hit.payload or {}
        
        if USE_PARENT_CHILD:
            parent_id = payload.get("parent_id")
            if parent_id and parent_id in seen_parents:
                continue
            if parent_id:
                seen_parents.add(parent_id)
            passage_text = payload.get("parent_text", payload.get("text", ""))
        else:
            passage_text = payload.get("text", "")

        results.append({
            "number": len(results) + 1,
            "doc": payload.get("doc_name", "unknown"),
            "chunk_id": payload.get("chunk_id", 0),
            "score": round(float(hit.score), 4),
            "text": passage_text,
            "tenant": payload.get("tenant", "public"),
            "allowed_roles": payload.get("allowed_roles", ["all"]),
        })

        if len(results) >= top_k:
            break

    return results

