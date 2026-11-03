import numpy as np
from rank_bm25 import BM25Okapi
from app.store_qdrant import client, COLLECTION_NAME, _scroll_all
from app.embeddings import get_embedder
from app.config import EMBEDDER_TYPE, EMBEDDING_DIM, DEFAULT_TOP_K

embedder = get_embedder(EMBEDDER_TYPE, target_dim=EMBEDDING_DIM)


def reciprocal_rank_fusion(dense_hits: list[dict], bm25_hits: list[dict], k: int = 60) -> list[dict]:
    """
    Fuses two ranked lists using Reciprocal Rank Fusion (RRF).
    RRF Score = sum(1 / (k + rank_in_list)) for each list where the doc appears.
    """
    scores = {}
    doc_lookup = {}

    def _add_hits(hits):
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit["chunk_id"]
            doc_name = hit["doc"]
            doc_lookup[(doc_name, chunk_id)] = hit
            
            if (doc_name, chunk_id) not in scores:
                scores[(doc_name, chunk_id)] = 0.0
            
            # RRF formula
            scores[(doc_name, chunk_id)] += 1.0 / (k + rank)

    _add_hits(dense_hits)
    _add_hits(bm25_hits)

    # Sort by RRF score descending
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    fused_results = []
    for rank, ((doc_name, chunk_id), rrf_score) in enumerate(sorted_docs, start=1):
        hit = doc_lookup[(doc_name, chunk_id)]
        hit["number"] = rank
        hit["score"] = round(float(rrf_score), 4)
        fused_results.append(hit)

    return fused_results


def hybrid_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    doc_filter: str | None = None,
    alpha: float = 0.5  # Weight between dense and sparse if needed
) -> list[dict]:
    """
    Performs Hybrid Search:
    1. Dense vector search via Qdrant.
    2. Sparse keyword search via BM25 over all loaded chunks.
    3. Fuses results using Reciprocal Rank Fusion (RRF).
    """
    # 1. Dense Search (Qdrant)
    q_vec = embedder.embed_query(query)
    
    q_filter = None
    if doc_filter:
        from qdrant_client.http import models
        q_filter = models.Filter(
            must=[models.FieldCondition(key="doc_name", match=models.MatchValue(value=doc_filter))]
        )

    dense_response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec.tolist(),
        query_filter=q_filter,
        limit=top_k * 2,  # fetch a larger pool for fusion
    ).points

    dense_hits = []
    for hit in dense_response:
        payload = hit.payload or {}
        dense_hits.append({
            "doc": payload.get("doc_name", "unknown"),
            "chunk_id": payload.get("chunk_id", 0),
            "score": float(hit.score),
            "text": payload.get("text", ""),
        })

    # 2. Sparse Search (BM25)
    # Scroll all chunks from Qdrant to build BM25 index on the fly
    all_records = _scroll_all(with_payload=True)
    if not all_records:
        return dense_hits[:top_k]

    # Filter records if doc_filter is active
    filtered_records = []
    for r in all_records:
        payload = r.payload or {}
        if doc_filter and payload.get("doc_name") != doc_filter:
            continue
        filtered_records.append(payload)

    if not filtered_records:
        return dense_hits[:top_k]

    # Tokenize corpus for BM25 (simple whitespace / lowercase tokenization)
    tokenized_corpus = [doc["text"].lower().split() for doc in filtered_records]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Get top indices from BM25
    top_bm25_indices = np.argsort(bm25_scores)[::-1][: top_k * 2]

    bm25_hits = []
    for idx in top_bm25_indices:
        if bm25_scores[idx] <= 0:
            continue  # Skip zero-score matches
        doc = filtered_records[idx]
        bm25_hits.append({
            "doc": doc.get("doc_name", "unknown"),
            "chunk_id": doc.get("chunk_id", 0),
            "score": float(bm25_scores[idx]),
            "text": doc.get("text", ""),
        })

    # 3. Fuse via RRF
    fused = reciprocal_rank_fusion(dense_hits, bm25_hits)
    return fused[:top_k]