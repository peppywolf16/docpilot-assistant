import numpy as np
from rank_bm25 import BM25Okapi
from app.store_qdrant import client, COLLECTION_NAME, _scroll_all, list_documents
from app.embeddings import get_embedder
from app.config import EMBEDDER_TYPE, EMBEDDING_DIM, DEFAULT_TOP_K
from app.reranker import rerank
from app.query.decomposition import decompose_query
from app.query.filter_extractor import extract_query_filters
from app.query.router import route_query

embedder = get_embedder(EMBEDDER_TYPE, target_dim=EMBEDDING_DIM)


def reciprocal_rank_fusion(dense_hits: list[dict], bm25_hits: list[dict], k: int = 60) -> list[dict]:
    """Fuses two ranked lists using Reciprocal Rank Fusion (RRF)."""
    scores = {}
    doc_lookup = {}

    def _add_hits(hits):
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit["chunk_id"]
            doc_name = hit["doc"]
            doc_lookup[(doc_name, chunk_id)] = hit
            
            if (doc_name, chunk_id) not in scores:
                scores[(doc_name, chunk_id)] = 0.0
            
            scores[(doc_name, chunk_id)] += 1.0 / (k + rank)

    _add_hits(dense_hits)
    _add_hits(bm25_hits)

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
    doc_filter: str | None = None
) -> list[dict]:
    """Performs Hybrid Search (Dense Qdrant + Sparse BM25 + RRF)."""
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
        limit=top_k * 2,
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

    all_records = _scroll_all(with_payload=True)
    if not all_records:
        return dense_hits[:top_k]

    filtered_records = []
    for r in all_records:
        payload = r.payload or {}
        if doc_filter and payload.get("doc_name") != doc_filter:
            continue
        filtered_records.append(payload)

    if not filtered_records:
        return dense_hits[:top_k]

    tokenized_corpus = [doc["text"].lower().split() for doc in filtered_records]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][: top_k * 2]

    bm25_hits = []
    for idx in top_bm25_indices:
        if bm25_scores[idx] <= 0:
            continue
        doc = filtered_records[idx]
        bm25_hits.append({
            "doc": doc.get("doc_name", "unknown"),
            "chunk_id": doc.get("chunk_id", 0),
            "score": float(bm25_scores[idx]),
            "text": doc.get("text", ""),
        })

    fused = reciprocal_rank_fusion(dense_hits, bm25_hits)
    return fused[:top_k]


def smart_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    doc_filter: str | None = None,
    candidate_pool_size: int = 20
) -> list[dict]:
    """
    Day 10 Smart Search Pipeline:
    1. Filter Extraction: detects file references and applies doc_filter.
    2. Intent Routing & Decomposition: breaks multi-part queries into sub-questions.
    3. Hybrid retrieval + RRF fusion for each sub-question.
    4. Cross-Encoder re-ranking against original query.
    5. Fallback: if filtered search returns < 2 chunks, retries unfiltered.
    """
    available_docs = list_documents()
    
    # 1. Automatic Filter Extraction if no manual filter provided
    active_filter = doc_filter
    clean_query = query
    if not active_filter:
        extracted_filter, clean_query = extract_query_filters(query, available_docs)
        active_filter = extracted_filter

    # 2. Check Intent / Decomposition
    sub_questions = decompose_query(clean_query)

    # 3. Retrieve candidate pools
    all_candidate_lists = []
    for sub_q in sub_questions:
        hits = hybrid_search(query=sub_q, top_k=candidate_pool_size, doc_filter=active_filter)
        all_candidate_lists.append(hits)

    # Merge candidates across sub-questions
    merged_candidates = all_candidate_lists[0] if all_candidate_lists else []
    for next_list in all_candidate_lists[1:]:
        merged_candidates = reciprocal_rank_fusion(merged_candidates, next_list)

    candidates = merged_candidates[:candidate_pool_size]

    # 4. Safety Fallback: If filtered search found too few results, search unfiltered
    if active_filter and len(candidates) < 2:
        print(f"Notice: Filtered search on '{active_filter}' returned few results. Falling back to unfiltered.")
        return smart_search(query=clean_query, top_k=top_k, doc_filter=None, candidate_pool_size=candidate_pool_size)

    if not candidates:
        return []

    # 5. Cross-Encoder re-ranks against the ORIGINAL question
    final_passages = rerank(
        query=query,
        passages=candidates,
        top_k=top_k
    )

    return final_passages


# Alias for drop-in compatibility
two_stage_search = smart_search