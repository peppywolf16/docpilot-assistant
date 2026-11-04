import numpy as np
from sentence_transformers import CrossEncoder
from app.config import DEFAULT_TOP_K

# Lightweight, ultra-fast cross-encoder model running on CPU
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

print(f"Loading local Cross-Encoder re-ranker: {RERANKER_MODEL_NAME}...")
reranker = CrossEncoder(RERANKER_MODEL_NAME)


def deduplicate_passages(passages: list[dict], similarity_threshold: float = 0.85) -> list[dict]:
    """Removes duplicate or near-identical passages based on text overlap."""
    unique_passages = []
    
    for p in passages:
        text_words = set(p["text"].lower().split())
        is_duplicate = False
        
        for kept in unique_passages:
            kept_words = set(kept["text"].lower().split())
            if not text_words or not kept_words:
                continue
            # Jaccard similarity
            intersection = len(text_words.intersection(kept_words))
            union = len(text_words.union(kept_words))
            similarity = intersection / union if union > 0 else 0
            
            if similarity >= similarity_threshold:
                is_duplicate = True
                break
                
        if not is_duplicate:
            unique_passages.append(p)
            
    return unique_passages



def reorder_lost_in_the_middle(passages: list[dict]) -> list[dict]:
    """
    Mitigates the 'Lost in the Middle' effect:
    LLMs pay most attention to the very beginning and very end of the prompt context.
    This reorders the top passages so the #1 best passage is first,
    the #2 best passage is last, and lower-ranked passages sit in the middle.
    
    Example order for 4 items: [Rank 1, Rank 3, Rank 4, Rank 2]
    """
    if len(passages) <= 2:
        return passages

    reordered = [None] * len(passages)
    left = 0
    right = len(passages) - 1

    for i, p in enumerate(passages):
        if i % 2 == 0:
            reordered[left] = p
            left += 1
        else:
            reordered[right] = p
            right -= 1

    return reordered


def rerank(query: str, passages: list[dict], top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Takes candidate passages (e.g. top 20 from Hybrid Search),
    scores them using a deep Cross-Encoder, sorts descending,
    and applies Lost-in-the-Middle reordering to the top-k.
    """
    if not passages:
        return []

    clean_passage = deduplicate_passages(passages)
    # 1. Prepare (query, text) pairs for the cross-encoder
    pairs = [[query, p["text"]] for p in passages]

    # 2. Compute cross-encoder relevance scores
    scores = reranker.predict(pairs)

    # 3. Attach re-ranker score and sort
    scored_passages = []
    for p, score in zip(passages, scores):
        p_copy = dict(p)
        p_copy["rerank_score"] = round(float(score), 4)
        scored_passages.append(p_copy)

    # Sort descending by neural cross-encoder score
    scored_passages.sort(key=lambda x: x["rerank_score"], reverse=True)

    # 4. Take top_k and re-assign rank numbers
    top_candidates = scored_passages[:top_k]
    for rank, p in enumerate(top_candidates, start=1):
        p["number"] = rank

    # 5. Apply Lost-in-the-Middle reordering for optimal LLM context attention
    final_passages = reorder_lost_in_the_middle(top_candidates)

    return final_passages



