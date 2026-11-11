import hashlib
import json
import numpy as np
from pathlib import Path
from diskcache import Cache
from app.config import INDEX_DIR
from app.embeddings import get_embedder

CACHE_DIR = INDEX_DIR / "production_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

prod_cache = Cache(str(CACHE_DIR))
embedder = get_embedder("gemini")

SEMANTIC_CACHE_REGISTRY_KEY = "semantic_cache_registry"
SEMANTIC_SIMILARITY_THRESHOLD = 0.88


def build_cache_key(question: str, tenant: str = "public", role: str = "all", prompt_version: str = "PROMPT_V2") -> str:
    """Exact cryptographic hash key."""
    normalized_q = question.strip().lower()
    raw = f"{tenant}::{role}::{prompt_version}::{normalized_q}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_response(
    question: str, 
    tenant: str = "public", 
    role: str = "all", 
    prompt_version: str = "PROMPT_V2"
) -> tuple[dict | None, str | None]:
    """
    Checks cache in 2 tiers:
    1. Exact hash match (0.1ms).
    2. Semantic vector match (cosine >= 0.96).
    Returns (data, match_type) where match_type is 'exact', 'semantic', or None.
    """
    # 1. Exact Cache Check
    exact_key = build_cache_key(question, tenant, role, prompt_version)
    if exact_key in prod_cache:
        data = prod_cache[exact_key]
        return (json.loads(data) if isinstance(data, str) else data), "exact"

    # 2. Semantic Cache Check (within same tenant & role)
    registry = prod_cache.get(SEMANTIC_CACHE_REGISTRY_KEY, [])
    # Filter registry to matching tenant and role
    tenant_entries = [e for e in registry if e["tenant"] == tenant and e["role"] == role]

    if tenant_entries:
        q_vec = embedder.embed_query(question)
        cached_vecs = np.array([e["vector"] for e in tenant_entries], dtype=np.float32)
        scores = cached_vecs @ q_vec  # cosine similarity on normalized vectors

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= SEMANTIC_SIMILARITY_THRESHOLD:
            target_key = tenant_entries[best_idx]["key"]
            if target_key in prod_cache:
                data = prod_cache[target_key]
                res = json.loads(data) if isinstance(data, str) else data
                res["semantic_score"] = round(best_score, 4)
                return res, "semantic"

    return None, None


def set_cached_response(
    question: str, 
    response_data: dict, 
    tenant: str = "public", 
    role: str = "all", 
    prompt_version: str = "PROMPT_V2", 
    expire_sec: int = 3600
):
    """Stores response in exact cache and indexes its vector for semantic match."""
    exact_key = build_cache_key(question, tenant, role, prompt_version)
    prod_cache.set(exact_key, response_data, expire=expire_sec)

    # Register for semantic search
    try:
        q_vec = embedder.embed_query(question).tolist()
        registry = prod_cache.get(SEMANTIC_CACHE_REGISTRY_KEY, [])
        registry.append({
            "key": exact_key,
            "tenant": tenant,
            "role": role,
            "question": question,
            "vector": q_vec,
        })
        # Keep registry bounded to last 500 queries
        prod_cache[SEMANTIC_CACHE_REGISTRY_KEY] = registry[-500:]
    except Exception as e:
        print(f"Warning: Failed to update semantic cache: {e}")