import os
import time
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from diskcache import Cache
from google import genai
from google.genai import types
from google.genai.errors import APIError


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in backend/.env")

client = genai.Client(api_key=api_key)

EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"


# Cache directory
CACHE_DIR = Path(__file__).resolve().parent.parent
cache = Cache(str(CACHE_DIR))

def retry_with_backoff(fn, max_retries=5, initial_delay=2.0):
    """Retries a call if rate-limited (429) or on temporary network errors."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            err_str = str(e).lower()
            if "429" in err_str or "resource_exhausted" in err_str:
                # Google rate limits reset over 1-minute windows; wait 20s
                sleep_time = max(delay, 20.0)
                print(f"Notice: Rate limit (429) hit. Waiting {sleep_time:.1f}s for quota window to reset...")
                time.sleep(sleep_time)
                delay *= 2
            else:
                print(f"Notice: API error ({e}). Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2


def embed(texts: list[str], mode: str = "document") -> np.ndarray:
    """
    Embeds texts into L2-normalized float32 vectors.
    Uses disk cache to avoid redundant API calls and save quota.
    """
    if not texts:
        return np.empty((0, 768), dtype=np.float32)

    task_type = "RETRIEVAL_QUERY" if mode == "query" else "RETRIEVAL_DOCUMENT"

    # Separate cached texts from uncached texts to minimize API calls
    cached_vectors = {}
    uncached_texts = []
    uncached_indices = []

    for idx, text in enumerate(texts):
        cache_key = f"embed:{EMBEDDING_MODEL}:{task_type}:{text}"
        if cache_key in cache:
            cached_vectors[idx] = cache[cache_key]
        else:
            uncached_texts.append(text)
            uncached_indices.append(idx)

    # If some texts are not in cache, call Google API
    if uncached_texts:
        def _call():
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=uncached_texts,
                config={"task_type": task_type},
            )
            return response.embeddings

        raw_embeddings = retry_with_backoff(_call)
        new_vecs = np.array([e.values for e in raw_embeddings], dtype=np.float32)

        # Normalize and store in cache
        norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        new_normalized = new_vecs / norms

        for idx, text, vec in zip(uncached_indices, uncached_texts, new_normalized):
            cache_key = f"embed:{EMBEDDING_MODEL}:{task_type}:{text}"
            cache[cache_key] = vec
            cached_vectors[idx] = vec

    # Reassemble in original order
    ordered_vectors = [cached_vectors[i] for i in range(len(texts))]
    return np.array(ordered_vectors, dtype=np.float32)


def generate(prompt: str) -> str:
    """Generates a text answer from Gemini, cached on disk by prompt text."""
    cache_key = f"gen:{CHAT_MODEL}:{prompt}"

    if cache_key in cache:
        return cache[cache_key]
    
    def _call():
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
        )
        return response.text or ""

    answer = retry_with_backoff(_call)
    cache[cache_key] = answer
    return answer

