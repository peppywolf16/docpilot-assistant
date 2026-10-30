import os
import time
import numpy as np
from dotenv import load_dotenv
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

def retry_with_backoff(fn, max_retries=3, initial_delay=2.0):
    """Retries a function if it hits rate limits (429) or temporary server errors (503)."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return fn()
        except APIError as e:
            if attempt == max_retries - 1:
                raise e
            # Retry on rate limit (429) or service unavailable (503)
            if e.code in [429, 503]:
                time.sleep(delay)
                delay *= 2
            else:
                raise e
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay)
            delay *= 2

def embed(texts: list[str], mode: str = "document") -> np.ndarray:
    """
    Embeds texts into L2-normalized float32 vectors.
    mode: 'document' (indexing) or 'query' (search).
    """
    if not texts:
        return np.empty((0, 768), dtype=np.float32)

    # In the new google-genai SDK, task_type is a simple string
    task_type = "RETRIEVAL_QUERY" if mode == "query" else "RETRIEVAL_DOCUMENT"

    def _call():
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config={"task_type": task_type},
        )
        return response.embeddings

    raw_embeddings = retry_with_backoff(_call)

    # Convert to 2D numpy matrix
    vectors = np.array([e.values for e in raw_embeddings], dtype=np.float32)

    # L2-normalize vectors so cosine similarity = dot product (A @ B)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return vectors / norms


def generate(prompt: str) -> str:
    """Generates a text answer from Gemini 1.5 Flash."""
    def _call():
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
        )
        return response.text or ""

    return retry_with_backoff(_call)

