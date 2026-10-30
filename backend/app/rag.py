import numpy as np
from app.llm import embed, generate

# In-memory stores
CHUNKS: list[dict] = []     # List of {"doc_name": str, "chunk_id": int, "text": str}
VECTORS: np.ndarray | None = None  # 2D NumPy array of shape (N, dimensions)

# Make the chunks
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Splits text into chunks of roughly chunk_size characters with overlap."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start += chunk_size - overlap
    return [c for c in chunks if c]

#  returns the how many chunks were made
def add_document(name: str, text: str) -> int:
    """Chunks, embeds, and stores a document in memory. Returns number of chunks created."""
    global CHUNKS, VECTORS

    new_text_chunks = chunk_text(text)
    if not new_text_chunks:
        return 0

    # 1. Embed the new chunks as documents
    new_vectors = embed(new_text_chunks, mode="document")

    # 2. Store metadata & text
    start_idx = len(CHUNKS)
    for i, chunk in enumerate(new_text_chunks):
        CHUNKS.append({
            "doc_name": name,
            "chunk_id": start_idx + i + 1,
            "text": chunk,
        })

    # 3. Append to NumPy matrix
    if VECTORS is None:
        VECTORS = new_vectors
    else:
        VECTORS = np.vstack([VECTORS, new_vectors])

    return len(new_text_chunks)

# Returns the answer to the question with proper citations
def ask(question: str, top_k: int = 4) -> dict:
    """Searches memory for top_k relevant chunks and generates a grounded, cited answer."""
    global CHUNKS, VECTORS

    if VECTORS is None or len(CHUNKS) == 0:
        return {
            "answer": "I don't have enough information in the provided documents.",
            "sources": [],
        }

    # 1. Embed the query
    q_vec = embed([question], mode="query")[0]  # shape (dim,)

    # 2. Cosine similarity = Dot product (because vectors are L2-normalized)
    scores = VECTORS @ q_vec  # shape (N,)

    # 3. Get top_k indices sorted by score descending
    actual_k = min(top_k, len(CHUNKS))
    top_indices = np.argsort(scores)[::-1][:actual_k]

    retrieved_passages = []
    passages_text_block = []

    for rank, idx in enumerate(top_indices, start=1):
        chunk_data = CHUNKS[idx]
        score = float(scores[idx])

        retrieved_passages.append({
            "number": rank,
            "doc": chunk_data["doc_name"],
            "chunk_id": chunk_data["chunk_id"],
            "score": round(score, 4),
            "text": chunk_data["text"],
        })

        passages_text_block.append(
            f"[{rank}] (Source: {chunk_data['doc_name']})\n{chunk_data['text']}"
        )

    context = "\n\n".join(passages_text_block)

    # 4. Strictly grounded prompt with citation instructions
    prompt = f"""You are a helpful technical assistant. Answer the user's question using ONLY the provided passages below.

Rules:
1. Cite the passages you use with numbers in brackets, like [1], [2].
2. If the answer cannot be found in the provided passages, reply EXACTLY with:
"I don't have enough information in the provided documents."
Do not make anything up or use outside knowledge.

Passages:
{context}

Question: {question}
Answer:"""

    answer = generate(prompt)

    return {
        "answer": answer.strip(),
        "sources": retrieved_passages,
    }