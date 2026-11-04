from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.config import DEFAULT_TOP_K
from app.loader import load_text_from_bytes
from app.store import add_document, delete_document, list_documents, search, CHUNKS
from app.llm import generate
from app.query.router import route_query

app = FastAPI(title="DocPilot API", version="0.1.0")


# Request schema for /ask
class AskRequest(BaseModel):
    question: str
    top_k: int = 4
    doc_filter: str | None = None  # <--- Added filter option


@app.get("/health")
def health():
    """Return backend status and total indexed chunks/docs"""
    return {
        "status": "ok",
        "total_chunks": len(CHUNKS),
        "total_documents": len(list_documents()),
    }


@app.get("/documents")
def get_documents():
    """Lists all document names currently in the persistent index."""
    return {"documents": list_documents()}

@app.delete("/documents/{name}")
def remove_document(name: str):
    """Deletes a document and all its chunks/vectors from the index."""
    success = delete_document(name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document '{name}' not found.")
    return {"message": f"Document '{name}' successfully deleted.", "documents": list_documents()}


@app.post("/ingest")
async def ingest_files(files: list[UploadFile] = File(...)):
    """Uploads and indexes .md, .txt, or .pdf files."""
    total_added = 0
    uploaded_names = []

    for file in files:
        content = await file.read()
        try:
            text = load_text_from_bytes(file.filename, content)
            chunks_created = add_document(file.filename, text)
            total_added += chunks_created
            uploaded_names.append(file.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process {file.filename}: {e}")

    return {
        "message": f"Successfully processed {len(files)} file(s).",
        "files": uploaded_names,
        "chunks_added": total_added,
        "total_chunks": len(CHUNKS),
    }


@app.post("/ask")
@app.post("/ask")
def ask_question(request: AskRequest):
    """Answers questions with routing, automatic filter extraction, and grounded citations."""
    
    # 1. Check intent route
    route = route_query(request.question)
    
    # Fast path: If it's just a greeting or general pleasantry, skip search completely!
    if route == "direct_chat":
        chat_prompt = f"You are DocPilot, a helpful technical documentation assistant. Respond pleasantly and briefly to the user's message: '{request.question}'"
        return {
            "answer": generate(chat_prompt).strip(),
            "sources": []
        }

    # 2. Technical question: Run smart two-stage search (decomposition + filter extraction)
    passages = search(
        query=request.question, 
        top_k=request.top_k, 
        doc_filter=request.doc_filter
    )

    if not passages:
        return {
            "answer": "I don't have enough information in the provided documents.",
            "sources": [],
        }

    context_blocks = [
        f"[{p['number']}] (Source: {p['doc']}, Chunk: {p['chunk_id']})\n{p['text']}"
        for p in passages
    ]
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful technical documentation assistant. Answer the user's question using ONLY the provided passages below.

Rules:
1. Cite the passages you use with numbers in brackets, like [1], [2].
2. If the answer cannot be found in the provided passages, reply EXACTLY with:
"I don't have enough information in the provided documents."
Do not make anything up or use outside knowledge.

Passages:
{context}

Question: {request.question}
Answer:"""

    answer = generate(prompt)

    return {
        "answer": answer.strip(),
        "sources": passages,
    }
