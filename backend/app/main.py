from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.config import DEFAULT_TOP_K
from app.loader import load_text_from_bytes
from app.store import add_document, delete_document, list_documents, search, CHUNKS
from app.llm import generate
from app.query.router import route_query
from app.generation.verifier import generate_verified_answer
from fastapi.responses import StreamingResponse
from app.llm import generate, generate_stream
from app.agent.graph import run_agent

from app.security.auth import create_access_token, verify_access_token, UserContext
from app.security.pii import mask_pii
from app.security.guardrails import detect_prompt_injection

app = FastAPI(title="DocPilot API", version="0.1.0")


# Request schema for /ask
class AskRequest(BaseModel):
    question: str
    top_k: int = 4
    doc_filter: str | None = None  # <--- Added filter option
    chat_history: list[dict] = []  # <--- Forward conversation history

class LoginRequest(BaseModel):
    username: str
    tenant: str = "engineering"
    role: str = "intern"  # "admin", "engineer", or "intern"


@app.post("/login")
def login(request: LoginRequest):
    """Issues a signed, tamper-proof JWT token containing tenant and role permissions."""
    token = create_access_token(
        user_id=request.username,
        tenant=request.tenant,
        role=request.role
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "tenant": request.tenant,
        "role": request.role
    }

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
            raw_text = load_text_from_bytes(file.filename, content)
            # Mask PII and leaked credentials before chunking & embedding!
            clean_text = mask_pii(raw_text)
            chunks_created = add_document(file.filename, clean_text)
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
def ask_question(request: AskRequest):
    """Answers questions with routing, automatic filter extraction, and verified citations."""

    # 0. Check prompt injection defense
    is_attack, attack_reason = detect_prompt_injection(request.question)
    if is_attack:
        return {
            "answerable": False,
            "answer": "Security Alert: Prompt injection attempt detected. Request blocked.",
            "citations": [],
            "sources": [],
            "fast_refusal": True
        }
    
    # 1. Check intent route
    route = route_query(request.question)
    
    # Fast path: If it's just a greeting or general pleasantry, skip search completely!
    if route == "direct_chat":
        chat_prompt = f"You are DocPilot, a helpful technical documentation assistant. Respond pleasantly and briefly to the user's message: '{request.question}'"
        return {
            "answerable": True,
            "answer": generate(chat_prompt).strip(),
            "citations": [],
            "citation_validity": 1.0,
            "sources": [],
            "fast_refusal": False,
        }

    # 2. Technical question: Execute LangGraph Self-Correcting Agent
    result = run_agent(
        question=request.question,
        chat_history=request.chat_history
    )
    return result


@app.post("/ask/stream")
def ask_question_stream(request: AskRequest):
    """Streams the verified answer token-by-token using Server-Sent Events (SSE)."""
    route = route_query(request.question)
    
    if route == "direct_chat":
        chat_prompt = f"You are DocPilot, a helpful technical documentation assistant. Respond pleasantly and briefly to the user's message: '{request.question}'"
        def chat_generator():
            for token in generate_stream(chat_prompt):
                yield token
        return StreamingResponse(chat_generator(), media_type="text/event-stream")

    # 1. Retrieve passages
    passages = search(
        query=request.question, 
        top_k=request.top_k, 
        doc_filter=request.doc_filter
    )

    if not passages:
        def empty_generator():
            yield "I don't have enough information in the provided documents."
        return StreamingResponse(empty_generator(), media_type="text/event-stream")

    # 2. Build context for streaming prompt
    context = "\n\n".join([f"[{p['number']}] (Source: {p['doc']})\n{p['text']}" for p in passages])
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

    def token_generator():
        # Stream text tokens from Gemini
        for token in generate_stream(prompt):
            yield token

    return StreamingResponse(token_generator(), media_type="text/event-stream")


