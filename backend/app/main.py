from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.rag import add_document, ask, CHUNKS

app = FastAPI(title="DocPilot API", version="0.1.0")


# Request schema for /ask
class AskRequest(BaseModel):
    question: str
    top_k: int = 4


@app.on_event("startup")
def preload_sample_docs():
    """Automatically ingest any sample docs found on disk at startup."""
    docs_dir = Path("data/sample_docs")
    if docs_dir.exists():
        for file_path in docs_dir.glob("*.md"):
            try:
                text = file_path.read_text(encoding="utf-8")
                add_document(file_path.name, text)
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
        print(f"Preloaded sample docs! Total indexed chunks in memory: {len(CHUNKS)}")


@app.get("/health")
def health():
    """Sanity check endpoint."""
    return {
        "status": "ok",
        "total_chunks": len(CHUNKS),
    }


@app.post("/ingest")
async def ingest_files(files: list[UploadFile] = File(...)):
    """Upload and index new .txt or .md files into memory."""
    total_added = 0
    uploaded_names = []

    for file in files:
        content = await file.read()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")

        chunks_created = add_document(file.filename, text)
        total_added += chunks_created
        uploaded_names.append(file.filename)

    return {
        "message": f"Successfully indexed {len(files)} files.",
        "files": uploaded_names,
        "chunks_added": total_added,
        "total_chunks": len(CHUNKS),
    }


@app.post("/ask")
def ask_question(request: AskRequest):
    """Ask a question against all documents currently in memory."""
    result = ask(question=request.question, top_k=request.top_k)
    return result