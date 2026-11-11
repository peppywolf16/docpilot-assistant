import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

load_dotenv(backend_dir / ".env")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from backend/.env")

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# Configure Settings to use models matching your account
# In Google GenAI v1beta, model names don't always need the "models/" prefix or use 2.5-flash
Settings.llm = Gemini(model="models/gemini-3.5-flash-lite", api_key=api_key)
Settings.embed_model = GeminiEmbedding(model_name="models/gemini-embedding-001", api_key=api_key)
Settings.chunk_size = 800
Settings.chunk_overlap = 100

DOCS_DIR = backend_dir / "data" / "sample_docs"


def build_llamaindex_pipeline():
    """Loads docs, builds a VectorStoreIndex, and creates a query engine."""
    print("Loading documents using LlamaIndex SimpleDirectoryReader...")
    reader = SimpleDirectoryReader(input_dir=str(DOCS_DIR), required_exts=[".md"])
    documents = reader.load_data()
    print(f"Loaded {len(documents)} document files.")

    print("Building in-memory VectorStoreIndex with Gemini embeddings...")
    index = VectorStoreIndex.from_documents(documents)
    
    # Configure retriever with top_k = 4
    query_engine = index.as_query_engine(similarity_top_k=4)
    return query_engine, index


if __name__ == "__main__":
    t0 = time.time()
    query_engine, _ = build_llamaindex_pipeline()
    print(f"LlamaIndex pipeline built in {time.time() - t0:.2f}s!")

    test_query = "What is the command to run the FastAPI development server?"
    print(f"\nTesting Query: '{test_query}'")
    response = query_engine.query(test_query)
    print("\n--- LlamaIndex Response ---")
    print(str(response).strip())
    print("\nRetrieved Source Nodes:")
    for node in response.source_nodes:
        print(f"  - Score: {node.score:.4f} | File: {Path(node.metadata.get('file_name', '')).name}")