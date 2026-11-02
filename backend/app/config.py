import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"

# Ensure directories exist
INDEX_DIR.mkdir(parents=True,exist_ok=True)
SAMPLE_DOCS_DIR.mkdir(parents=True,exist_ok=True)

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from backend/.env")

# Models
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"

# RAG Hyperparameters
CHUNKER_STRATEGY = os.getenv("CHUNKER_STRATEGY", "markdown")  # "markdown" or "recursive"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 4