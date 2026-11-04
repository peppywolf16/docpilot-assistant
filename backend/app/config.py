import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"

# Embedding Model Selection
EMBEDDER_TYPE = os.getenv("EMBEDDER_TYPE", "gemini")  # "gemini" or "bge-small"
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "0")) or None  # None for full, or e.g. 768, 512

# Model-specific index folder (e.g. data/index/gemini_full/ or data/index/bge-small_full/)
index_folder_name = f"{EMBEDDER_TYPE}_{EMBEDDING_DIM if EMBEDDING_DIM else 'full'}"
INDEX_DIR = DATA_DIR / "index" / index_folder_name

# Ensure directories exist
INDEX_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY and EMBEDDER_TYPE == "gemini":
    raise ValueError("GEMINI_API_KEY is missing from backend/.env")

# Models
CHAT_MODEL = "gemini-3.5-flash-lite"

# RAG Hyperparameters
CHUNKER_STRATEGY = os.getenv("CHUNKER_STRATEGY", "markdown")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 4

# Query Transformation Strategy: "none", "multi_query", "hyde", "step_back"
QUERY_TRANSFORM_STRATEGY = os.getenv("QUERY_TRANSFORM_STRATEGY", "none")