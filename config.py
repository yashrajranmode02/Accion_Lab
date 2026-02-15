"""
Configuration file for FAQ RAG Chatbot.
Contains all configurable parameters for the application including OpenAI settings,
chunking parameters, FAISS settings, and file paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==================== OpenAI Configuration ====================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# OpenAI Models
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions
CHAT_MODEL = "gpt-4o-mini"  # For responses.create

# ==================== Chunking Parameters ====================
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 50  # tokens
TIKTOKEN_ENCODING = "cl100k_base"  # Compatible with OpenAI models

# ==================== FAISS Parameters ====================
TOP_K_RETRIEVAL = 5  # Number of top chunks to retrieve
FAISS_INDEX_TYPE = "FlatL2"  # Exact similarity search

# ==================== File Paths ====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
PROCESSED_DIR = DATA_DIR / "processed"

# Processed data files
CHUNKS_PKL_PATH = PROCESSED_DIR / "chunks.pkl"
METADATA_PKL_PATH = PROCESSED_DIR / "metadata.pkl"
FAISS_INDEX_PATH = PROCESSED_DIR / "faiss.index"

# ==================== Session Configuration ====================
MAX_HISTORY_TURNS = 5  # Number of conversation turns to keep in memory

# ==================== Supported File Formats ====================
SUPPORTED_FORMATS = [".md", ".txt", ".docx"]

# ==================== Embedding Configuration ====================
EMBEDDING_BATCH_SIZE = 100  # Process embeddings in batches for efficiency

# ==================== Router Configuration ====================
ROUTER_MODEL = "gpt-5-nano"  # Lightweight model for query classification
FAQ_SUMMARY_PATH = PROCESSED_DIR / "docsSummaryForRouter.md"
GENERAL_RESPONSE_MODEL = "gpt-5-nano"  # For out-of-scope queries
