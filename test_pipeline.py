"""
End-to-end Test Script for FAQ RAG Chatbot Pipeline.

This script runs the entire RAG pipeline:
1. Ingestion
2. Chunking
3. Embedding
4. FAISS Index Building
5. Retrieval Test
"""

import sys
import os
from pathlib import Path

# Add the current directory to sys.path to allow imports from rag and config
sys.path.append(str(Path(__file__).parent))

from config import (
    RAW_DOCS_DIR,
    SUPPORTED_FORMATS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TIKTOKEN_ENCODING,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    EMBEDDING_BATCH_SIZE,
    FAISS_INDEX_PATH,
    CHUNKS_PKL_PATH,
    METADATA_PKL_PATH,
    FAISS_INDEX_TYPE,
    TOP_K_RETRIEVAL
)

from rag._01_ingest import load_documents
from rag._02_chunk import create_chunks
from rag._03_embed import generate_embeddings
from rag._04_build_index import build_and_save_index
from rag._05_retrieve import Retriever


def run_test_pipeline():
    """
    Runs the complete RAG pipeline and performs a sample retrieval.
    """
    print("\n🚀 Starting RAG Pipeline Test...")
    print("=" * 50)

    # Step 1: Ingestion
    print("\n[Step 1/5] Ingesting documents...")
    docs = load_documents(RAW_DOCS_DIR, SUPPORTED_FORMATS)
    if not docs:
        print("❌ No documents found. Please add files to data/raw_docs/")
        return

    # Step 2: Chunking
    print("\n[Step 2/5] Creating chunks...")
    chunks = create_chunks(docs, CHUNK_SIZE, CHUNK_OVERLAP, TIKTOKEN_ENCODING)
    print(f"✓ Created {len(chunks)} chunks")

    # Step 3: Embedding
    print("\n[Step 3/5] Generating embeddings (OpenAI)...")
    chunk_texts = [chunk['text'] for chunk in chunks]
    embeddings = generate_embeddings(
        chunk_texts,
        EMBEDDING_MODEL,
        OPENAI_API_KEY,
        EMBEDDING_BATCH_SIZE
    )

    # Step 4: Build Index
    print("\n[Step 4/5] Building and saving FAISS index...")
    build_and_save_index(
        chunks,
        embeddings,
        FAISS_INDEX_PATH,
        CHUNKS_PKL_PATH,
        METADATA_PKL_PATH,
        FAISS_INDEX_TYPE
    )

    # Step 5: Retrieval Test
    print("\n[Step 5/5] Testing retrieval...")
    retriever = Retriever(
        FAISS_INDEX_PATH,
        CHUNKS_PKL_PATH,
        METADATA_PKL_PATH,
        EMBEDDING_MODEL,
        OPENAI_API_KEY,
        TOP_K_RETRIEVAL
    )

    test_query = "What are the pricing plans?"
    print(f"\nQuery: '{test_query}'")
    results = retriever.retrieve(test_query)

    print("\nRetrieval Results:")
    for i, res in enumerate(results, 1):
        print(f"{i}. [Source: {res['source_doc']}] (Dist: {res['distance']:.4f})")
        print(f"   Text: {res['text'][:150]}...")

    print("\n✅ Pipeline Test Completed Successfully!")
    print("=" * 50)


if __name__ == "__main__":
    try:
        run_test_pipeline()
    except Exception as e:
        print(f"\n❌ Pipeline Test Failed: {e}")
        import traceback
        traceback.print_exc()
