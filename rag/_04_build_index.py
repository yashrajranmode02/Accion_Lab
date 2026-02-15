"""
FAISS Index Building Module for FAQ RAG Chatbot.

This module handles creating a FAISS vector index from embeddings and saving
the index along with chunk metadata to disk.
"""

import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict


def build_faiss_index(embeddings: np.ndarray, index_type: str = "FlatL2") -> faiss.Index:
    """
    Build a FAISS index from embeddings.
    
    Creates a FAISS index using the specified index type. Default is FlatL2
    for exact similarity search.
    
    Args:
        embeddings (np.ndarray): Array of embeddings with shape (n_vectors, embedding_dim)
        index_type (str): Type of FAISS index ('FlatL2' for exact search) (default: 'FlatL2')
        
    Returns:
        faiss.Index: FAISS index object containing the vectors
        
    Libraries:
        faiss, numpy
    """
    dimension = embeddings.shape[1]
    
    # Create index based on type
    if index_type == "FlatL2":
        index = faiss.IndexFlatL2(dimension)
    else:
        raise ValueError(f"Unsupported index type: {index_type}")
    
    # Add vectors to index
    index.add(embeddings)
    
    print(f"✓ Built FAISS index: {index.ntotal} vectors, {dimension} dimensions")
    return index


def save_index_and_metadata(
    index: faiss.Index,
    chunks: List[Dict],
    index_path: Path,
    chunks_path: Path,
    metadata_path: Path
) -> None:
    """
    Save FAISS index and chunk metadata to disk.
    
    Saves the FAISS index to a .index file and chunk data to .pkl files.
    The chunks are stored with text and metadata (no embeddings in PKL).
    
    Args:
        index (faiss.Index): FAISS index object
        chunks (List[Dict]): List of chunk dictionaries with metadata
        index_path (Path): Path to save FAISS index file
        chunks_path (Path): Path to save chunks.pkl file
        metadata_path (Path): Path to save metadata.pkl file
        
    Returns:
        None
        
    Libraries:
        faiss, pickle, pathlib
    """
    # Ensure parent directories exist
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, str(index_path))
    print(f"✓ Saved FAISS index: {index_path}")
    
    # Save chunks (text + metadata only, no embeddings)
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"✓ Saved chunks: {chunks_path} ({len(chunks)} chunks)")
    
    # Create metadata mapping: index → chunk_id
    metadata = {
        'chunk_id_mapping': [chunk['chunk_id'] for chunk in chunks],
        'total_chunks': len(chunks),
        'embedding_dimension': index.d
    }
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Saved metadata: {metadata_path}")


def build_and_save_index(
    chunks: List[Dict],
    embeddings: np.ndarray,
    index_path: Path,
    chunks_path: Path,
    metadata_path: Path,
    index_type: str = "FlatL2"
) -> None:
    """
    Build FAISS index and save all components to disk.
    
    This is the main function that orchestrates index building and saving.
    
    Args:
        chunks (List[Dict]): List of chunk dictionaries with metadata
        embeddings (np.ndarray): Array of embeddings
        index_path (Path): Path to save FAISS index file
        chunks_path (Path): Path to save chunks.pkl file
        metadata_path (Path): Path to save metadata.pkl file
        index_type (str): Type of FAISS index (default: 'FlatL2')
        
    Returns:
        None
        
    Libraries:
        faiss, pickle, numpy, pathlib
    """
    print("\n" + "="*50)
    print("Building FAISS Index")
    print("="*50)
    
    # Build index
    index = build_faiss_index(embeddings, index_type)
    
    # Save index and metadata
    save_index_and_metadata(
        index, 
        chunks, 
        index_path, 
        chunks_path, 
        metadata_path
    )
    
    print("\n✓ Index building complete!")
    print("="*50)


if __name__ == "__main__":
    """
    Example usage: Complete pipeline from ingestion to index building.
    """
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
        FAISS_INDEX_TYPE
    )
    from rag._01_ingest import load_documents
    from rag._02_chunk import create_chunks
    from rag._03_embed import generate_embeddings
    
    # Step 1: Load documents
    print("\n[1/4] Loading documents...")
    docs = load_documents(RAW_DOCS_DIR, SUPPORTED_FORMATS)
    
    # Step 2: Create chunks
    print("\n[2/4] Creating chunks...")
    chunks = create_chunks(docs, CHUNK_SIZE, CHUNK_OVERLAP, TIKTOKEN_ENCODING)
    
    # Step 3: Generate embeddings
    print("\n[3/4] Generating embeddings...")
    chunk_texts = [chunk['text'] for chunk in chunks]
    embeddings = generate_embeddings(
        chunk_texts,
        EMBEDDING_MODEL,
        OPENAI_API_KEY,
        EMBEDDING_BATCH_SIZE
    )
    
    # Step 4: Build and save index
    print("\n[4/4] Building and saving index...")
    build_and_save_index(
        chunks,
        embeddings,
        FAISS_INDEX_PATH,
        CHUNKS_PKL_PATH,
        METADATA_PKL_PATH,
        FAISS_INDEX_TYPE
    )
