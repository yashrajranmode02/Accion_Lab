"""
Retrieval Module for FAQ RAG Chatbot.

This module handles loading the FAISS index and retrieving top-k most relevant
chunks for a given query.
"""

import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from rag._03_embed import embed_query


def load_index_and_metadata(
    index_path: Path,
    chunks_path: Path,
    metadata_path: Path
) -> Tuple[faiss.Index, List[Dict], Dict]:
    """
    Load FAISS index, chunks, and metadata from disk.
    
    Args:
        index_path (Path): Path to FAISS index file
        chunks_path (Path): Path to chunks.pkl file
        metadata_path (Path): Path to metadata.pkl file
        
    Returns:
        Tuple[faiss.Index, List[Dict], Dict]: FAISS index, chunks list, and metadata dict
        
    Libraries:
        faiss, pickle, pathlib
    """
    # Load FAISS index
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    index = faiss.read_index(str(index_path))
    
    # Load chunks
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    
    # Load metadata
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"✓ Loaded index: {index.ntotal} vectors")
    print(f"✓ Loaded chunks: {len(chunks)} chunks")
    
    return index, chunks, metadata


def retrieve_top_k(
    query: str,
    index: faiss.Index,
    chunks: List[Dict],
    embedding_model: str,
    api_key: str,
    top_k: int = 5
) -> List[Dict]:
    """
    Retrieve top-k most relevant chunks for a query.
    
    Embeds the query, searches the FAISS index, and returns the most relevant chunks
    with their text and metadata.
    
    Args:
        query (str): User query string
        index (faiss.Index): FAISS index object
        chunks (List[Dict]): List of chunk dictionaries
        embedding_model (str): Name of the OpenAI embedding model
        api_key (str): OpenAI API key
        top_k (int): Number of top results to retrieve (default: 5)
        
    Returns:
        List[Dict]: List of top-k chunk dictionaries with text and metadata
        
    Libraries:
        faiss, numpy, openai (via embed_query)
    """
    # Embed the query
    query_embedding = embed_query(query, embedding_model, api_key)
    
    # Reshape for FAISS (needs 2D array)
    query_embedding = query_embedding.reshape(1, -1)
    
    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve corresponding chunks
    retrieved_chunks = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(chunks):  # Safety check
            chunk = chunks[idx].copy()
            chunk['distance'] = float(distance)
            retrieved_chunks.append(chunk)
    
    return retrieved_chunks


class Retriever:
    """
    Retriever class for loading index once and performing multiple retrievals.
    
    This class maintains loaded index and chunks for efficient repeated queries.
    
    Attributes:
        index (faiss.Index): Loaded FAISS index
        chunks (List[Dict]): Loaded chunks with metadata
        metadata (Dict): Loaded metadata
        embedding_model (str): OpenAI embedding model name
        api_key (str): OpenAI API key
        top_k (int): Number of results to retrieve
        
    Libraries:
        faiss, pickle, numpy, openai
    """
    
    def __init__(
        self,
        index_path: Path,
        chunks_path: Path,
        metadata_path: Path,
        embedding_model: str,
        api_key: str,
        top_k: int = 5
    ):
        """
        Initialize the Retriever with paths and configuration.
        
        Args:
            index_path (Path): Path to FAISS index file
            chunks_path (Path): Path to chunks.pkl file
            metadata_path (Path): Path to metadata.pkl file
            embedding_model (str): Name of the OpenAI embedding model
            api_key (str): OpenAI API key
            top_k (int): Number of top results to retrieve (default: 5)
            
        Returns:
            None
            
        Libraries:
            faiss, pickle, pathlib, openai
        """
        self.index, self.chunks, self.metadata = load_index_and_metadata(
            index_path, chunks_path, metadata_path
        )
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.top_k = top_k
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve top-k chunks for a query.
        
        Args:
            query (str): User query string
            top_k (int): Number of results to retrieve (default: use instance top_k)
            
        Returns:
            List[Dict]: List of top-k chunk dictionaries
            
        Libraries:
            faiss, numpy, openai
        """
        k = top_k if top_k is not None else self.top_k
        return retrieve_top_k(
            query,
            self.index,
            self.chunks,
            self.embedding_model,
            self.api_key,
            k
        )


if __name__ == "__main__":
    """
    Example usage of the retrieval module.
    """
    from config import (
        FAISS_INDEX_PATH,
        CHUNKS_PKL_PATH,
        METADATA_PKL_PATH,
        EMBEDDING_MODEL,
        OPENAI_API_KEY,
        TOP_K_RETRIEVAL
    )
    
    # Initialize retriever
    retriever = Retriever(
        FAISS_INDEX_PATH,
        CHUNKS_PKL_PATH,
        METADATA_PKL_PATH,
        EMBEDDING_MODEL,
        OPENAI_API_KEY,
        TOP_K_RETRIEVAL
    )
    
    # Test query
    test_query = "How do I reset my password?"
    print(f"\nQuery: {test_query}")
    print("="*50)
    
    results = retriever.retrieve(test_query)
    
    for i, chunk in enumerate(results, 1):
        print(f"\n[Result {i}]")
        print(f"Source: {chunk['source_doc']}")
        print(f"Distance: {chunk['distance']:.4f}")
        print(f"Text: {chunk['text'][:200]}...")
