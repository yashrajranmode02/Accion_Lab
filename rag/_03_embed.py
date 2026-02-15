"""
Embedding Generation Module for RAG Chatbot.

This module handles generating embeddings using OpenAI's text-embedding-3-small model.
Supports batch processing for efficiency.
"""

from openai import OpenAI
import numpy as np
from typing import List
import time


def generate_embeddings(
    texts: List[str], 
    model: str, 
    api_key: str, 
    batch_size: int = 100
) -> np.ndarray:
    """
    Generate embeddings for a list of texts using OpenAI's embedding model.
    
    Processes texts in batches to handle rate limits and improve efficiency.
    Returns a numpy array of embeddings.
    
    Args:
        texts (List[str]): List of text strings to embed
        model (str): Name of the OpenAI embedding model (e.g., 'text-embedding-3-small')
        api_key (str): OpenAI API key
        batch_size (int): Number of texts to process in each batch (default: 100)
        
    Returns:
        np.ndarray: Array of embeddings with shape (len(texts), embedding_dim)
        
    Libraries:
        openai, numpy
    """
    client = OpenAI(api_key=api_key)
    all_embeddings = []
    
    print(f"Generating embeddings for {len(texts)} texts...")
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            # Call OpenAI API
            response = client.embeddings.create(
                input=batch,
                model=model
            )
            
            # Extract embeddings
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            print(f"✓ Processed batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")
            
            # Small delay to avoid rate limits
            if i + batch_size < len(texts):
                time.sleep(0.1)
                
        except Exception as e:
            print(f"✗ Error processing batch {i // batch_size + 1}: {e}")
            raise
    
    # Convert to numpy array
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    print(f"\n✓ Generated embeddings: shape {embeddings_array.shape}")
    
    return embeddings_array


def embed_query(query: str, model: str, api_key: str) -> np.ndarray:
    """
    Generate embedding for a single query string.
    
    This is a convenience function for embedding user queries at runtime.
    
    Args:
        query (str): The query text to embed
        model (str): Name of the OpenAI embedding model
        api_key (str): OpenAI API key
        
    Returns:
        np.ndarray: Embedding vector as numpy array with shape (embedding_dim,)
        
    Libraries:
        openai, numpy
    """
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.embeddings.create(
            input=[query],
            model=model
        )
        
        embedding = response.data[0].embedding
        return np.array(embedding, dtype=np.float32)
        
    except Exception as e:
        print(f"✗ Error embedding query: {e}")
        raise


if __name__ == "__main__":
    """
    Example usage of the embedding module.
    """
    from config import (
        RAW_DOCS_DIR, 
        SUPPORTED_FORMATS, 
        CHUNK_SIZE, 
        CHUNK_OVERLAP, 
        TIKTOKEN_ENCODING,
        EMBEDDING_MODEL,
        OPENAI_API_KEY,
        EMBEDDING_BATCH_SIZE
    )
    from rag._01_ingest import load_documents
    from rag._02_chunk import create_chunks
    
    # Load and chunk documents
    docs = load_documents(RAW_DOCS_DIR, SUPPORTED_FORMATS)
    chunks = create_chunks(docs, CHUNK_SIZE, CHUNK_OVERLAP, TIKTOKEN_ENCODING)
    
    # Extract chunk texts
    chunk_texts = [chunk['text'] for chunk in chunks]
    
    # Generate embeddings
    embeddings = generate_embeddings(
        chunk_texts, 
        EMBEDDING_MODEL, 
        OPENAI_API_KEY,
        EMBEDDING_BATCH_SIZE
    )
    
    print(f"\nEmbedding dimension: {embeddings.shape[1]}")
