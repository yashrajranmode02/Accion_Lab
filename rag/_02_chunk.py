"""
Text Chunking Module for FAQ RAG Chatbot.

This module handles splitting documents into fixed-size token chunks with overlap.
Uses tiktoken for accurate token counting compatible with OpenAI models.
"""

import tiktoken
import uuid
from typing import List, Dict
from rag._01_ingest import Document


def count_tokens(text: str, encoding_name: str) -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text (str): The text to count tokens for
        encoding_name (str): Name of the tiktoken encoding (e.g., 'cl100k_base')
        
    Returns:
        int: Number of tokens in the text
        
    Libraries:
        tiktoken
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def chunk_text(
    text: str, 
    chunk_size: int, 
    overlap: int, 
    encoding_name: str
) -> List[str]:
    """
    Split text into chunks of approximately chunk_size tokens with overlap.
    
    This function uses character-based approximation for efficiency while ensuring
    chunks don't exceed the token limit.
    
    Args:
        text (str): The text to chunk
        chunk_size (int): Target size of each chunk in tokens
        overlap (int): Number of overlapping tokens between chunks
        encoding_name (str): Name of the tiktoken encoding
        
    Returns:
        List[str]: List of text chunks
        
    Libraries:
        tiktoken
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        # Get chunk tokens
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        
        # Decode back to text
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Move start position (accounting for overlap)
        start += (chunk_size - overlap)
    
    return chunks


def create_chunks(
    documents: List[Document], 
    chunk_size: int, 
    overlap: int, 
    encoding_name: str
) -> List[Dict]:
    """
    Create chunks from documents with metadata.
    
    Processes each document into fixed-size chunks and attaches metadata including
    chunk_id, text, and source_doc.
    
    Args:
        documents (List[Document]): List of Document objects to chunk
        chunk_size (int): Target size of each chunk in tokens
        overlap (int): Number of overlapping tokens between chunks
        encoding_name (str): Name of the tiktoken encoding
        
    Returns:
        List[Dict]: List of chunk dictionaries with keys: chunk_id, text, source_doc
        
    Libraries:
        tiktoken, uuid
    """
    all_chunks = []
    
    for doc in documents:
        # Split document into chunks
        text_chunks = chunk_text(doc.text, chunk_size, overlap, encoding_name)
        
        # Create chunk objects with metadata
        for idx, segment_text in enumerate(text_chunks):
            chunk = {
                'chunk_id': str(uuid.uuid4()),
                'text': segment_text,
                'source_doc': doc.source,
                'chunk_index': idx
            }
            all_chunks.append(chunk)
        
        print(f"✓ Chunked: {doc.source} → {len(text_chunks)} chunks")
    
    print(f"\nTotal chunks created: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    """
    Example usage of the chunking module.
    """
    from config import RAW_DOCS_DIR, SUPPORTED_FORMATS, CHUNK_SIZE, CHUNK_OVERLAP, TIKTOKEN_ENCODING
    from rag.ingest import load_documents
    
    # Load documents
    docs = load_documents(RAW_DOCS_DIR, SUPPORTED_FORMATS)
    
    # Create chunks
    chunks = create_chunks(docs, CHUNK_SIZE, CHUNK_OVERLAP, TIKTOKEN_ENCODING)
    
    # Display sample chunks
    print("\n--- Sample Chunk ---")
    if chunks:
        sample = chunks[0]
        print(f"ID: {sample['chunk_id']}")
        print(f"Source: {sample['source_doc']}")
        print(f"Text preview: {sample['text'][:200]}...")
