"""
RAG Package - Retrieval-Augmented Generation components.
Contains modules for document ingestion, chunking, embedding, indexing, and retrieval.
"""

from rag._01_ingest import load_documents, Document
from rag._02_chunk import create_chunks
from rag._03_embed import generate_embeddings, embed_query
from rag._04_build_index import build_and_save_index
from rag._05_retrieve import Retriever, retrieve_top_k

__all__ = [
    'load_documents',
    'Document',
    'create_chunks',
    'generate_embeddings',
    'embed_query',
    'build_and_save_index',
    'Retriever',
    'retrieve_top_k'
]
