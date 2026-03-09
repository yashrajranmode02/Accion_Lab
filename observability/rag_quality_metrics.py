"""
RAG quality metrics computation for the FAQ chatbot.

This module computes semantic quality scores for each RAG response and
pushes them into Prometheus-compatible metrics via `observability.metrics`.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from rag._03_embed import embed_query
from observability.metrics import observe_rag_quality_metrics
from observability.tracing import span

try:  # Optional dependency
    from observability.phoenix_logger import log_rag_event  # type: ignore
except Exception:  # pragma: no cover
    log_rag_event = None  # type: ignore[assignment]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_rag_quality_scores(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    answer: str,
    embedding_model: str,
    api_key: str,
    query_type: str,
    retriever_model: str,
) -> Dict[str, float]:
    """
    Compute RAG quality scores based on embeddings.
    """
    with span("rag.post_processing", {"query_text": query}):
        # Build text lists
        doc_texts = [c.get("text", "") for c in retrieved_chunks if c.get("text")]
        if not doc_texts:
            scores = {
                "retrieval_relevance_score": 0.0,
                "hallucination_score": 1.0,
                "context_utilization_score": 0.0,
                "answer_groundedness_score": 0.0,
            }
            observe_rag_quality_metrics(query_type, retriever_model, **scores)
            return scores

        # Embed query, documents, and answer
        query_emb = embed_query(query, embedding_model, api_key)
        doc_embs = np.vstack([embed_query(t, embedding_model, api_key) for t in doc_texts])
        answer_emb = embed_query(answer, embedding_model, api_key)

        # Retrieval relevance: avg cosine(query, doc_i)
        rel_scores = [_cosine_similarity(query_emb, d) for d in doc_embs]
        retrieval_relevance = float(np.mean(rel_scores))

        # Context utilization: avg cosine(answer, doc_i)
        util_scores = [_cosine_similarity(answer_emb, d) for d in doc_embs]
        context_utilization = float(np.mean(util_scores))

        # Answer groundedness: cosine(answer, mean(doc_i))
        mean_doc = np.mean(doc_embs, axis=0)
        answer_groundedness = _cosine_similarity(answer_emb, mean_doc)

        # Hallucination: 1 - groundedness, clamped to [0,1]
        hallucination_score = float(max(0.0, min(1.0, 1.0 - answer_groundedness)))

        scores = {
            "retrieval_relevance_score": retrieval_relevance,
            "hallucination_score": hallucination_score,
            "context_utilization_score": context_utilization,
            "answer_groundedness_score": answer_groundedness,
        }

        # Record in Prometheus-style metrics
        observe_rag_quality_metrics(query_type, retriever_model, **scores)

        # Optionally log to Phoenix for richer RAG visualization
        if log_rag_event is not None:
            try:
                log_rag_event(
                    query=query,
                    query_embedding=query_emb.tolist(),
                    retrieved_chunks=retrieved_chunks,
                    document_embeddings=doc_embs.tolist(),
                    answer=answer,
                    scores=scores,
                    embedding_model=embedding_model,
                )
            except Exception:
                # Observability must not break main flow
                pass

        return scores

