"""
Prometheus-style metrics for the FAQ RAG chatbot.

This module exposes a thin abstraction over `prometheus_client` so the rest
of the codebase can record metrics without depending directly on Prometheus.

If `prometheus_client` is not installed, all functions become no-ops so that
observability does not break the core application.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional

try:  # Optional dependency
    from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Counter = Histogram = Gauge = Summary = None  # type: ignore[misc,assignment]
    start_http_server = None  # type: ignore[assignment]


_METRICS_SERVER_STARTED = False
_METRICS_LOCK = threading.Lock()


def start_metrics_server(port: int = 8000) -> None:
    """
    Start the Prometheus metrics HTTP server on the given port.

    Safe to call multiple times; the server will only be started once.
    """
    global _METRICS_SERVER_STARTED
    if start_http_server is None:
        # Prometheus client not installed; nothing to start.
        return

    with _METRICS_LOCK:
        if _METRICS_SERVER_STARTED:
            return
        start_http_server(port)
        _METRICS_SERVER_STARTED = True


# In-memory snapshot used by the Observability UI.
_STATE: Dict[str, Any] = {
    "request_count": 0,
    "error_count": 0,
    "request_latency_ms": [],
    "retrieval_latency_ms": [],
    "generation_latency_ms": [],
    "token_usage": 0,
    "retrieval_relevance_score": None,
    "hallucination_score": None,
    "context_utilization_score": None,
    "answer_groundedness_score": None,
}


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


if Counter is not None and Histogram is not None and Gauge is not None and Summary is not None:  # pragma: no cover - metrics wiring
    REQUEST_COUNT = Counter(
        "rag_request_total", "Total number of RAG requests", ["endpoint"]
    )
    ERROR_COUNT = Counter(
        "rag_request_errors_total", "Total number of errored RAG requests", ["endpoint"]
    )
    REQUEST_LATENCY = Histogram(
        "rag_request_latency_seconds",
        "End-to-end RAG request latency in seconds",
        ["endpoint"],
    )
    RETRIEVAL_LATENCY = Histogram(
        "rag_retrieval_latency_seconds",
        "Document retrieval latency in seconds",
    )
    GENERATION_LATENCY = Histogram(
        "rag_generation_latency_seconds",
        "GPT generation latency in seconds",
    )
    TOKEN_USAGE = Counter(
        "rag_token_usage_total", "Total tokens used by GPT generations"
    )
    RETRIEVAL_RELEVANCE = Gauge(
        "retrieval_relevance_score",
        "Cosine similarity between query and retrieved documents",
        ["query_type", "retriever_model"],
    )
    HALLUCINATION_SCORE = Gauge(
        "hallucination_score",
        "Hallucination score (0 grounded, 1 likely hallucinated)",
        ["query_type", "retriever_model"],
    )
    CONTEXT_UTILIZATION = Gauge(
        "context_utilization_score",
        "Average similarity between answer and retrieved context",
        ["query_type", "retriever_model"],
    )
    ANSWER_GROUNDEDNESS = Gauge(
        "answer_groundedness_score",
        "Similarity between answer and combined retrieved context",
        ["query_type", "retriever_model"],
    )
else:  # pragma: no cover - optional dependency path
    REQUEST_COUNT = ERROR_COUNT = REQUEST_LATENCY = None  # type: ignore[assignment]
    RETRIEVAL_LATENCY = GENERATION_LATENCY = TOKEN_USAGE = None  # type: ignore[assignment]
    RETRIEVAL_RELEVANCE = HALLUCINATION_SCORE = CONTEXT_UTILIZATION = ANSWER_GROUNDEDNESS = None  # type: ignore[assignment]


def observe_request(endpoint: str, latency_ms: float, error: bool = False) -> None:
    """
    Record a completed RAG request.
    """
    _STATE["request_count"] += 1
    _STATE["request_latency_ms"].append(latency_ms)
    if error:
        _STATE["error_count"] += 1

    if REQUEST_COUNT is not None and REQUEST_LATENCY is not None and ERROR_COUNT is not None:
        REQUEST_COUNT.labels(endpoint=endpoint).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency_ms / 1000.0)
        if error:
            ERROR_COUNT.labels(endpoint=endpoint).inc()


def observe_retrieval(latency_ms: float) -> None:
    """
    Record retrieval latency.
    """
    _STATE["retrieval_latency_ms"].append(latency_ms)
    if RETRIEVAL_LATENCY is not None:
        RETRIEVAL_LATENCY.observe(latency_ms / 1000.0)


def observe_generation(latency_ms: float, token_usage: Optional[int] = None) -> None:
    """
    Record generation latency and optional token usage.
    """
    _STATE["generation_latency_ms"].append(latency_ms)
    if token_usage:
        _STATE["token_usage"] += int(token_usage)

    if GENERATION_LATENCY is not None:
        GENERATION_LATENCY.observe(latency_ms / 1000.0)
    if TOKEN_USAGE is not None and token_usage:
        TOKEN_USAGE.inc(token_usage)


def observe_rag_quality_metrics(
    query_type: str,
    retriever_model: str,
    retrieval_relevance_score: float,
    hallucination_score: float,
    context_utilization_score: float,
    answer_groundedness_score: float,
) -> None:
    """
    Record advanced RAG quality metrics.
    """
    _STATE["retrieval_relevance_score"] = retrieval_relevance_score
    _STATE["hallucination_score"] = hallucination_score
    _STATE["context_utilization_score"] = context_utilization_score
    _STATE["answer_groundedness_score"] = answer_groundedness_score

    if RETRIEVAL_RELEVANCE is not None:
        RETRIEVAL_RELEVANCE.labels(
            query_type=query_type, retriever_model=retriever_model
        ).set(retrieval_relevance_score)
    if HALLUCINATION_SCORE is not None:
        HALLUCINATION_SCORE.labels(
            query_type=query_type, retriever_model=retriever_model
        ).set(hallucination_score)
    if CONTEXT_UTILIZATION is not None:
        CONTEXT_UTILIZATION.labels(
            query_type=query_type, retriever_model=retriever_model
        ).set(context_utilization_score)
    if ANSWER_GROUNDEDNESS is not None:
        ANSWER_GROUNDEDNESS.labels(
            query_type=query_type, retriever_model=retriever_model
        ).set(answer_groundedness_score)


def get_current_metrics_snapshot() -> Dict[str, Any]:
    """
    Return a shallow snapshot of the metrics state for the Observability UI.
    """
    # Basic aggregation to keep the UI concise.
    def _agg(values):
        if not values:
            return {"count": 0, "avg_ms": None, "p95_ms": None}
        sorted_vals = sorted(values)
        count = len(sorted_vals)
        avg = sum(sorted_vals) / count
        p95 = sorted_vals[int(0.95 * (count - 1))]
        return {"count": count, "avg_ms": avg, "p95_ms": p95}

    return {
        "request_count": _STATE["request_count"],
        "error_count": _STATE["error_count"],
        "request_latency": _agg(_STATE["request_latency_ms"]),
        "retrieval_latency": _agg(_STATE["retrieval_latency_ms"]),
        "generation_latency": _agg(_STATE["generation_latency_ms"]),
        "token_usage": _STATE["token_usage"],
        "retrieval_relevance_score": _STATE["retrieval_relevance_score"],
        "hallucination_score": _STATE["hallucination_score"],
        "context_utilization_score": _STATE["context_utilization_score"],
        "answer_groundedness_score": _STATE["answer_groundedness_score"],
        "metrics_endpoint": f"http://localhost:{os.getenv('METRICS_PORT', '8000')}/metrics",
    }


class RequestTimer:
    """
    Helper context manager for timing and recording a full RAG request.
    """

    def __init__(self, endpoint: str = "chat"):
        self._endpoint = endpoint
        self._start_ms: Optional[float] = None
        self._error = False

    def __enter__(self):
        self._start_ms = _now_ms()
        return self

    def mark_error(self):
        self._error = True

    def __exit__(self, exc_type, exc, tb):
        if self._start_ms is None:
            return
        latency_ms = _now_ms() - self._start_ms
        if exc is not None:
            self._error = True
        observe_request(self._endpoint, latency_ms, error=self._error)

