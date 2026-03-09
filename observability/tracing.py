"""
OpenTelemetry tracing setup for the FAQ RAG chatbot.

This module centralizes tracer initialization so the rest of the codebase
can simply import `tracer` and create spans:

    from observability.tracing import tracer, span

    with span("bert_retrieval", query=user_query):
        ...
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


_TRACER: Optional[trace.Tracer] = None
_SPAN_LOG: List[Dict[str, Any]] = []


def init_tracer(service_name: str = "faq-rag-chatbot") -> trace.Tracer:
    """
    Initialize and return a global tracer.

    Safe to call multiple times; the same tracer instance is reused.
    """
    global _TRACER
    if _TRACER is not None:
        return _TRACER

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    # Export to console for local debugging.
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _TRACER = trace.get_tracer(service_name)
    return _TRACER


# Eagerly initialize a default tracer for convenience.
tracer: trace.Tracer = init_tracer()


@contextmanager
def span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Convenience context manager for creating spans with optional attributes.
    """
    import time

    start = time.perf_counter()
    with tracer.start_as_current_span(name) as current_span:
        if attributes:
            for key, value in attributes.items():
                current_span.set_attribute(key, value)
        try:
            yield current_span
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            _SPAN_LOG.append(
                {
                    "name": name,
                    "duration_ms": duration_ms,
                    "attributes": attributes or {},
                }
            )
            # Keep only the most recent 100 spans
            if len(_SPAN_LOG) > 100:
                del _SPAN_LOG[0 : len(_SPAN_LOG) - 100]


def get_span_log_snapshot() -> List[Dict[str, Any]]:
    """
    Return a shallow copy of the recent span log for UI display.
    """
    return list(_SPAN_LOG)

