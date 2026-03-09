"""
Gradio Observability Dashboard components for the FAQ RAG chatbot.

This module defines a reusable tab that can be added to the main UI to
visualize high-level metrics and link out to external tools like Grafana,
Phoenix, and LangSmith.
"""

from __future__ import annotations

import gradio as gr

from observability.metrics import get_current_metrics_snapshot
from observability.tracing import get_span_log_snapshot


def build_observability_tab():
    """
    Build the Observability tab with framework-specific views:
    - Prometheus Metrics
    - OpenTelemetry Traces
    """
    gr.Markdown("## Observability Dashboard")

    with gr.Tab("Prometheus Metrics"):
        gr.Markdown("**Prometheus Metrics Snapshot**")
        with gr.Row():
            rag_metrics = gr.JSON(label="RAG Pipeline Metrics")
            retrieval_metrics = gr.JSON(label="Retrieval Metrics")
        with gr.Row():
            generation_metrics = gr.JSON(label="Generation Metrics")
            system_metrics = gr.JSON(label="System Metrics & Endpoint")

        def _load_metrics():
            snapshot = get_current_metrics_snapshot()
            rag = {
                "request_count": snapshot["request_count"],
                "error_count": snapshot["error_count"],
                "request_latency": snapshot["request_latency"],
            }
            retrieval = {
                "retrieval_latency": snapshot["retrieval_latency"],
            }
            generation = {
                "generation_latency": snapshot["generation_latency"],
                "token_usage": snapshot["token_usage"],
            }
            system = {
                "metrics_endpoint": snapshot["metrics_endpoint"],
            }
            return rag, retrieval, generation, system

        refresh_button = gr.Button("Refresh Metrics")
        refresh_button.click(
            fn=_load_metrics,
            inputs=[],
            outputs=[rag_metrics, retrieval_metrics, generation_metrics, system_metrics],
        )

    with gr.Tab("OpenTelemetry Traces"):
        gr.Markdown("**Recent OpenTelemetry Spans (in-process log)**")
        spans_json = gr.JSON(label="Span Log")

        def _load_spans():
            return get_span_log_snapshot()

        spans_button = gr.Button("Refresh Spans")
        spans_button.click(fn=_load_spans, inputs=[], outputs=spans_json)

    # Grafana integration removed per user request.
