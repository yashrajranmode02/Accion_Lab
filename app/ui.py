"""
Gradio UI for FAQ RAG Chatbot.

This module provides a simple chat interface that delegates all logic to the router.
The UI is purely presentational - it displays responses from the router without
performing any LLM calls or decision-making.
"""

import gradio as gr
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    FAQ_SUMMARY_PATH,
    ROUTER_MODEL,
    GENERAL_RESPONSE_MODEL,
    CHAT_MODEL,
    OPENAI_API_KEY,
    FAISS_INDEX_PATH,
    CHUNKS_PKL_PATH,
    METADATA_PKL_PATH,
    EMBEDDING_MODEL,
    TOP_K_RETRIEVAL,
    MAX_HISTORY_TURNS
)
from llm.router import QueryRouter


# Initialize router (handles all LLM logic)
router = QueryRouter(
    summary_path=FAQ_SUMMARY_PATH,
    router_model=ROUTER_MODEL,
    general_model=GENERAL_RESPONSE_MODEL,
    chat_model=CHAT_MODEL,
    api_key=OPENAI_API_KEY,
    index_path=FAISS_INDEX_PATH,
    chunks_path=CHUNKS_PKL_PATH,
    metadata_path=METADATA_PKL_PATH,
    embedding_model=EMBEDDING_MODEL,
    top_k=TOP_K_RETRIEVAL,
    max_history_turns=MAX_HISTORY_TURNS
)


def chat_handler(message: str, history: list) -> str:
    """
    Handle incoming chat messages by delegating to the router.
    
    This function is purely presentational - it passes the message to the router
    and displays whatever response the router returns. No LLM calls or logic here.
    
    Args:
        message (str): User's message
        history (list): Chat history (not used in current implementation)
        
    Returns:
        str: Response message from router
        
    Libraries:
        None (just passes through to router)
    """
    # Delegate all logic to the router
    result = router.handle_query(message)
    
    # Simply return the response from the router
    return result['response']


def create_ui():
    """
    Create and configure the Gradio chat interface.
    
    Returns:
        gr.ChatInterface: Configured Gradio chat interface
        
    Libraries:
        gradio
    """
    interface = gr.ChatInterface(
        fn=chat_handler,
        title="ZenithDesk FAQ Chatbot",
        description="Ask questions about ZenithDesk using our knowledge base, or chat generally!",
        examples=[
            "How do I reset my password?",
            "What are the pricing plans?",
            "Hello!",
            "How do I integrate with Slack?"
        ],
        theme=gr.themes.Glass()
    )
    
    return interface


if __name__ == "__main__":
    """
    Launch the Gradio UI.
    """
    print("\n" + "="*50)
    print("Starting ZenithDesk FAQ Chatbot UI")
    print("="*50)
    print(f"Router Model: {ROUTER_MODEL}")
    print(f"General Response Model: {GENERAL_RESPONSE_MODEL}")
    print("="*50 + "\n")
    
    # Create and launch interface
    ui = create_ui()
    ui.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860
    )

