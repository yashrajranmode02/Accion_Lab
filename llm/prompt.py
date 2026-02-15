"""
Prompt Building Module for FAQ RAG Chatbot.

This module handles constructing prompts with retrieved context and conversation history
for the RAG-based question answering system.
"""

from typing import List, Dict


def format_context(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into readable context for the prompt.
    
    Combines multiple document chunks with source references into a
    structured context string that the LLM can use to answer questions.
    
    Args:
        chunks (List[Dict]): Retrieved chunks with 'text' and 'source_doc' keys
        
    Returns:
        str: Formatted context string
        
    Libraries:
        None
    """
    if not chunks:
        return "No relevant context found."
    
    context_parts = []
    for idx, chunk in enumerate(chunks, 1):
        text = chunk.get('text', '').strip()
        source = chunk.get('source_doc', 'Unknown')
        context_parts.append(f"[Context {idx}] (Source: {source})\n{text}")
    
    return "\n\n".join(context_parts)


def format_history(history: List[Dict]) -> List[Dict]:
    """
    Convert conversation history to OpenAI message format.
    
    Takes the sliding window history from the router and formats it
    for inclusion in the OpenAI messages array.
    
    Args:
        history (List[Dict]): Conversation history in format:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        
    Returns:
        List[Dict]: Same history (already in correct format)
        
    Libraries:
        None
    """
    # History is already in the correct format from router
    return history


def build_rag_prompt(query: str, chunks: List[Dict], history: List[Dict]) -> List[Dict]:
    """
    Build complete prompt with system instruction, context, history, and query.
    
    Constructs the full messages array for OpenAI chat completions API,
    including:
    - System instructions (answer from context only)
    - Retrieved context from FAISS
    - Conversation history (if any)
    - Current user query
    
    Args:
        query (str): User's current question
        chunks (List[Dict]): Retrieved document chunks from FAISS
        history (List[Dict]): Conversation history (last N turns)
        
    Returns:
        List[Dict]: Complete messages array ready for OpenAI API
        
    Libraries:
        None
    """
    # System instruction
    system_message = {
        "role": "system",
        "content": """You are a helpful ZenithDesk support assistant. Follow these rules strictly:

1. Answer questions ONLY using the provided context below.
2. If the context does not contain enough information to answer the question, respond with: "I don't have enough information to answer that question based on the available documentation."
3. Be concise, clear, and helpful in your responses.
4. When referencing information, you may mention which context chunk it came from.
5. If the user asks follow-up questions, use the conversation history to maintain context."""
    }
    
    # Format retrieved context
    formatted_context = format_context(chunks)
    
    # Build context message (injected as a user message with context)
    context_message = {
        "role": "user",
        "content": f"""Here is the relevant context from the ZenithDesk documentation:

{formatted_context}

Please use this context to answer my question."""
    }
    
    # Build messages array
    messages = [system_message]
    
    # Add context
    messages.append(context_message)
    
    # Add conversation history (if any)
    if history:
        messages.extend(format_history(history))
    
    # Add current query
    current_query = {
        "role": "user",
        "content": query
    }
    messages.append(current_query)
    
    return messages


if __name__ == "__main__":
    """
    Example usage of the prompt building module.
    """
    # Sample chunks
    sample_chunks = [
        {
            "text": "To reset your password, click 'Forgot Password' on the login page. You'll receive an email with a reset link.",
            "source_doc": "account_management.md"
        },
        {
            "text": "Password reset links expire after 24 hours for security. You can request a new link if needed.",
            "source_doc": "security_faq.md"
        }
    ]
    
    # Sample history
    sample_history = [
        {"role": "user", "content": "How do I access my account?"},
        {"role": "assistant", "content": "You can access your account by logging in at zenithdesk.com/login"}
    ]
    
    # Build prompt
    query = "I forgot my password, what should I do?"
    messages = build_rag_prompt(query, sample_chunks, sample_history)
    
    print("\n" + "="*50)
    print("Sample Prompt Structure")
    print("="*50)
    for msg in messages:
        print(f"\n[{msg['role'].upper()}]")
        print(msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content'])
