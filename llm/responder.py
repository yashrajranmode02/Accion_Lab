"""
Response Generation Module for FAQ RAG Chatbot.

This module handles calling OpenAI's chat completions API to generate
responses based on prompts constructed with retrieved context.
"""

from openai import OpenAI
from typing import List, Dict


def generate_response(
    messages: List[Dict],
    model: str,
    api_key: str,
    temperature: float = 0.3,
    max_tokens: int = 500
) -> str:
    """
    Generate a response using OpenAI's chat completions API.
    
    Calls OpenAI with the provided messages array (which includes
    system instructions, context, history, and query) and returns
    the generated response text.
    
    Args:
        messages (List[Dict]): Complete messages array for OpenAI
        model (str): OpenAI model name (e.g., 'gpt-4o-mini')
        api_key (str): OpenAI API key
        temperature (float): Sampling temperature (default: 0.3 for focused responses)
        max_tokens (int): Maximum tokens in response (default: 500)
        
    Returns:
        str: Generated response text
        
    Libraries:
        openai
    """
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        error_message = f"I apologize, but I encountered an error generating a response: {str(e)}"
        print(f"✗ Error in generate_response: {e}")
        return error_message


if __name__ == "__main__":
    """
    Example usage of the response generation module.
    """
    from config import CHAT_MODEL, OPENAI_API_KEY
    
    # Sample messages
    sample_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer based on the provided context."
        },
        {
            "role": "user",
            "content": "Context: ZenithDesk offers three pricing plans: Starter, Pro, and Enterprise.\n\nQuestion: What pricing plans are available?"
        }
    ]
    
    print("\n" + "="*50)
    print("Testing Response Generation")
    print("="*50)
    
    response = generate_response(
        sample_messages,
        CHAT_MODEL,
        OPENAI_API_KEY
    )
    
    print(f"\nGenerated Response:\n{response}")
