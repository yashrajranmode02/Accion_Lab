"""
Query Router Module for FAQ RAG Chatbot.

This module classifies user queries as in-scope (ZenithDesk-related) or out-of-scope
using a lightweight LLM (gpt-5-nano) and FAQ summary context.
For IN_SCOPE queries, it performs RAG retrieval and response generation.
"""

from openai import OpenAI
from pathlib import Path
from typing import Literal, List, Dict
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from rag._05_retrieve import Retriever
from llm.prompt import build_rag_prompt
from llm.responder import generate_response


def load_faq_summary(summary_path: Path) -> str:
    """
    Load the FAQ summary document for routing context.
    
    This summary provides the router with information about what topics
    are covered in the FAQ knowledge base.
    
    Args:
        summary_path (Path): Path to the FAQ summary markdown file
        
    Returns:
        str: Content of the FAQ summary
        
    Libraries:
        pathlib
    """
    if not summary_path.exists():
        raise FileNotFoundError(f"FAQ summary not found: {summary_path}")
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        return f.read()


def classify_query(
    query: str,
    faq_summary: str,
    model: str,
    api_key: str
) -> Literal["IN_SCOPE", "OUT_OF_SCOPE"]:
    """
    Classify a user query as in-scope or out-of-scope for the FAQ system.
    
    Uses a lightweight LLM to determine if the query is related to topics
    covered in the FAQ knowledge base (ZenithDesk features, billing, etc.).
    
    Args:
        query (str): User's question
        faq_summary (str): Summary of FAQ topics and coverage
        model (str): OpenAI model name (e.g., 'gpt-5-nano')
        api_key (str): OpenAI API key
        
    Returns:
        Literal["IN_SCOPE", "OUT_OF_SCOPE"]: Classification result
        
    Libraries:
        openai
    """
    client = OpenAI(api_key=api_key)
    
    # Build classification prompt
    system_prompt = """You are a query classifier for a ZenithDesk FAQ chatbot.
Your job is to determine if a user's question is related to ZenithDesk (the product described below) or not.

ZenithDesk FAQ Coverage:
{faq_summary}

Instructions:
- If the query is about ZenithDesk features, billing, setup, integrations, API, troubleshooting, or security, respond with: IN_SCOPE
- If the query is a greeting, small talk, or unrelated to ZenithDesk, respond with: OUT_OF_SCOPE
- Respond with ONLY "IN_SCOPE" or "OUT_OF_SCOPE" - no other text.""".format(faq_summary=faq_summary)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
        
        )
        
        classification = response.choices[0].message.content.strip().upper()
        
        # Validate response
        if classification in ["IN_SCOPE", "OUT_OF_SCOPE"]:
            return classification
        else:
            # Default to IN_SCOPE if unclear (safer to attempt retrieval)
            print(f"⚠️ Unexpected classification response: {classification}, defaulting to IN_SCOPE")
            return "IN_SCOPE"
            
    except Exception as e:
        print(f"✗ Error in query classification: {e}")
        # Default to IN_SCOPE on error
        return "IN_SCOPE"


class QueryRouter:
    """
    Query router class for classification and response generation.
    
    This class handles all LLM logic:
    - Classifies queries as IN_SCOPE or OUT_OF_SCOPE
    - Generates responses for OUT_OF_SCOPE queries
    - Returns structured responses for the UI to display
    
    Attributes:
        faq_summary (str): Loaded FAQ summary content
        router_model (str): Model for classification
        general_model (str): Model for general responses
        api_key (str): OpenAI API key
        
    Libraries:
        openai, pathlib
    """
    
    def __init__(
        self, 
        summary_path: Path, 
        router_model: str,
        general_model: str,
        chat_model: str,
        api_key: str,
        index_path: Path,
        chunks_path: Path,
        metadata_path: Path,
        embedding_model: str,
        top_k: int,
        max_history_turns: int
    ):
        """
        Initialize the QueryRouter with RAG capabilities.
        
        Args:
            summary_path (Path): Path to FAQ summary file
            router_model (str): OpenAI model for classification
            general_model (str): OpenAI model for general responses
            chat_model (str): OpenAI model for RAG responses
            api_key (str): OpenAI API key
            index_path (Path): Path to FAISS index file
            chunks_path (Path): Path to chunks pickle file
            metadata_path (Path): Path to metadata pickle file
            embedding_model (str): OpenAI embedding model name
            top_k (int): Number of chunks to retrieve
            max_history_turns (int): Maximum conversation turns to keep
            
        Returns:
            None
            
        Libraries:
            pathlib, openai, faiss (via Retriever)
        """
        # Router settings
        self.faq_summary = load_faq_summary(summary_path)
        self.router_model = router_model
        self.general_model = general_model
        self.chat_model = chat_model
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        
        # RAG settings
        self.top_k = top_k
        self.max_history_turns = max_history_turns
        self.embedding_model = embedding_model
        
        # Initialize retriever
        try:
            self.retriever = Retriever(
                index_path=index_path,
                chunks_path=chunks_path,
                metadata_path=metadata_path,
                embedding_model=embedding_model,
                api_key=api_key,
                top_k=top_k
            )
            print(f"✓ Retriever loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load retriever: {e}")
            print(f"  IN_SCOPE queries will use fallback message until index is built.")
            self.retriever = None
        
        # Conversation memory (sliding window)
        self.history: List[Dict] = []
        
        print(f"✓ Router initialized")
        print(f"  - Classification model: {router_model}")
        print(f"  - General response model: {general_model}")
        print(f"  - RAG chat model: {chat_model}")
        print(f"  - Max history turns: {max_history_turns}")
    
    def classify(self, query: str) -> Literal["IN_SCOPE", "OUT_OF_SCOPE"]:
        """
        Classify a user query.
        
        Args:
            query (str): User's question
            
        Returns:
            Literal["IN_SCOPE", "OUT_OF_SCOPE"]: Classification result
            
        Libraries:
            openai
        """
        return classify_query(query, self.faq_summary, self.router_model, self.api_key)
    
    def generate_general_response(self, query: str) -> str:
        """
        Generate a general response for OUT_OF_SCOPE queries.
        
        Uses the general response model to provide helpful answers to
        queries that are not related to the FAQ knowledge base.
        
        Args:
            query (str): User's question
            
        Returns:
            str: Generated response
            
        Libraries:
            openai
        """
        try:
            response = self.client.chat.completions.create(
                model=self.general_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that assists with answering FAQs related to a company named ZenithDesk, you can exchange pleasantries but decline answering questions out of the scope of your purpose by saying 'I apologize, but I cannot answer that question.'"
                    },
                    {"role": "user", "content": query}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def _update_history(self, user_query: str, assistant_response: str):
        """
        Update conversation history with sliding window.
        
        Adds the latest user query and assistant response to history,
        and maintains a sliding window of the last N turns.
        
        Args:
            user_query (str): User's question
            assistant_response (str): Assistant's response
            
        Returns:
            None
            
        Libraries:
            None
        """
        # Add user message
        self.history.append({"role": "user", "content": user_query})
        
        # Add assistant message
        self.history.append({"role": "assistant", "content": assistant_response})
        
        # Maintain sliding window (keep last N turns = 2N messages)
        max_messages = self.max_history_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]
    
    def handle_query(self, query: str) -> dict:
        """
        Handle a user query with classification and response generation.
        
        This is the main entry point for the UI. It classifies the query
        and generates appropriate responses:
        - IN_SCOPE: Retrieves context from FAISS and generates RAG response
        - OUT_OF_SCOPE: Generates general LLM response
        
        Args:
            query (str): User's question
            
        Returns:
            dict: Response dictionary with keys:
                - classification (str): "IN_SCOPE" or "OUT_OF_SCOPE"
                - response (str): The text to display to the user
                
        Libraries:
            openai, faiss (via Retriever)
        """
        # Classify the query
        classification = self.classify(query)
        
        if classification == "IN_SCOPE":
            # FAQ-related query - perform RAG
            if self.retriever is None:
                # Fallback if retriever not loaded
                response_text = "Main RAG functionality is being worked on, please try later."
            else:
                try:
                    # Retrieve relevant chunks
                    retrieved_chunks = self.retriever.retrieve(query, top_k=self.top_k)
                    
                    # Build prompt with context and history
                    messages = build_rag_prompt(query, retrieved_chunks, self.history)
                    
                    # Generate response
                    response_text = generate_response(
                        messages,
                        self.chat_model,
                        self.api_key
                    )
                    
                    # Update conversation memory
                    self._update_history(query, response_text)
                    
                except Exception as e:
                    print(f"✗ Error in RAG pipeline: {e}")
                    response_text = "I apologize, but I encountered an error processing your question."
            
            response = {
                "classification": "IN_SCOPE",
                "response": response_text
            }
        else:
            # General query - generate response (no memory for out-of-scope)
            general_response = self.generate_general_response(query)
            response = {
                "classification": "OUT_OF_SCOPE",
                "response": general_response
            }
        
        return response


if __name__ == "__main__":
    """
    Example usage of the router module.
    """
    from config import (
        FAQ_SUMMARY_PATH, ROUTER_MODEL, GENERAL_RESPONSE_MODEL, CHAT_MODEL,
        OPENAI_API_KEY, FAISS_INDEX_PATH, CHUNKS_PKL_PATH, METADATA_PKL_PATH,
        EMBEDDING_MODEL, TOP_K_RETRIEVAL, MAX_HISTORY_TURNS
    )
    
    # Initialize router with full RAG capabilities
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
    
    # Test queries
    test_queries = [
        "How do I reset my password?",
        "What are the pricing plans?",
        "Hello!",
        "What's the weather today?",
        "How do I integrate with Slack?"
    ]
    
    print("\n" + "="*50)
    print("Router Query Handling Tests")
    print("="*50)
    
    for query in test_queries:
        result = router.handle_query(query)
        print(f"\nQuery: '{query}'")
        print(f"Classification: {result['classification']}")
        print(f"Response: {result['response'][:100]}...")
