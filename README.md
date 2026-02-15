# ZenithDesk FAQ Chatbot - Quick Start Guide

## Overview
This is an MVP FAQ chatbot for ZenithDesk with RAG (Retrieval-Augmented Generation) capabilities and intelligent query routing.

## Features
- **Query Router**: Classifies queries as in-scope (ZenithDesk-related) or out-of-scope
- **RAG Pipeline**: Document ingestion, chunking, embedding, and FAISS indexing
- **Gradio UI**: Simple chat interface with fallback responses

## Setup

### 1. Prerequisites
- Python 3.11+
- OpenAI API key

### 2. Installation
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_api_key_here
```

### 4. Build RAG Index (One-time)
```bash
# Add your FAQ documents to data/raw_docs/
# Then run the pipeline
python test_pipeline.py
```

This will create:
- `data/processed/faiss.index` - Vector index
- `data/processed/chunks.pkl` - Text chunks
- `data/processed/metadata.pkl` - Metadata

### 5. Launch UI
```bash
python app/ui.py
```

Access at: `http://127.0.0.1:7860`

## Project Structure
```
faq-rag-chatbot/
├── app/
│   ├── ui.py              # Gradio interface
│   └── __init__.py
├── rag/
│   ├── _01_ingest.py      # Document loading
│   ├── _02_chunk.py       # Text chunking
│   ├── _03_embed.py       # OpenAI embeddings
│   ├── _04_build_index.py # FAISS indexing
│   ├── _05_retrieve.py    # Top-k retrieval
│   └── __init__.py
├── llm/
│   ├── router.py          # Query classification
│   └── __init__.py
├── data/
│   ├── raw_docs/          # Input FAQ documents
│   └── processed/         # Generated index files
├── config.py              # Configuration
├── test_pipeline.py       # End-to-end test
└── requirements.txt
```

## How It Works

### Query Routing
1. User submits a question
2. Router classifies using `gpt-5-nano` and FAQ summary
3. **IN_SCOPE**: Shows fallback message (RAG coming soon)
4. **OUT_OF_SCOPE**: Generates general response using `gpt-4o-mini`

### RAG Pipeline (Offline)
1. **Ingest**: Load `.md`, `.txt`, `.docx` files
2. **Chunk**: Split into 500-token chunks (50 overlap)
3. **Embed**: Generate OpenAI embeddings
4. **Index**: Build FAISS vector index
5. **Store**: Save to `data/processed/`

## Testing

### Test Router
```bash
python llm/router.py
```

### Test RAG Pipeline
```bash
python test_pipeline.py
```

## Configuration

Key settings in `config.py`:
- `ROUTER_MODEL`: "gpt-5-nano"
- `EMBEDDING_MODEL`: "text-embedding-3-small"
- `CHUNK_SIZE`: 500 tokens
- `CHUNK_OVERLAP`: 50 tokens
- `TOP_K_RETRIEVAL`: 5 chunks

## Current Status

✅ **Completed**:
- RAG storage pipeline
- Query router
- Basic Gradio UI
- Fallback responses
- Full RAG retrieval integration
- Prompt templates
- Session memory

## Troubleshooting

### "OPENAI_API_KEY not set"
Add your API key to `.env` file

### "FAQ summary not found"
Ensure `data/processed/docsSummaryForRouter.md` exists

### "No documents found"
Add FAQ documents to `data/raw_docs/` and run `test_pipeline.py`
