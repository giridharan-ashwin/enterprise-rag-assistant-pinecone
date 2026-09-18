# Enterprise RAG Assistant

A production-oriented RAG starter project using **OpenAI + Pinecone + FastAPI**.

## Architecture

```text
User
  ↓
FastAPI
  ↓
RAG Service
  ├── Pinecone (vector retrieval)
  └── OpenAI (embeddings + generation)
  ↓
Answer + sources
```

## Current vector configuration

- Pinecone index: `enterprise-rag`
- Metric: cosine
- Dimension: 512
- Embedding model: `text-embedding-3-small` with `dimensions=512`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your OpenAI and Pinecone API keys to `.env`.

Then ingest the sample document:

```bash
python -m app.ingest data/sample/company_handbook.md
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Open Swagger:

`http://localhost:8000/docs`

## Query

```json
{
  "question": "What is the remote work policy?",
  "top_k": 5
}
```
