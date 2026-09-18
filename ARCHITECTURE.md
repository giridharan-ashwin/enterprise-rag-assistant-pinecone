# Enterprise RAG Assistant — Architecture

## Baseline

```text
Document
  ↓
Chunking
  ↓
OpenAI embeddings (512 dimensions)
  ↓
Pinecone
  ↓
Semantic retrieval
  ↓
Context assembly
  ↓
OpenAI LLM
  ↓
Grounded answer + sources
```

## Why 512 dimensions?

The Pinecone index created for this project is configured for 512-dimensional
vectors. The OpenAI embedding request explicitly asks for the same 512-dimensional
representation so the vector dimensions match at both ends.

## Next phases

1. Hybrid retrieval
2. Reranking
3. Query rewriting
4. Evaluation
5. Access control
6. Observability
7. Agentic retrieval and tool use
