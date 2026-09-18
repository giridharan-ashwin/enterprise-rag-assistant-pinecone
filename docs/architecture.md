# Architecture Notes

## Phase 1 — Baseline RAG

1. Ingestion
2. Chunking
3. Embedding
4. Vector storage
5. Top-k retrieval
6. Prompt construction
7. Grounded generation
8. Source return

## Phase 2 — Retrieval Quality

- Query rewriting
- Hybrid lexical + semantic retrieval
- Reranking
- Metadata filters
- Parent-child chunking
- Retrieval evaluation

## Phase 3 — Production Readiness

- Authentication and authorization
- Tenant isolation
- Document-level access control
- PII handling
- Prompt-injection defenses
- Observability
- Cost tracking
- Latency budgets
- Evaluation datasets

## Phase 4 — Agentic Extension

```text
Agent
 ├── Retrieve Knowledge
 ├── Search Structured Data
 ├── Call Enterprise API
 ├── Ask for Human Approval
 └── Produce Auditable Result
```
