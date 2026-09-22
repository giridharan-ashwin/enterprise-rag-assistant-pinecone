# Methodology

## System Architecture

The system implements an enterprise Retrieval-Augmented Generation architecture:

User Question
    ↓
Query Embedding
    ↓
Dense Retrieval + Sparse Retrieval
    ↓
Reciprocal Rank Fusion
    ↓
Evidence Selection
    ↓
Context Assembly
    ↓
Grounded LLM Generation
    ↓
Answer + Citations + Abstention

## Dense Retrieval

Dense retrieval uses:

- OpenAI `text-embedding-3-small`
- 512-dimensional embeddings
- Pinecone
- cosine similarity

The system retrieves the highest-ranked candidate chunks from the enterprise knowledge corpus.

## Sparse Retrieval

Sparse retrieval uses BM25.

The lexical representation includes:

- document name
- section heading
- document text

This allows exact enterprise terminology to contribute to retrieval.

## Hybrid Retrieval

Dense and sparse results are combined using Reciprocal Rank Fusion (RRF).

For a document with rank `r`:

    RRF(r) = 1 / (k + r)

where:

    k = 60

The final ranking is based on the combined RRF score.

## Evidence Selection

The production retrieval pipeline does not apply the global dense similarity threshold before hybrid fusion.

Instead, dense and sparse retrieval results are first combined.

A result is retained when:

1. its dense similarity meets the configured threshold, or
2. it receives independent support from both dense and sparse retrieval.

This design was introduced after observing that applying a global dense threshold before fusion could remove legitimate evidence from multi-section questions.

## Generation

The generation layer uses an OpenAI chat model with structured JSON output.

The model receives:

- the user question
- retrieved source
- retrieved section
- chunk index
- retrieved text

The generation prompt requires:

- grounded answers
- no invented facts
- source citations
- explicit abstention when evidence is insufficient
- JSON output

## Citation Model

Each citation identifies:

- source
- section
- chunk index

This allows answer claims to be traced back to retrieved enterprise knowledge.

## Abstention

The system can abstain when:

- retrieval produces no usable context, or
- the generation model determines that the supplied evidence is insufficient.

Unknown questions are included explicitly in the benchmark to evaluate this behavior.

## Evaluation Dimensions

### Retrieval

- Known-query Top-1 accuracy
- Recall@3
- Unknown-query rejection
- Known-query rejection
- Multi-section recall

### Answer Quality

- Correctness
- Faithfulness
- Citation accuracy
- Abstention behavior

### System Performance

The production API additionally records:

- retrieval latency
- generation latency
- total latency
- context count
- prompt tokens
- completion tokens
- total tokens
- estimated generation cost