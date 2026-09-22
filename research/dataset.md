# Dataset and Evaluation Corpus

## Corpus

The evaluation corpus consists of synthetic enterprise-style Markdown documents covering operational and organizational knowledge domains.

The corpus contains 20 stress-test documents with approximately five sections per document, resulting in approximately 100 retrieval chunks.

## Evaluation Cases

The retrieval benchmark contains 150 cases.

### Known Queries

135 cases represent questions for which relevant evidence exists in the knowledge base.

### Unknown Queries

15 cases represent questions for which the knowledge base does not contain sufficient information.

These cases evaluate the system's ability to reject unsupported questions rather than generate plausible but unsupported answers.

### Multi-Section Queries

15 cases require evidence from multiple knowledge sections.

Each multi-section case contains two expected source-section targets.

These cases specifically evaluate whether retrieval can provide complete evidence coverage rather than only identifying one semantically similar chunk.

## Answer Evaluation Set

The answer-quality evaluation contains 60 cases:

- 30 known single-section cases
- 15 multi-section cases
- 15 unknown cases

The answer evaluation uses the retrieved evidence and evaluates the generated answer using an LLM-based judge.

## Evaluation Philosophy

The benchmark intentionally includes both answerable and unanswerable questions.

This is important because a RAG system should not be evaluated solely on whether it can answer known questions.

A production enterprise RAG system must also:

1. retrieve relevant evidence,
2. cover all required evidence for multi-part questions,
3. avoid unsupported answers,
4. provide traceable citations.