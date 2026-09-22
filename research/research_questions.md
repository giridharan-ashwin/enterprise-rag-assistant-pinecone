# Research Question

## Working Title

Evaluating Dense, Hybrid, and Threshold-Based Retrieval Strategies for Enterprise RAG Systems

## Primary Research Question

How do dense similarity thresholds and hybrid dense-sparse retrieval affect retrieval accuracy, unsupported-query rejection, and multi-section question answering in enterprise Retrieval-Augmented Generation (RAG) systems?

## Secondary Questions

1. How accurately does dense vector retrieval identify the correct enterprise knowledge section?
2. Does similarity-threshold filtering improve rejection of unsupported questions?
3. What retrieval failures are introduced by aggressive similarity thresholds?
4. Can sparse lexical retrieval recover relevant evidence missed by dense retrieval?
5. Does Reciprocal Rank Fusion (RRF) improve evidence coverage for multi-section questions?
6. How does retrieval quality affect downstream answer correctness, faithfulness, and citation accuracy?
7. How frequently does the generation layer abstain when sufficient evidence is unavailable?

## Hypothesis

A dense similarity threshold can improve unsupported-query rejection by filtering semantically weak matches, but an overly restrictive threshold can reject legitimate evidence, particularly when a question requires multiple enterprise knowledge sections.

Hybrid dense-sparse retrieval is expected to improve evidence coverage by combining semantic similarity with lexical matching.

## Research Scope

The study focuses on enterprise-style RAG workloads involving:

- policy questions
- operational questions
- security questions
- infrastructure questions
- procurement questions
- data-retention questions
- multi-section questions
- unsupported questions

The study evaluates retrieval and answer-generation behavior rather than attempting to establish a universally optimal retrieval configuration.

## Primary Contributions

The project provides:

1. A reproducible enterprise-style RAG evaluation corpus.
2. A controlled comparison of dense and hybrid retrieval.
3. An analysis of similarity-threshold behavior.
4. A multi-section retrieval evaluation.
5. An answer-quality evaluation using correctness, faithfulness, citation accuracy, and abstention.
6. Failure analysis connecting retrieval behavior to downstream answer quality.
7. An open-source implementation and reproducible evaluation framework.