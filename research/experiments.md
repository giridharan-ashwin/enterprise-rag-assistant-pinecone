# Experimental Design

## Experiment 1 — Dense Retrieval Baseline

The first configuration uses dense vector retrieval alone.

Purpose:

- establish semantic retrieval performance
- measure Top-1 accuracy
- measure Recall@3
- measure unknown-query behavior
- identify semantic ranking failures

## Experiment 2 — Dense Retrieval with Similarity Threshold

A similarity threshold is applied to dense retrieval.

Purpose:

- evaluate whether low-similarity results can be rejected
- measure improvement in unsupported-query rejection
- measure the corresponding impact on known-query retrieval

The threshold experiments evaluated multiple threshold values.

Observed behavior showed that increasing the threshold improved unsupported-query rejection but eventually reduced retrieval recall.

## Experiment 3 — Hybrid Retrieval

Dense retrieval is combined with BM25 sparse retrieval using Reciprocal Rank Fusion.

Purpose:

- determine whether lexical evidence complements semantic retrieval
- recover exact terminology that may not receive the highest dense similarity
- improve evidence coverage

## Experiment 4 — Hybrid Retrieval with Evidence Selection

The production configuration performs:

Dense retrieval
+
BM25 retrieval
↓
RRF
↓
Evidence selection

The evidence-selection strategy was introduced after analyzing a multi-section vendor-management query.

The query:

"What happens when a vendor is onboarded and later offboarded?"

produced dense similarity scores below the global 0.50 threshold for both:

- Vendor Onboarding
- Vendor Offboarding

Applying the threshold before hybrid fusion therefore rejected relevant evidence.

BM25 independently surfaced the relevant terminology, allowing hybrid retrieval to recover the evidence.

## Experiment 5 — Answer Quality

The answer evaluation measures:

- correctness
- faithfulness
- citation accuracy
- abstention behavior

The evaluation set contains 60 cases.

## Experimental Principle

Retrieval and generation are evaluated separately.

This distinction is important because a correct retrieval system can still produce an inadequate answer, while a strong language model can produce a plausible answer despite incomplete evidence.

The evaluation therefore treats retrieval quality and answer quality as separate experimental dimensions.

## Reproducibility

All benchmark cases, evaluation utilities, retrieval implementations, and generated evaluation artifacts are maintained in the project repository.

The intended experiment workflow is:

1. Build or load the evaluation corpus.
2. Execute retrieval configurations.
3. Record retrieval results.
4. Evaluate retrieval metrics.
5. Generate answers using retrieved contexts.
6. Judge answer quality.
7. Analyze failure cases.