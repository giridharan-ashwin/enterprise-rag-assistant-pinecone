from __future__ import annotations

from typing import Any

from app.clients import pinecone_index
from app.config import settings
from app.services.embeddings import embed_query
from app.services.hybrid import (
    bm25_retriever,
    reciprocal_rank_fusion,
)


def dense_retrieve(
    question: str,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve documents from Pinecone using dense vector similarity.

    No reranking is performed.
    """

    vector = embed_query(question)

    limit = top_k or settings.top_k

    response = pinecone_index.query(
        vector=vector,
        top_k=limit,
        include_metadata=True,
        namespace=settings.pinecone_namespace,
    )

    results: list[dict[str, Any]] = []

    for match in response["matches"]:
        metadata = match.get("metadata", {})

        score = float(match.get("score", 0.0))

        results.append(
            {
                "id": str(match.get("id", "")),
                "source": metadata.get("source", "unknown"),
                "section": metadata.get("section", "unknown"),
                "chunk_index": metadata.get("chunk_index", -1),
                "score": score,
                "dense_score": score,
                "text": metadata.get("text", ""),
            }
        )

    return results


def retrieve(
    question: str,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Production hybrid retrieval.

    Pipeline:

        Dense retrieval
             +
        BM25 retrieval
             ↓
        Reciprocal Rank Fusion
             ↓
        Evidence selection
             ↓
        Final contexts

    The Pinecone reranker is intentionally not used.

    Important:
    Dense similarity thresholding is NOT performed before RRF.
    This allows BM25 to recover lexically strong evidence that may
    have a lower dense similarity score, which is especially useful
    for multi-section questions.
    """

    final_top_k = top_k or settings.top_k

    candidate_k = max(final_top_k * 2, 10)

    # ---------------------------------------------------------
    # 1. Dense retrieval
    # ---------------------------------------------------------

    dense_results = dense_retrieve(
        question,
        top_k=candidate_k,
    )

    # ---------------------------------------------------------
    # 2. Sparse BM25 retrieval
    # ---------------------------------------------------------

    sparse_results = bm25_retriever.retrieve(
        question,
        top_k=candidate_k,
    )

    # ---------------------------------------------------------
    # 3. Reciprocal Rank Fusion
    # ---------------------------------------------------------

    fused_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        top_k=candidate_k,
    )

    if not fused_results:
        return []

    # ---------------------------------------------------------
    # 4. Evidence selection
    # ---------------------------------------------------------
    #
    # Keep strong dense matches OR documents that receive
    # meaningful support from both retrieval systems.
    #
    # This prevents the global 0.50 dense threshold from
    # destroying legitimate multi-section evidence.
    # ---------------------------------------------------------

    selected: list[dict[str, Any]] = []

    for result in fused_results:

        dense_score = result.get("dense_score")

        dense_rank = result.get("dense_rank")

        bm25_rank = result.get("bm25_rank")

        strong_dense = (
            dense_score is not None
            and dense_score >= settings.similarity_threshold
        )

        hybrid_supported = (
            dense_rank is not None
            and bm25_rank is not None
        )

        if strong_dense or hybrid_supported:
            selected.append(result)

    # ---------------------------------------------------------
    # 5. Return final contexts
    # ---------------------------------------------------------

    return selected[:final_top_k]