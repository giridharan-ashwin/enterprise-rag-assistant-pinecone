from __future__ import annotations

from typing import Any

from app.clients import pinecone_index
from app.config import settings
from app.services.embeddings import embed_query


def dense_retrieve(
    question: str,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve documents from Pinecone using dense vector similarity.

    No Pinecone reranking is performed.
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
    Production retrieval pipeline:

        Dense retrieval
             +
        BM25 sparse retrieval
             ↓
        Reciprocal Rank Fusion
             ↓
        Dense similarity threshold
             ↓
        Final contexts

    Pinecone reranking is intentionally disabled.
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
    # 2. Dense similarity threshold
    # ---------------------------------------------------------

    dense_results = [
        result
        for result in dense_results
        if result["dense_score"] >= settings.similarity_threshold
    ]

    if not dense_results:
        return []

    # ---------------------------------------------------------
    # 3. Sparse BM25 retrieval
    # ---------------------------------------------------------

    from app.services.hybrid import (
        bm25_retriever,
        reciprocal_rank_fusion,
    )

    sparse_results = bm25_retriever.retrieve(
        question,
        top_k=candidate_k,
    )

    # ---------------------------------------------------------
    # 4. Reciprocal Rank Fusion
    # ---------------------------------------------------------

    fused_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        top_k=final_top_k,
    )

    return fused_results