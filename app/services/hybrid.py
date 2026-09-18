from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.config import settings
from app.services.chunker import chunk_text

STRESS_DATA_DIR = Path("data/stress_test")


def tokenize(text: str) -> list[str]:
    """
    Simple normalized tokenizer for BM25.

    Keeps technical tokens reasonably intact while making
    matching case-insensitive.
    """
    return re.findall(r"[a-zA-Z0-9_.:/-]+", text.lower())


def load_stress_chunks() -> list[dict[str, Any]]:
    """
    Recreate the same chunks that were ingested into Pinecone.

    The chunk IDs/metadata remain aligned with the Pinecone corpus.
    """
    chunks: list[dict[str, Any]] = []

    for path in sorted(STRESS_DATA_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")

        document_chunks = chunk_text(
            text=text,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        for index, chunk in enumerate(document_chunks):
            chunks.append(
                {
                    "source": path.name,
                    "section": chunk["section"],
                    "chunk_index": index,
                    "text": chunk["text"],
                    "lexical_text": (
                        f"{chunk['section']} "
                        f"{path.stem.replace('_', ' ')} "
                        f"{chunk['text']}"
                    ),
                    "id": f"{path.name}-{index}",
                }
            )

    return chunks


class BM25Retriever:
    def __init__(self, documents: list[dict[str, Any]]):
        self.documents = documents

        tokenized_documents = [tokenize(document["lexical_text"]) for document in documents]

        self.index = BM25Okapi(tokenized_documents)

    def retrieve(
        self,
        question: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        tokens = tokenize(question)

        scores = self.index.get_scores(tokens)

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results = []

        for index in ranked_indexes:
            document = self.documents[index]

            results.append(
                {
                    **document,
                    "bm25_score": float(scores[index]),
                }
            )

        return results


_documents = load_stress_chunks()
bm25_retriever = BM25Retriever(_documents)


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Combine dense and BM25 rankings using Reciprocal Rank Fusion.

    RRF score:
        1 / (k + rank)

    Scores from the two retrieval systems don't need to be
    on the same numerical scale.
    """

    fused: dict[str, dict[str, Any]] = {}

    # Dense contribution
    for rank, result in enumerate(dense_results, start=1):
        document_id = result["id"]

        if document_id not in fused:
            fused[document_id] = {
                **result,
                "dense_rank": rank,
                "bm25_rank": None,
                "rrf_score": 0.0,
            }

        fused[document_id]["rrf_score"] += 1 / (k + rank)

    # BM25 contribution
    for rank, result in enumerate(sparse_results, start=1):
        document_id = result["id"]

        if document_id not in fused:
            fused[document_id] = {
                **result,
                "dense_rank": None,
                "bm25_rank": rank,
                "dense_score": None,
                "rrf_score": 0.0,
            }

        fused[document_id]["bm25_rank"] = rank
        fused[document_id]["bm25_score"] = result["bm25_score"]
        fused[document_id]["rrf_score"] += 1 / (k + rank)

    ranked = sorted(
        fused.values(),
        key=lambda result: result["rrf_score"],
        reverse=True,
    )

    return ranked[:top_k]


def hybrid_retrieve(
    question: str,
    dense_retriever,
    dense_top_k: int = 10,
    sparse_top_k: int = 10,
    final_top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Run dense and BM25 retrieval independently,
    then fuse their rankings with RRF.
    """

    dense_results = dense_retriever(
        question,
        dense_top_k,
    )

    sparse_results = bm25_retriever.retrieve(
        question,
        sparse_top_k,
    )

    return reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        top_k=final_top_k,
    )
