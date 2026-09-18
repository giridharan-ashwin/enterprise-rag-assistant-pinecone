from __future__ import annotations

from app.clients import pinecone_index
from app.services.embeddings import embed_query
from app.services.hybrid import hybrid_retrieve


NAMESPACE = "stress-test"


def dense_retrieve(
    question: str,
    top_k: int = 10,
):
    vector = embed_query(question)

    response = pinecone_index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        namespace=NAMESPACE,
    )

    results = []

    for match in response.get("matches", []):
        metadata = match.get("metadata", {}) or {}

        results.append(
            {
                "id": str(match["id"]),
                "source": metadata.get("source", "unknown"),
                "section": metadata.get("section", "unknown"),
                "chunk_index": metadata.get("chunk_index", -1),
                "text": metadata.get("text", ""),
                "score": float(match.get("score", 0.0)),
            }
        )

    return results


def main():
    test_queries = [
        "What is the process for deploying an API?",
        "What are the requirements for privileged access?",
        "How are employee expenses submitted?",
        "What happens during a security incident?",
        "What is the Kubernetes deployment policy?",
        "What is the data retention policy?",
        "What is the vendor onboarding process?",
        "What happens when a vendor is onboarded and later offboarded?",
        "What is the API versioning standard?",
        "What is the incident escalation process?",
    ]

    for question in test_queries:
        print()
        print("=" * 90)
        print(question)
        print("=" * 90)

        results = hybrid_retrieve(
            question=question,
            dense_retriever=dense_retrieve,
            dense_top_k=10,
            sparse_top_k=10,
            final_top_k=5,
        )

        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. "
                f"{result['source']} | "
                f"{result['section']} | "
                f"RRF={result['rrf_score']:.5f} | "
                f"dense_rank={result.get('dense_rank')} | "
                f"bm25_rank={result.get('bm25_rank')}"
            )


if __name__ == "__main__":
    main()
