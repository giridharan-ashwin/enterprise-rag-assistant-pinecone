from app.clients import pinecone_client, pinecone_index
from app.config import settings
from app.services.embeddings import embed_query


def retrieve(
    question: str,
    top_k: int | None = None,
) -> list[dict]:

    vector = embed_query(question)

    limit = top_k or settings.top_k

    response = pinecone_index.query(
    vector=vector,
    top_k=limit,
    include_metadata=True,
    namespace=settings.pinecone_namespace,
)

    candidates = []

    for match in response["matches"]:
        score = float(match.get("score", 0))

        # Stage 1: similarity filtering
        if score < settings.similarity_threshold:
            continue

        metadata = match.get("metadata", {})

        candidates.append(
            {
                "source": metadata.get("source", "unknown"),
                "section": metadata.get("section", "unknown"),
                "chunk_index": metadata.get("chunk_index", -1),
                "score": score,
                "text": metadata.get("text", ""),
            }
        )

    # Nothing survived the similarity threshold.
    if not candidates:
        return []

    # Stage 2: reranking
    documents = [
        {
            "id": str(index),
            "text": candidate["text"],
        }
        for index, candidate in enumerate(candidates)
    ]

    rerank_response = pinecone_client.inference.rerank(
        model=settings.rerank_model,
        query=question,
        documents=documents,
        top_n=min(settings.rerank_top_n, len(documents)),
        return_documents=False,
    )

    reranked_results = []

    for result in rerank_response.data:
        candidate_index = result.index
        candidate = candidates[candidate_index]

        reranked_results.append(
            {
                **candidate,
                "rerank_score": float(result.score),
            }
        )

    return reranked_results