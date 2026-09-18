from app.clients import openai_client
from app.config import settings


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    response = openai_client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
        dimensions=settings.embedding_dimensions,
    )
    return [item.embedding for item in response.data]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
