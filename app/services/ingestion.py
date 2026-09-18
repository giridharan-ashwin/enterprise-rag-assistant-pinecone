from pathlib import Path

from app.clients import pinecone_index
from app.config import settings
from app.services.chunker import chunk_text
from app.services.embeddings import embed_texts


def ingest_file(path: str) -> int:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    text = file_path.read_text(encoding="utf-8")

    chunks = chunk_text(
        text=text,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    if not chunks:
        return 0

    chunk_texts = [chunk["text"] for chunk in chunks]

    embeddings = embed_texts(chunk_texts)

    vectors = []

    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_id = f"{file_path.name}-{index}"

        vectors.append(
            {
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "source": file_path.name,
                    "section": chunk["section"],
                    "chunk_index": index,
                    "text": chunk["text"],
                },
            }
        )

        pinecone_index.upsert(
        vectors=vectors,
        namespace=settings.pinecone_namespace,
        )

    return len(vectors)