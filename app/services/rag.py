from app.services.generation import generate_answer
from app.services.retrieval import retrieve


NO_CONTEXT_MESSAGE = (
    "I don't have enough information in the knowledge base "
    "to answer that question."
)


def answer_question(
    question: str,
    top_k: int | None = None,
) -> dict:

    contexts = retrieve(
        question,
        top_k,
    )

    if not contexts:
        return {
            "answer": NO_CONTEXT_MESSAGE,
            "sources": [],
        }

    answer = generate_answer(
        question,
        contexts,
    )

    return {
        "answer": answer,
        "sources": contexts,
    }