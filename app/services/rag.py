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
        top_k=top_k,
    )

    if not contexts:
        return {
            "answer": NO_CONTEXT_MESSAGE,
            "citations": [],
            "abstained": True,
            "contexts": [],
        }

    result = generate_answer(
        question,
        contexts,
    )

    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "abstained": result.get(
            "abstained",
            False,
        ),
        "contexts": contexts,
    }