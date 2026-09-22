from fastapi import FastAPI, HTTPException

from app.models import AskRequest, AskResponse
from app.services.rag import answer_question


app = FastAPI(
    title="Enterprise RAG Assistant",
    version="1.0.0",
    description=(
        "Enterprise RAG assistant with hybrid retrieval, "
        "threshold filtering, grounded answers, and citations."
    ),
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "enterprise-rag-assistant",
    }


@app.post(
    "/api/ask",
    response_model=AskResponse,
)
def ask(request: AskRequest) -> AskResponse:
    try:
        result = answer_question(
            request.question,
            request.top_k,
        )

        return AskResponse(
            answer=result["answer"],
            citations=result.get("citations", []),
            abstained=result.get("abstained", False),
            contexts=result.get("contexts", []),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# Backward-compatible endpoint.
@app.post(
    "/query",
    response_model=AskResponse,
)
def query(request: AskRequest) -> AskResponse:
    return ask(request)