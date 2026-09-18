from fastapi import FastAPI, HTTPException
from app.models import QueryRequest, QueryResponse
from app.services.rag import answer_question

app = FastAPI(title="Enterprise RAG Assistant", version="0.1.0")

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        return QueryResponse(**answer_question(request.question, request.top_k))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
