from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int | None = Field(default=None, ge=1, le=20)

class Source(BaseModel):
    source: str
    section: str
    chunk_index: int
    score: float
    rerank_score: float | None = None
    text: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
