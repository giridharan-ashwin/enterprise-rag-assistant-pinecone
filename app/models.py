from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )


class Citation(BaseModel):
    source: str
    section: str
    chunk_index: int | None = None


class RetrievedContext(BaseModel):
    source: str
    section: str
    chunk_index: int | None = None
    score: float | None = None
    text: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    abstained: bool
    contexts: list[RetrievedContext]