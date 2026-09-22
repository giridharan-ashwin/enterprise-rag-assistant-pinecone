from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 512

    pinecone_api_key: str
    pinecone_index_name: str = "enterprise-rag"
    pinecone_namespace: str = ""
    top_k: int = 5
    similarity_threshold: float = 0.50

    rerank_model: str = "bge-reranker-v2-m3"
    rerank_top_n: int = 3

    chunk_size: int = 900
    chunk_overlap: int = 120

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
