"""Application configuration via pydantic-settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    redis_url: str = "redis://localhost:6379"
    memory_server_url: str = "http://localhost:9200"

    ollama_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:7b"
    embedding_model: str = "all-MiniLM-L6-v2"

    chunk_size: int = 500
    chunk_overlap: int = 150
    top_k: int = 5

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5
    bm25_weight: float = 0.5

    cache_distance_threshold: float = 0.10
    cache_ttl: int = 3600  # seconds
    session_ttl: int = 86400  # 24 hours

    # Guardrails (opt-in)
    guardrails_enabled: bool = False
    max_query_length: int = 1000
    pii_redaction: bool = True
    max_response_length: int = 2000
    offtopic_threshold: float = 0.15


settings = Settings()
