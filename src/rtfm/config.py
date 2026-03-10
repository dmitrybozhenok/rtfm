"""Application configuration via pydantic-settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    redis_url: str = "redis://localhost:6379"
    memory_server_url: str = "http://localhost:9200"

    ollama_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen2.5:7b"
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_dims: int = 768

    chunk_size: int = 500
    chunk_overlap: int = 150
    top_k: int = 5

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5
    bm25_weight: float = 0.5

    cache_distance_threshold: float = 0.10
    cache_ttl: int = 3600  # seconds
    session_ttl: int = 86400  # 24 hours

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"
    log_format: str = "json"  # "json" or "text"

    # Observability
    tracing_enabled: bool = False
    tracing_exporter_url: str = ""  # OTLP gRPC endpoint, e.g. "http://localhost:4317"

    # Authentication (opt-in)
    auth_enabled: bool = False
    auth_username: str = "admin"
    auth_password: str = "changeme"

    # Guardrails (opt-in)
    guardrails_enabled: bool = False
    max_query_length: int = 1000
    pii_redaction: bool = True
    max_response_length: int = 2000
    offtopic_threshold: float = 0.15


settings = Settings()
