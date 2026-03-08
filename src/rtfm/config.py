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
    chunk_overlap: int = 50
    top_k: int = 5

    cache_distance_threshold: float = 0.15
    cache_ttl: int = 3600  # seconds
    session_ttl: int = 86400  # 24 hours


settings = Settings()
