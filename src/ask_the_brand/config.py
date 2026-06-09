"""Central, env-driven configuration for the Ask-the-Brand app.

All tunables live here so they can be overridden via .env without touching code.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"

    # --- Weaviate (local docker-compose) ---
    weaviate_host: str = "localhost"
    weaviate_http_port: int = 8080
    weaviate_grpc_port: int = 50051
    collection_name: str = "AcmeDoc"

    # --- Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- Retrieval ---
    default_alpha: float = 0.5  # 1.0 = pure vector, 0.0 = pure BM25
    default_top_k: int = 5
    summary_top_k: int = 40
    refusal_score_threshold: float = 0.15

    # --- App ---
    api_base_url: str = "http://localhost:8000"

    # Date of the headline event used as the default split point for comparisons.
    launch_date: str = "2026-04-10"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
