"""
SmartRAG — Central Configuration
All settings loaded from environment variables with sane defaults.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Literal


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────────────────────────
    APP_NAME: str = "SmartRAG"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False
    SECRET_KEY: str = "changeme-in-production-use-openssl-rand-hex-32"

    # ── LLM ──────────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str
    LLM_MODEL: str = "gpt-4o-mini"               # cheap default; swap to gpt-4o for quality
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 1024

    # ── RAG Engine ───────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    TOP_K_RETRIEVAL: int = 20          # candidates for first-stage retrieval
    TOP_K_RERANK: int = 5              # chunks sent to LLM after reranking
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    HALLUCINATION_THRESHOLD: float = 0.5   # below this → flag as uncertain

    # ── Vector Store ─────────────────────────────────────────────────────────
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    CHROMA_PERSIST_PATH: str = "./chroma_data"

    # ── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://smartrag:smartrag@localhost:5432/smartrag"

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_QUERIES_PER_MIN: int = 60
    MAX_DOCUMENT_SIZE_MB: int = 50
    MAX_DOCUMENTS_PER_TENANT: int = 500

    # ── CORS ─────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "https://smartrag.yourdomain.com"]

    model_config = {"env_file": ".env", "case_sensitive": True}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
