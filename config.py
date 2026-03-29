from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    environment: Literal["development", "test", "production"] = "production"
    log_level: str = "INFO"

    database_url: str = Field(..., description="Database URL")
    database_pool_size: int = 10
    database_max_overflow: int = 20

    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str = ""
    qdrant_collection_documents: str = "documents"
    qdrant_collection_web: str = "web_results"

    redis_url: str = "redis://:password@redis:6379/0"
    celery_broker_url: str = ""
    celery_result_backend: str = ""

    embedding_url: str = "http://embedding:8001"
    reranker_url: str = "http://reranker:8002"

    anthropic_api_key: str = Field(..., description="Anthropic API key")
    llm_model: str = "claude-sonnet-4-5"
    llm_max_tokens: int = 8192
    llm_temperature: float = 0.3

    tavily_api_key: str = Field(..., description="Tavily search API key")
    tavily_max_results: int = 8
    tavily_search_depth: Literal["basic", "advanced"] = "advanced"

    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    max_file_size_mb: int = 50
    storage_path: str = "/data/uploads"

    rag_top_k_initial: int = 20
    rag_top_k_reranked: int = 8
    rag_hybrid_alpha: float = 0.7
    rag_min_score: float = 0.3

    web_fetch_timeout_sec: int = 15
    web_max_content_chars: int = 20_000  # truncate long pages

    report_language: str = "vi"
    report_output_path: str = "/data/reports"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @field_validator("celery_broker_url", mode="before")
    @classmethod
    def default_broker(cls, v, info):
        return v or info.data.get("redis_url", "")

    @field_validator("celery_result_backend", mode="before")
    @classmethod
    def default_backend(cls, v, info):
        return v or info.data.get("redis_url", "")

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


SUPPORTED_MIME_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "doc",
    "text/plain": "txt",
    "text/markdown": "md",
}

# BGE-M3 dense vector dim
VECTOR_DIM = 1024

# Celery task names (single source of truth)
TASK_INGEST_DOCUMENT = "tasks.ingest_document"
TASK_RUN_PIPELINE = "tasks.run_pipeline"
TASK_GENERATE_REPORT = "tasks.generate_report"
TASK_WEB_SEARCH = "tasks.web_search"
TASK_INDEX_CHUNKS = "tasks.index_chunks"