from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Literal, Optional

from celery.result import AsyncResult
from fastapi import (
    FastAPI, HTTPException, UploadFile, File, Depends, Query, Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from config import get_settings, SUPPORTED_MIME_TYPES

app = FastAPI(
    title="Multi-Agent System",
    description="Multi-Agent System — MVP Linear Pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_cfg():
    return get_settings()


def get_celery():
    from services.worker.tasks import celery_app
    return celery_app


# ─── Request / Response schemas ───────────────────────────────────────────────

class DocumentResponse(BaseModel):
    document_id: str
    task_id: str
    status: str
    filename: str
    message: str


class ReportRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=1000,
                       description="Research query / topic")
    document_ids: Optional[list[str]] = Field(
        None, description="Restrict RAG to these document IDs"
    )
    include_domains: Optional[list[str]] = None
    exclude_domains: Optional[list[str]] = None
    output_formats: list[Literal["pdf", "docx"]] = ["pdf", "docx"]


class ReportResponse(BaseModel):
    report_id: str
    task_id: str
    status: str
    message: str


class WebSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    max_results: int = Field(5, ge=1, le=20)
    fetch_content: bool = True
    include_domains: Optional[list[str]] = None


class RAGSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(8, ge=1, le=30)
    document_ids: Optional[list[str]] = None
    min_score: float = Field(0.0, ge=0.0, le=1.0)


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _task_status(task_id: str) -> TaskStatusResponse:
    result = AsyncResult(task_id, app=get_celery())
    if result.state == "PENDING":
        return TaskStatusResponse(task_id=task_id, status="pending")
    elif result.state == "STARTED":
        return TaskStatusResponse(task_id=task_id, status="running")
    elif result.state == "SUCCESS":
        return TaskStatusResponse(task_id=task_id, status="completed",
                                  result=result.result if isinstance(result.result, dict) else {})
    elif result.state == "FAILURE":
        return TaskStatusResponse(task_id=task_id, status="failed",
                                  error=str(result.result))
    else:
        return TaskStatusResponse(task_id=task_id, status=result.state.lower())


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health(cfg=Depends(get_cfg)):
    return {
        "status": "ok",
        "service": "Multi-Agent System",
        "version": "1.0.0",
        "llm_model": cfg.llm_model,
        "embedding_url": cfg.embedding_url,
        "qdrant_url": cfg.qdrant_url,
    }


# ─── Documents ────────────────────────────────────────────────────────────────

@app.post("/documents/upload", response_model=DocumentResponse, status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    cfg=Depends(get_cfg),
):
    content_type = file.content_type or ""
    suffix = Path(file.filename or "").suffix.lower().lstrip(".")

    file_type = SUPPORTED_MIME_TYPES.get(content_type) or (
        suffix if suffix in ("pdf", "docx", "txt", "md") else None
    )
    if file_type is None:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {content_type or suffix}. "
                   f"Allowed: PDF, DOCX, TXT, MD",
        )

    # Size check
    file_bytes = await file.read()
    max_bytes = cfg.max_file_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(file_bytes)//1024//1024}MB). "
                   f"Max: {cfg.max_file_size_mb}MB",
        )

    # Save to storage
    document_id = str(uuid.uuid4())
    storage_dir = Path(cfg.storage_path)
    storage_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{document_id}.{file_type}"
    file_path = storage_dir / safe_name

    file_path.write_bytes(file_bytes)

    sha256 = hashlib.sha256(file_bytes).hexdigest()

    # Dispatch Celery task
    from services.worker.tasks import ingest_document
    task = ingest_document.delay(
        file_path=str(file_path),
        document_id=document_id,
        file_type=file_type,
        extra_payload={
            "original_filename": file.filename,
            "sha256": sha256,
            "file_size": len(file_bytes),
        },
    )

    return DocumentResponse(
        document_id=document_id,
        task_id=task.id,
        status="queued",
        filename=file.filename or safe_name,
        message="Document queued for ingestion. Poll /jobs/{task_id} for status.",
    )


@app.get("/documents/{document_id}/status")
async def document_status(document_id: str):
    return {"document_id": document_id, "message": "Query DB for status"}


@app.delete("/documents/{document_id}", status_code=200)
async def delete_document(document_id: str, cfg=Depends(get_cfg)):
    from services.ingestion.pipeline import IngestionPipeline
    pipeline = IngestionPipeline(
        embedding_url=cfg.embedding_url,
        qdrant_url=cfg.qdrant_url,
        qdrant_api_key=cfg.qdrant_api_key,
    )
    pipeline.delete_document(document_id)

    for ext in ("pdf", "docx", "txt", "md"):
        path = Path(cfg.storage_path) / f"{document_id}.{ext}"
        if path.exists():
            path.unlink()

    return {"document_id": document_id, "status": "deleted"}


# ─── Reports ──────────────────────────────────────────────────────────────────

@app.post("/reports/generate", response_model=ReportResponse, status_code=202)
async def generate_report(req: ReportRequest, cfg=Depends(get_cfg)):
    from services.worker.tasks import run_pipeline, generate_report as format_task
    from celery import chain

    report_id = str(uuid.uuid4())

    task_chain = chain(
        run_pipeline.s(
            query=req.query,
            report_id=report_id,
            document_ids=req.document_ids,
            include_domains=req.include_domains,
            exclude_domains=req.exclude_domains,
        ),
        format_task.s(
            output_formats=req.output_formats,
            output_dir=cfg.report_output_path,
        ),
    )

    result = task_chain.apply_async()

    return ReportResponse(
        report_id=report_id,
        task_id=result.id,
        status="queued",
        message=f"Report queued. Poll /jobs/{result.id} for status. "
                f"Download at /reports/{report_id}/download when complete.",
    )


@app.get("/reports/{report_id}/download")
async def download_report(
    report_id: str,
    fmt: Literal["pdf", "docx"] = Query("pdf"),
    cfg=Depends(get_cfg),
):
    out_dir = Path(cfg.report_output_path)
    file_path = out_dir / f"{report_id}.{fmt}"

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report {report_id}.{fmt} not found. "
                   "Check task status — may still be generating.",
        )

    media_types = {
        "pdf":  "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    return FileResponse(
        path=str(file_path),
        media_type=media_types[fmt],
        filename=f"report_{report_id[:8]}.{fmt}",
    )


# ─── Search ───────────────────────────────────────────────────────────────────

@app.post("/search/web")
async def web_search(req: WebSearchRequest, cfg=Depends(get_cfg)):
    from services.retrieval.web_search import WebSearchPipeline
    searcher = WebSearchPipeline(
        tavily_api_key=cfg.tavily_api_key,
        max_results=req.max_results,
        fetch_full_content=req.fetch_content,
    )
    try:
        result = searcher.search(
            query=req.query,
            include_domains=req.include_domains,
        )
        return {
            "query": result.query,
            "answer": result.tavily_answer,
            "results": [
                {
                    "url":      r.url,
                    "title":    r.title,
                    "snippet":  r.snippet,
                    "content_preview": (r.full_content or "")[:500],
                    "score":    r.score,
                    "word_count": r.word_count,
                }
                for r in result.results
            ],
            "total": result.total_fetched,
            "elapsed_ms": result.elapsed_ms,
        }
    finally:
        searcher.close()


@app.post("/search/rag")
async def rag_search(req: RAGSearchRequest, cfg=Depends(get_cfg)):
    from services.retrieval.hybrid_search import HybridSearcher
    searcher = HybridSearcher(
        qdrant_url=cfg.qdrant_url,
        embedding_url=cfg.embedding_url,
        reranker_url=cfg.reranker_url,
        collection=cfg.qdrant_collection_documents,
        qdrant_api_key=cfg.qdrant_api_key,
    )
    try:
        result = searcher.search(
            query=req.query,
            top_k_initial=req.top_k * 3,
            top_k_final=req.top_k,
            min_score=req.min_score,
            document_ids=req.document_ids,
        )
        return {
            "query":      result.query,
            "chunks":     [c.to_source_dict() for c in result.chunks],
            "total":      result.total_retrieved,
            "returned":   len(result.chunks),
            "elapsed_ms": result.elapsed_ms,
        }
    finally:
        searcher.close()


# ─── Job status ───────────────────────────────────────────────────────────────

@app.get("/jobs/{task_id}", response_model=TaskStatusResponse)
async def job_status(task_id: str):
    return _task_status(task_id)


# ─── Global error handler ─────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    import logging
    logging.getLogger("api").exception(f"Unhandled: {request.url}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})