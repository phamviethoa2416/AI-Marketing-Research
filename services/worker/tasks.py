from __future__ import annotations

import uuid
from pathlib import Path

from celery import Celery
from celery.utils.log import get_task_logger

from config import get_settings, TASK_INGEST_DOCUMENT, TASK_RUN_PIPELINE, TASK_GENERATE_REPORT

log = get_task_logger(__name__)

def create_celery_app() -> Celery:
    cfg = get_settings()
    app = Celery(
        "multi-agents-system-workers",
        broker=cfg.celery_broker_url,
        backend=cfg.celery_result_backend,
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="Asia/Ho_Chi_Minh",
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        result_expires=86400,
        task_routes={
            "tasks.ingest_document": {"queue": "ingest"},
            "tasks.run_pipeline": {"queue": "pipeline"},
            "tasks.generate_report": {"queue": "pipeline"},
            "tasks.export_pdf": {"queue": "format"},
            "tasks.export_docx": {"queue": "format"},
        },
        task_queues={
            "default": {"exchange": "default", "routing_key": "default"},
            "ingest": {"exchange": "ingest", "routing_key": "ingest"},
            "pipeline": {"exchange": "pipeline", "routing_key": "pipeline"},
            "format": {"exchange": "format", "routing_key": "format"},
        },
        task_default_queue="default",
        worker_max_tasks_per_child=50,
    )
    return app


celery_app = create_celery_app()

_ingestion_pipeline = None
_web_searcher       = None
_rag_searcher       = None
_linear_pipeline    = None

def _get_ingestion():
    global _ingestion_pipeline
    if _ingestion_pipeline is None:
        cfg = get_settings()
        from services.ingestion.pipeline import IngestionPipeline
        _ingestion_pipeline = IngestionPipeline(
            embedding_url=cfg.embedding_url,
            qdrant_url=cfg.qdrant_url,
            qdrant_api_key=cfg.qdrant_api_key,
            collection=cfg.qdrant_collection_documents,
            chunk_size=cfg.chunk_size_tokens,
            chunk_overlap=cfg.chunk_overlap_tokens,
        )
    return _ingestion_pipeline

def _get_web_searcher():
    global _web_searcher
    if _web_searcher is None:
        cfg = get_settings()
        from services.retrieval.web_search import WebSearchPipeline
        _web_searcher = WebSearchPipeline(
            tavily_api_key=cfg.tavily_api_key,
            embedding_url=cfg.embedding_url,
            qdrant_url=cfg.qdrant_url,
            qdrant_api_key=cfg.qdrant_api_key,
            qdrant_collection=cfg.qdrant_collection_web,
            max_results=cfg.tavily_max_results,
            search_depth=cfg.tavily_search_depth,
        )
    return _web_searcher

def _get_rag_searcher():
    global _rag_searcher
    if _rag_searcher is None:
        cfg = get_settings()
        from services.retrieval.hybrid_search import HybridSearcher
        _rag_searcher = HybridSearcher(
            qdrant_url=cfg.qdrant_url,
            embedding_url=cfg.embedding_url,
            reranker_url=cfg.reranker_url,
            collection=cfg.qdrant_collection_documents,
            qdrant_api_key=cfg.qdrant_api_key,
        )
    return _rag_searcher

def _get_linear_pipeline():
    global _linear_pipeline
    if _linear_pipeline is None:
        cfg = get_settings()
        from services.pipeline.linear import LinearPipeline
        _linear_pipeline = LinearPipeline(
            anthropic_api_key=cfg.anthropic_api_key,
            web_searcher=_get_web_searcher(),
            rag_searcher=_get_rag_searcher(),
            llm_model=cfg.llm_model,
            llm_max_tokens=cfg.llm_max_tokens,
            llm_temperature=cfg.llm_temperature,
            rag_top_k=cfg.rag_top_k_reranked,
        )
    return _linear_pipeline

@celery_app.task(
    name=TASK_INGEST_DOCUMENT,
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    soft_time_limit=300,
    time_limit=360,
)
def ingest_document(
        self,
        file_path: str,
        document_id: str,
        file_type: str | None = None,
        extra_payload: dict | None = None,
) -> dict:
    log.info(f"[ingest] START doc={document_id} file={file_path}")
    try:
        pipeline = _get_ingestion()
        result = pipeline.ingest(
            file_path=Path(file_path),
            document_id=document_id,
            file_type=file_type,
            extra_payload=extra_payload or {},
        )
        out = {
            "document_id": result.document_id,
            "chunk_count": result.chunk_count,
            "indexed_count": result.indexed_count,
            "elapsed_sec": result.elapsed_sec,
            "success": result.success,
            "errors": result.errors,
        }
        log.info(f"[ingest] DONE doc={document_id}: {result.indexed_count} chunks indexed")
        return out

    except Exception as exc:
        log.error(f"[ingest] FAILED doc={document_id}: {exc}")
        raise self.retry(exc=exc)


@celery_app.task(
    name=TASK_RUN_PIPELINE,
    bind=True,
    max_retries=1,
    default_retry_delay=60,
    soft_time_limit=180,
    time_limit=240,
)
def run_pipeline(
        self,
        query: str,
        report_id: str | None = None,
        document_ids: list[str] | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
) -> dict:
    log.info(f"[pipeline] START report={report_id} query='{query[:60]}'")
    try:
        pipeline = _get_linear_pipeline()
        output = pipeline.run(
            query=query,
            document_ids=document_ids,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
        result = {
            "report_id": report_id or str(uuid.uuid4()),
            "query": output.query,
            "report_title": output.report_title,
            "markdown_content": output.markdown_content,
            "sources": [
                {
                    "url": s.url,
                    "title": s.title,
                    "snippet": s.snippet,
                    "source_type": s.source_type,
                    "relevance_score": s.relevance_score,
                }
                for s in output.sources
            ],
            "quality": output.quality_summary,
            "success": True,
        }
        log.info(
            f"[pipeline] DONE report={report_id}: "
            f"{output.word_count} words, {output.total_elapsed_ms}ms"
        )
        return result

    except Exception as exc:
        log.error(f"[pipeline] FAILED report={report_id}: {exc}")
        raise self.retry(exc=exc)


@celery_app.task(
    name=TASK_GENERATE_REPORT,
    bind=True,
    max_retries=1,
    soft_time_limit=120,
    time_limit=150,
)
def generate_report(
        self,
        pipeline_result: dict,
        output_formats: list[str] | None = None,
        output_dir: str | None = None,
) -> dict:
    output_formats = output_formats or ["pdf", "docx"]
    cfg = get_settings()
    out_dir = Path(output_dir or cfg.report_output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_id = pipeline_result.get("report_id", str(uuid.uuid4()))
    log.info(f"[format] START report={report_id} formats={output_formats}")

    outputs: dict[str, str] = {}

    if "pdf" in output_formats:
        try:
            from services.formatter.pdf_formatter import PDFFormatter
            pdf_path = out_dir / f"{report_id}.pdf"
            PDFFormatter().generate(pipeline_result, str(pdf_path))
            outputs["pdf_path"] = str(pdf_path)
            log.info(f"[format] PDF done: {pdf_path}")
        except Exception as e:
            log.error(f"[format] PDF failed: {e}")
            outputs["pdf_error"] = str(e)

    if "docx" in output_formats:
        try:
            from services.formatter.docx_formatter import DocxFormatter
            docx_path = out_dir / f"{report_id}.docx"
            DocxFormatter().generate(pipeline_result, str(docx_path))
            outputs["docx_path"] = str(docx_path)
            log.info(f"[format] DOCX done: {docx_path}")
        except Exception as e:
            log.error(f"[format] DOCX failed: {e}")
            outputs["docx_error"] = str(e)

    return {
        "report_id": report_id,
        "success": "pdf_path" in outputs or "docx_path" in outputs,
        **outputs,
    }