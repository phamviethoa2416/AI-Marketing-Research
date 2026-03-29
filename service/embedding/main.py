from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from FlagEmbedding import BGEM3FlagModel

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field, field_validator
from starlette.responses import Response

MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-m3")
DEVICE = os.getenv("DEVICE", "cpu")
MAX_BATCH = int(os.getenv("MAX_BATCH", "32"))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "8192"))
CACHE_DIR = os.getenv("CACHE_DIR", "/models")
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "256"))
QUEUE_BATCH_REQUESTS = int(os.getenv("QUEUE_BATCH_REQUESTS", "16"))
QUEUE_BATCH_WAIT_MS = int(os.getenv("QUEUE_BATCH_WAIT_MS", "10"))
INFER_TIMEOUT_SEC = float(os.getenv("INFER_TIMEOUT_SEC", "30"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("embedding")

REQUEST_COUNT  = Counter("embedding_requests_total",   "Total embedding requests", ["mode"])
REQUEST_ERRORS = Counter("embedding_errors_total",     "Total embedding errors")
REQUEST_LAT    = Histogram("embedding_latency_seconds", "Embedding latency", ["mode"],
                           buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30])
BATCH_SIZE_H   = Histogram("embedding_batch_size",     "Texts per batch",
                           buckets=[1, 2, 4, 8, 16, 32, 64])
MODEL_LOADED   = Gauge("embedding_model_loaded",       "1 if model is loaded")

_model: BGEM3FlagModel | None = None
_inference_queue: asyncio.Queue[InferenceTask] | None = None
_worker_task: asyncio.Task[None] | None = None


@dataclass
class InferenceTask:
    req: EmbeddingRequest
    mode: str
    batch_size: int
    max_length: int
    future: asyncio.Future[dict[str, Any]]


def get_model() -> BGEM3FlagModel:
    if _model is None:
        raise RuntimeError("Model not loaded")
    return _model


def load_model() -> None:
    global _model
    from FlagEmbedding import BGEM3FlagModel
    log.info(f"Loading model {MODEL_NAME} on {DEVICE}...")
    t0 = time.time()
    use_fp16 = DEVICE != "cpu"
    _model = BGEM3FlagModel(
        model_name_or_path=MODEL_NAME,
        device=DEVICE,
        use_fp16=use_fp16,
        cache_dir=CACHE_DIR,
    )
    log.info(f"Model loaded in {time.time() - t0:.2f}s")
    MODEL_LOADED.set(1)


def _queue_or_raise() -> asyncio.Queue[InferenceTask]:
    if _inference_queue is None:
        raise RuntimeError("Inference queue is not initialized")
    return _inference_queue


def _extract_embeddings(output: dict[str, Any], text_count: int, return_dense: bool, return_sparse: bool) -> tuple[list[dict[str, Any]], Optional[int]]:
    embeddings: list[dict[str, Any]] = []
    for i in range(text_count):
        entry: dict[str, Any] = {"index": i}
        if return_dense:
            entry["dense"] = output["dense_vecs"][i].tolist()
        if return_sparse:
            sparse_weights = output["lexical_weights"][i]
            items = sorted(sparse_weights.items(), key=lambda x: x[0])
            entry["sparse_indices"] = [int(k) for k, _ in items]
            entry["sparse_values"] = [float(v) for _, v in items]
        embeddings.append(entry)

    dense_dim = len(embeddings[0]["dense"]) if return_dense and embeddings else None
    return embeddings, dense_dim


async def _process_group(tasks: list[InferenceTask]) -> None:
    if not tasks:
        return

    first = tasks[0]
    return_dense = first.mode in ("dense", "hybrid")
    return_sparse = first.mode in ("sparse", "hybrid")

    all_texts: list[str] = []
    offsets: list[tuple[int, int]] = []
    for task in tasks:
        start = len(all_texts)
        all_texts.extend(task.req.texts)
        offsets.append((start, len(all_texts)))

    model = get_model()
    output = await asyncio.to_thread(
        model.encode,
        all_texts,
        batch_size=first.batch_size,
        max_length=first.max_length,
        return_dense=return_dense,
        return_sparse=return_sparse,
        return_colbert_vecs=False,
    )

    for task, (start, end) in zip(tasks, offsets):
        if task.future.cancelled():
            continue
        text_count = end - start
        chunk_output: dict[str, Any] = {}
        if return_dense:
            chunk_output["dense_vecs"] = output["dense_vecs"][start:end]
        if return_sparse:
            chunk_output["lexical_weights"] = output["lexical_weights"][start:end]

        embeddings, dense_dim = _extract_embeddings(chunk_output, text_count, return_dense, return_sparse)
        task.future.set_result(
            {
                "embeddings": embeddings,
                "dense_dim": dense_dim,
            }
        )


async def _process_batch(batch: list[InferenceTask]) -> None:
    grouped: dict[tuple[str, int, int], list[InferenceTask]] = defaultdict(list)
    for task in batch:
        grouped[(task.mode, task.batch_size, task.max_length)].append(task)

    for group_tasks in grouped.values():
        try:
            await _process_group(group_tasks)
        except Exception as exc:
            for task in group_tasks:
                if not task.future.done():
                    task.future.set_exception(exc)


async def _inference_worker() -> None:
    queue = _queue_or_raise()
    try:
        while True:
            first = await queue.get()
            batch = [first]

            deadline = time.monotonic() + (QUEUE_BATCH_WAIT_MS / 1000.0)
            while len(batch) < QUEUE_BATCH_REQUESTS:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=timeout)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            await _process_batch(batch)

            for _ in batch:
                queue.task_done()
    except asyncio.CancelledError:
        log.info("Inference worker stopped")
        raise


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Texts to embed")
    mode: str = Field("dense", description="dense | sparse | hybrid")
    batch_size: Optional[int] = Field(None, ge=1, le=64)
    max_seq_len: Optional[int] = Field(None, ge=1, le=8192)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v):
        if len(v) == 0:
            raise ValueError("texts must not be empty")
        if len(v) > 512:
            raise ValueError("texts must not exceed 512 elements")
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f"texts[{i}] must not be empty")
            if len(text) > 32768:
                raise ValueError(f"texts[{i}] too long ({len(text)} chars)")
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v):
        allowed = {"dense", "sparse", "hybrid"}
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}")
        return v


class EmbeddingResponse(BaseModel):
    model: str
    mode: str
    count: int
    dense_dim: Optional[int] = None
    embeddings: list[dict]
    elapsed_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _inference_queue, _worker_task

    load_model()
    _inference_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
    _worker_task = asyncio.create_task(_inference_worker())

    yield

    if _worker_task is not None:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass

    MODEL_LOADED.set(0)
    log.info("Embedding service shutting down...")


app = FastAPI(
    title="Embedding Service",
    description="A service that generates dense, sparse, or hybrid embeddings using a BGE-M3 model.",
    version="1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    queue_size = _inference_queue.qsize() if _inference_queue is not None else None
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "model_loaded": _model is not None,
        "queue_size": queue_size,
    }


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/info")
async def info():
    return {
        "model": MODEL_NAME,
        "device": DEVICE,
        "max_batch_size": MAX_BATCH,
        "max_seq_len": MAX_SEQ_LEN,
        "queue_batch_requests": QUEUE_BATCH_REQUESTS,
        "queue_batch_wait_ms": QUEUE_BATCH_WAIT_MS,
        "max_queue_size": MAX_QUEUE_SIZE,
        "dense_dim": 1024,
        "supports_sparse": True,
        "supports_colbert": True,
    }


@app.post("/embed", response_model=EmbeddingResponse)
async def embed(req: EmbeddingRequest, request: Request):
    t0 = time.time()
    mode = req.mode

    REQUEST_COUNT.labels(mode).inc()
    BATCH_SIZE_H.observe(len(req.texts))

    try:
        batch_size = min(req.batch_size or MAX_BATCH, MAX_BATCH)
        max_length = min(req.max_seq_len or MAX_SEQ_LEN, MAX_SEQ_LEN)

        queue = _queue_or_raise()
        if queue.full():
            raise HTTPException(status_code=503, detail="Inference queue is full, retry later")

        task = InferenceTask(
            req=req,
            mode=mode,
            batch_size=batch_size,
            max_length=max_length,
            future=asyncio.get_running_loop().create_future(),
        )
        await queue.put(task)

        result = await asyncio.wait_for(task.future, timeout=INFER_TIMEOUT_SEC)
        embeddings = result["embeddings"]
        dense_dim = result["dense_dim"]

        elapsed_ms = (time.time() - t0) * 1000
        REQUEST_LAT.labels(mode=mode).observe(elapsed_ms / 1000)

        return EmbeddingResponse(
            model=MODEL_NAME,
            mode=mode,
            count=len(embeddings),
            dense_dim=dense_dim,
            embeddings=embeddings,
            elapsed_ms=round(elapsed_ms, 2),
        )

    except asyncio.TimeoutError:
        REQUEST_ERRORS.inc()
        raise HTTPException(status_code=504, detail="Inference timeout")
    except HTTPException:
        raise
    except Exception as e:
        REQUEST_ERRORS.inc()
        log.exception("Embedding failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    REQUEST_ERRORS.inc()
    log.exception(f"Unhandled error on {request.url}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})