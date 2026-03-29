from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from FlagEmbedding import FlagReranker

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field, field_validator
from starlette.responses import Response

MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-reranker-v2-m3")
DEVICE = os.getenv("DEVICE", "cpu")
MAX_BATCH = int(os.getenv("MAX_BATCH_SIZE", "16"))
CACHE_DIR = os.getenv("CACHE_DIR", "/models")
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "256"))
QUEUE_BATCH_REQUESTS = int(os.getenv("QUEUE_BATCH_REQUESTS", "16"))
QUEUE_BATCH_WAIT_MS = int(os.getenv("QUEUE_BATCH_WAIT_MS", "10"))
INFER_TIMEOUT_SEC = float(os.getenv("INFER_TIMEOUT_SEC", "30"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("reranker")

REQUEST_COUNT  = Counter("reranker_requests_total",    "Total rerank requests")
REQUEST_ERRORS = Counter("reranker_errors_total",      "Total reranker errors")
REQUEST_LAT    = Histogram("reranker_latency_seconds", "Rerank latency",
                           buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 15])
PAIR_COUNT_H   = Histogram("reranker_pair_count",      "Pairs per request",
                           buckets=[1, 2, 4, 8, 16, 32, 64])
MODEL_LOADED   = Gauge("reranker_model_loaded",        "1 if model is loaded")

_model: FlagReranker | None = None
_inference_queue: asyncio.Queue[RerankTask] | None = None
_worker_task: asyncio.Task[None] | None = None


@dataclass
class RerankTask:
    req: RerankRequest
    batch_size: int
    future: asyncio.Future[dict[str, Any]]


def get_model() -> FlagReranker:
    if _model is None:
        raise RuntimeError("Model not loaded")
    return _model


def load_model() -> None:
    global _model
    from FlagEmbedding import FlagReranker
    log.info(f"Loading {MODEL_NAME} on {DEVICE}")
    t0 = time.time()
    use_fp16 = DEVICE != "cpu"
    _model = FlagReranker(
        MODEL_NAME,
        use_fp16=use_fp16,
        device=DEVICE,
        cache_dir=CACHE_DIR,
    )
    log.info(f"Reranker loaded in {time.time() - t0:.1f}s ✓")
    MODEL_LOADED.set(1)


def _queue_or_raise() -> asyncio.Queue[RerankTask]:
    if _inference_queue is None:
        raise RuntimeError("Inference queue is not initialized")
    return _inference_queue


def _safe_sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


async def _process_batch(batch: list[RerankTask]) -> None:
    if not batch:
        return

    grouped: dict[tuple, list[RerankTask]] = defaultdict(list)
    for task in batch:
        key = (task.batch_size, task.req.normalize)
        grouped[key].append(task)

    for (batch_size, normalize), group_tasks in grouped.items():
        await _process_group(group_tasks, batch_size, normalize)


async def _process_group(batch: list[RerankTask], batch_size: int, normalize: bool) -> None:
    if not batch:
        return

    all_pairs: list[tuple[str, str]] = []
    offsets: list[tuple[int, int]] = []

    for task in batch:
        start = len(all_pairs)
        all_pairs.extend([(p.query, p.passage) for p in task.req.pairs])
        offsets.append((start, len(all_pairs)))

    model = get_model()
    raw = await asyncio.to_thread(
        cast(Any, model).compute_score,
        all_pairs,
        batch_size=batch_size,
        normalize=False,
    )

    raw_scores: list[float] = raw if isinstance(raw, list) else [raw]

    for task, (start, end) in zip(batch, offsets):
        if task.future.cancelled():
            continue

        chunk_scores = raw_scores[start:end]
        results = []
        for i, score in enumerate(chunk_scores):
            results.append(RerankResult(
                index=i,
                score=float(score),
                normalized_score=round(_safe_sigmoid(score), 6) if normalize else None,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        task.future.set_result({"results": results})


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

            try:
                await _process_batch(batch)
            except Exception as exc:
                for task in batch:
                    if not task.future.done():
                        task.future.set_exception(exc)

            for _ in batch:
                queue.task_done()
    except asyncio.CancelledError:
        log.info("Inference worker stopped")
        raise


class RerankPair(BaseModel):
    query: str = Field(..., min_length=1)
    passage: str = Field(..., min_length=1)


class RerankRequest(BaseModel):
    pairs: list[RerankPair] = Field(..., min_length=1)
    normalize: bool = Field(True, description="Apply sigmoid to raw scores")
    batch_size: Optional[int] = Field(None, ge=1, le=64)

    @field_validator("pairs")
    @classmethod
    def validate_pairs(cls, v):
        if len(v) == 0:
            raise ValueError("pairs must not be empty")
        if len(v) > 256:
            raise ValueError("max 256 pairs per request")
        return v


class RerankResult(BaseModel):
    index: int
    score: float
    normalized_score: Optional[float] = None


class RerankResponse(BaseModel):
    model: str
    count: int
    results: list[RerankResult]
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
    log.info("Reranker service shutting down...")


app = FastAPI(
    title="Reranker Service",
    description="BGE-Reranker-v2-M3 cross-encoder reranking",
    version="1.0.0",
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
        "queue_batch_requests": QUEUE_BATCH_REQUESTS,
        "queue_batch_wait_ms": QUEUE_BATCH_WAIT_MS,
        "max_queue_size": MAX_QUEUE_SIZE,
        "supports_normalize": True,
    }


@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest, request: Request):
    t0 = time.time()

    REQUEST_COUNT.inc()
    PAIR_COUNT_H.observe(len(req.pairs))

    try:
        batch_size = min(req.batch_size or MAX_BATCH, MAX_BATCH)

        queue = _queue_or_raise()
        if queue.full():
            raise HTTPException(status_code=503, detail="Inference queue is full, retry later")

        task = RerankTask(
            req=req,
            batch_size=batch_size,
            future=asyncio.get_running_loop().create_future(),
        )
        await queue.put(task)

        result = await asyncio.wait_for(task.future, timeout=INFER_TIMEOUT_SEC)
        results = result["results"]

        elapsed_ms = (time.time() - t0) * 1000
        REQUEST_LAT.observe(elapsed_ms / 1000)

        return RerankResponse(
            model=MODEL_NAME,
            count=len(results),
            results=results,
            elapsed_ms=round(elapsed_ms, 2),
        )

    except asyncio.TimeoutError:
        REQUEST_ERRORS.inc()
        raise HTTPException(status_code=504, detail="Reranking timeout")
    except HTTPException:
        raise
    except Exception as e:
        REQUEST_ERRORS.inc()
        log.exception("Rerank failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    REQUEST_ERRORS.inc()
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
