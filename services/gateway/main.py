from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field
from starlette.responses import Response

EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://embedding:8001")
RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:8002")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

HEALTH_TIMEOUT_SEC = float(os.getenv("HEALTH_TIMEOUT_SEC", "5"))
SEARCH_TIMEOUT_SEC = float(os.getenv("SEARCH_TIMEOUT_SEC", "15"))
EMBED_TIMEOUT_SEC = float(os.getenv("EMBED_TIMEOUT_SEC", "30"))
RERANK_TIMEOUT_SEC = float(os.getenv("RERANK_TIMEOUT_SEC", "30"))

OUTBOUND_RETRIES = int(os.getenv("OUTBOUND_RETRIES", "1"))
RETRY_BACKOFF_SEC = float(os.getenv("RETRY_BACKOFF_SEC", "0.2"))
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
log = logging.getLogger("gateway")

SEARCH_COUNT = Counter("gateway_search_total", "Tavily searches", ["status"])
SEARCH_LAT = Histogram(
    "gateway_search_latency_seconds",
    "Search latency",
    buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10],
)
EMBED_PROXY = Counter("gateway_embed_proxy_total", "Embed proxied")
RERANK_PROXY = Counter("gateway_rerank_proxy_total", "Rerank proxied")

_http_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    if _http_client is None:
        raise RuntimeError("HTTP client not initialized")
    return _http_client


def _request_id(request: Optional[Request]) -> str:
    if request is None:
        return "system"
    return request.headers.get("x-request-id") or f"gw-{int(time.time() * 1000)}"


def _client_ip(request: Optional[Request]) -> str:
    if request is None or request.client is None:
        return "unknown"
    return request.client.host or "unknown"


def _log_outbound(
        *,
        req_id: str,
        source: str,
        method: str,
        url: str,
        attempt: int,
        status_code: Optional[int],
        elapsed_ms: float,
        message: str,
        level: int = logging.INFO,
) -> None:
    status_text = str(status_code) if status_code is not None else "error"
    log.log(
        level,
        "[%s] %s method=%s url=%s attempt=%s status=%s elapsed_ms=%.1f %s",
        req_id,
        source,
        method,
        url,
        attempt,
        status_text,
        elapsed_ms,
        message,
    )


async def _send_with_retry(
        *,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        timeout: float,
        source: str,
        req_id: str,
        **kwargs: Any,
) -> httpx.Response:
    attempts = max(1, OUTBOUND_RETRIES + 1)
    for attempt in range(1, attempts + 1):
        t0 = time.time()
        try:
            response = await client.request(method, url, timeout=timeout, **kwargs)
            elapsed_ms = (time.time() - t0) * 1000
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < attempts:
                _log_outbound(
                    req_id=req_id,
                    source=source,
                    method=method,
                    url=url,
                    attempt=attempt,
                    status_code=response.status_code,
                    elapsed_ms=elapsed_ms,
                    message="retryable response, retrying",
                    level=logging.WARNING,
                )
                await asyncio.sleep(RETRY_BACKOFF_SEC * attempt)
                continue

            _log_outbound(
                req_id=req_id,
                source=source,
                method=method,
                url=url,
                attempt=attempt,
                status_code=response.status_code,
                elapsed_ms=elapsed_ms,
                message="completed",
                level=logging.INFO,
            )
            return response
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            elapsed_ms = (time.time() - t0) * 1000
            is_last_attempt = attempt == attempts
            _log_outbound(
                req_id=req_id,
                source=source,
                method=method,
                url=url,
                attempt=attempt,
                status_code=None,
                elapsed_ms=elapsed_ms,
                message=f"failed: {type(exc).__name__}",
                level=logging.ERROR if is_last_attempt else logging.WARNING,
            )
            if is_last_attempt:
                raise
            await asyncio.sleep(RETRY_BACKOFF_SEC * attempt)

    raise RuntimeError("Unexpected retry flow")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    max_results: int = Field(5, ge=1, le=20)
    search_depth: str = Field("advanced", pattern="^(basic|advanced)$")
    include_domains: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)
    include_answer: bool = Field(True)
    include_raw_content: bool = Field(False)


class SearchResult(BaseModel):
    title: str
    url: str
    content: str
    score: float
    published_date: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    answer: Optional[str]
    results: list[SearchResult]
    result_count: int
    elapsed_ms: float


class ServiceStatus(BaseModel):
    name: str
    url: str
    status: str  # ok | degraded | down
    latency_ms: Optional[float]
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    environment: str
    services: list[ServiceStatus]
    elapsed_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=5.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    log.info("Gateway started ✓")
    yield
    await _http_client.aclose()
    log.info("Gateway shutdown")


app = FastAPI(
    title="Multi-Agent System Gateway",
    description="Central API gateway for the MAS platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if ENVIRONMENT != "production" else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _check_service(
        client: httpx.AsyncClient, name: str, url: str
) -> ServiceStatus:
    t0 = time.time()
    try:
        r = await _send_with_retry(
            client=client,
            method="GET",
            url=f"{url}/health",
            timeout=HEALTH_TIMEOUT_SEC,
            source=f"health:{name}",
            req_id="health-check",
        )
        latency_ms = (time.time() - t0) * 1000
        if r.status_code == 200:
            return ServiceStatus(
                name=name, url=url, status="ok", latency_ms=round(latency_ms, 1)
            )
        return ServiceStatus(
            name=name,
            url=url,
            status="degraded",
            latency_ms=round(latency_ms, 1),
            detail=f"HTTP {r.status_code}",
        )
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        return ServiceStatus(
            name=name,
            url=url,
            status="down",
            latency_ms=round(latency_ms, 1),
            detail=str(e)[:100],
        )


@app.get("/health", response_model=HealthResponse)
async def health(client: httpx.AsyncClient = Depends(get_http_client)):
    t0 = time.time()
    services_to_check = [
        ("embedding", EMBEDDING_URL),
        ("reranker", RERANKER_URL),
        ("qdrant", QDRANT_URL),
    ]

    statuses = await asyncio.gather(
        *[_check_service(client, name, url) for name, url in services_to_check]
    )

    overall = "ok"
    for s in statuses:
        if s.status == "down":
            overall = "degraded"
            break
        if s.status == "degraded" and overall == "ok":
            overall = "degraded"

    return HealthResponse(
        status=overall,
        environment=ENVIRONMENT,
        services=list(statuses),
        elapsed_ms=round((time.time() - t0) * 1000, 1),
    )


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search", response_model=SearchResponse)
async def search(
        req: SearchRequest,
        request: Request,
        client: httpx.AsyncClient = Depends(get_http_client),
):
    if not TAVILY_API_KEY:
        raise HTTPException(status_code=503, detail="Search services not configured")

    t0 = time.time()
    req_id = _request_id(request)
    client_ip = _client_ip(request)
    try:
        payload: dict[str, Any] = {
            "api_key": TAVILY_API_KEY,
            "query": req.query,
            "max_results": req.max_results,
            "search_depth": req.search_depth,
            "include_answer": req.include_answer,
            "include_raw_content": req.include_raw_content,
        }
        if req.include_domains:
            payload["include_domains"] = req.include_domains
        if req.exclude_domains:
            payload["exclude_domains"] = req.exclude_domains

        log.info(
            "[%s] inbound path=/search client=%s query_len=%s max_results=%s",
            req_id,
            client_ip,
            len(req.query),
            req.max_results,
        )

        r = await _send_with_retry(
            client=client,
            method="POST",
            url="https://api.tavily.com/search",
            timeout=SEARCH_TIMEOUT_SEC,
            source="tavily:search",
            req_id=req_id,
            json=payload,
        )
        r.raise_for_status()
        data = r.json()

        results = [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", ""),
                score=item.get("score", 0.0),
                published_date=item.get("published_date"),
            )
            for item in data.get("results", [])
        ]

        elapsed_ms = (time.time() - t0) * 1000
        SEARCH_COUNT.labels(status="ok").inc()
        SEARCH_LAT.observe(elapsed_ms / 1000)

        return SearchResponse(
            query=req.query,
            answer=data.get("answer"),
            results=results,
            result_count=len(results),
            elapsed_ms=round(elapsed_ms, 2),
        )
    except httpx.HTTPStatusError as e:
        SEARCH_COUNT.labels(status="error").inc()
        log.error(f"Tavily error: {e.response.status_code} {e.response.text[:200]}")
        raise HTTPException(status_code=502, detail="Search provider error")
    except Exception as e:
        SEARCH_COUNT.labels(status="error").inc()
        log.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed")
async def proxy_embed(
        request: Request,
        client: httpx.AsyncClient = Depends(get_http_client),
):
    EMBED_PROXY.inc()
    body = await request.body()
    req_id = _request_id(request)
    client_ip = _client_ip(request)

    try:
        log.info(
            "[%s] inbound path=/embed client=%s body_bytes=%s",
            req_id,
            client_ip,
            len(body),
        )
        r = await _send_with_retry(
            client=client,
            method="POST",
            url=f"{EMBEDDING_URL}/embed",
            timeout=EMBED_TIMEOUT_SEC,
            source="proxy:embed",
            req_id=req_id,
            content=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            body = r.json()
        except Exception:
            body = {"detail": r.text[:500]}
        return JSONResponse(status_code=r.status_code, content=body)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/rerank")
async def proxy_rerank(
        request: Request,
        client: httpx.AsyncClient = Depends(get_http_client),
):
    RERANK_PROXY.inc()
    body = await request.body()
    req_id = _request_id(request)
    client_ip = _client_ip(request)
    try:
        log.info(
            "[%s] inbound path=/rerank client=%s body_bytes=%s",
            req_id,
            client_ip,
            len(body),
        )
        r = await _send_with_retry(
            client=client,
            method="POST",
            url=f"{RERANKER_URL}/rerank",
            timeout=RERANK_TIMEOUT_SEC,
            source="proxy:rerank",
            req_id=req_id,
            content=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            body = r.json()
        except Exception:
            body = {"detail": r.text[:500]}
        return JSONResponse(status_code=r.status_code, content=body)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    log.exception(f"Unhandled error: {request.url}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
