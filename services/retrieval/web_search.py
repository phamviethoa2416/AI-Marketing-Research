from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import httpx

log = logging.getLogger("web_search")

# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class WebResult:
    url: str
    title: str
    snippet: str
    full_content: str
    score: float
    published_date: Optional[str] = None
    domain: str = ""
    content_hash: str = ""
    fetch_success: bool = True
    word_count: int = 0

    def __post_init__(self):
        self.domain = urlparse(self.url).netloc
        combined = self.full_content or self.snippet
        self.content_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]
        self.word_count = len((self.full_content or self.snippet).split())


@dataclass
class WebSearchResult:
    query: str
    results: list[WebResult]
    tavily_answer: Optional[str]
    total_fetched: int
    elapsed_ms: float
    errors: list[str] = field(default_factory=list)

    @property
    def combined_context(self) -> str:
        parts: list[str] = []
        if self.tavily_answer:
            parts.append(f"**Tổng hợp trực tiếp:** {self.tavily_answer}")
        for r in self.results:
            content = r.full_content or r.snippet
            if content:
                parts.append(f"**[{r.title}]({r.url})**\n{content[:3000]}")
        return "\n\n---\n\n".join(parts)


class ContentExtractor:
    def __init__(self, max_chars: int = 20_000, timeout: float = 15.0):
        self.max_chars = max_chars
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; MAS-ResearchBot/1.0; "
                    "+https://github.com/phamviethoa2416/AI-Marketing-Research)"
                )
            },
            limits=httpx.Limits(max_connections=20),
        )

    def fetch_and_extract(self, url: str) -> tuple[str, bool]:
        try:
            response = self._client.get(url)
            if response.status_code != 200:
                return "", False

            content_type = response.headers.get("content-type", "")
            if "html" not in content_type.lower():
                # PDF, binary, etc — skip
                return "", False

            html = response.text
            return self._extract_markdown(html, url), True

        except httpx.TimeoutException:
            log.debug(f"Timeout fetching {url}")
            return "", False
        except Exception as e:
            log.debug(f"Fetch error {url}: {e}")
            return "", False

    def _extract_markdown(self, html: str, url: str = "") -> str:
        try:
            from readability import Document as ReadabilityDoc
            doc = ReadabilityDoc(html)
            main_html = doc.summary(html_partial=True)
            title = doc.title()
        except Exception:
            # Fall back to full HTML parsing
            main_html = html
            title = ""

        try:
            from markdownify import markdownify as md
            markdown = md(
                main_html,
                heading_style="ATX",
                bullets="-",
                strip=["script", "style", "nav", "footer", "header", "aside"],
            )
        except Exception:
            # Last resort: strip all tags
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(main_html, "html.parser")
            markdown = soup.get_text()

        markdown = self._clean_markdown(markdown)

        # Truncate
        if len(markdown) > self.max_chars:
            markdown = markdown[: self.max_chars] + "\n\n[... nội dung bị cắt bớt ...]"

        return markdown

    def _clean_markdown(self, text: str) -> str:
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        lines = [l for l in text.split("\n") if l.strip()]
        return "\n".join(lines).strip()

    def close(self):
        self._client.close()


# ─── Main web search pipeline ─────────────────────────────────────────────────

class WebSearchPipeline:
    def __init__(
        self,
        tavily_api_key: str,
        embedding_url: str = "",
        qdrant_url: str = "",
        qdrant_api_key: str = "",
        qdrant_collection: str = "web_results",
        max_results: int = 8,
        fetch_full_content: bool = True,
        max_content_chars: int = 20_000,
        search_depth: str = "advanced",
    ):
        self.api_key = tavily_api_key
        self.max_results = max_results
        self.fetch_full = fetch_full_content
        self.search_depth = search_depth
        self.collection = qdrant_collection

        self._tavily_client = httpx.Client(timeout=30.0)
        self._extractor = ContentExtractor(max_chars=max_content_chars) if fetch_full_content else None

        # Optional: index results into Qdrant
        self._qdrant = None
        self._embedder = None
        if embedding_url and qdrant_url:
            from qdrant_client import QdrantClient
            from services.ingestion.pipeline import EmbeddingClient
            self._qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)
            self._embedder = EmbeddingClient(embedding_url)
            self._ensure_collection()

    def search(
        self,
        query: str,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        include_answer: bool = True,
        fetch_content: bool | None = None,
    ) -> WebSearchResult:
        t0 = time.time()
        errors: list[str] = []
        should_fetch = self.fetch_full if fetch_content is None else fetch_content

        # ── 1. Tavily search ────────────────────────────────────────────────────
        try:
            tavily_results, tavily_answer = self._call_tavily(
                query, include_domains, exclude_domains, include_answer
            )
        except Exception as e:
            log.error(f"Tavily error: {e}")
            errors.append(str(e))
            return WebSearchResult(
                query=query,
                results=[],
                tavily_answer=None,
                total_fetched=0,
                elapsed_ms=round((time.time() - t0) * 1000, 1),
                errors=errors,
            )

        # ── 2. Fetch full page content ──────────────────────────────────────────
        results: list[WebResult] = []
        seen_hashes: set[str] = set()

        for item in tavily_results:
            url = item.get("url", "")
            title = item.get("title", "")
            snippet = item.get("content", "")
            score = item.get("score", 0.0)

            # Skip if URL already seen
            if url in {r.url for r in results}:
                continue

            full_content = snippet
            fetch_success = True

            if should_fetch and self._extractor:
                fetched, fetch_success = self._extractor.fetch_and_extract(url)
                if fetched and len(fetched) > len(snippet):
                    full_content = fetched

            web_result = WebResult(
                url=url,
                title=title,
                snippet=snippet,
                full_content=full_content,
                score=score,
                published_date=item.get("published_date"),
                fetch_success=fetch_success,
            )

            # Dedup by content hash
            if web_result.content_hash in seen_hashes:
                continue
            seen_hashes.add(web_result.content_hash)

            # Quality filter: skip very short content
            if web_result.word_count < 20:
                continue

            results.append(web_result)

        # ── 3. Optional: Index into Qdrant ─────────────────────────────────────
        if self._qdrant and self._embedder and results:
            try:
                self._index_results(query, results)
            except Exception as e:
                log.warning(f"Failed to index web results: {e}")
                errors.append(f"Index warning: {e}")

        elapsed_ms = round((time.time() - t0) * 1000, 1)
        log.info(
            f"Web search '{query[:50]}': "
            f"{len(results)} results in {elapsed_ms}ms "
            f"(fetch={'yes' if should_fetch else 'no'})"
        )

        return WebSearchResult(
            query=query,
            results=results,
            tavily_answer=tavily_answer,
            total_fetched=len(results),
            elapsed_ms=elapsed_ms,
            errors=errors,
        )

    def _call_tavily(
        self,
        query: str,
        include_domains: list[str] | None,
        exclude_domains: list[str] | None,
        include_answer: bool,
    ) -> tuple[list[dict], str | None]:
        payload: dict = {
            "api_key":       self.api_key,
            "query":         query,
            "max_results":   self.max_results,
            "search_depth":  self.search_depth,
            "include_answer": include_answer,
            "include_raw_content": False,
        }
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        r = self._tavily_client.post("https://api.tavily.com/search", json=payload)
        r.raise_for_status()
        data = r.json()

        return data.get("results", []), data.get("answer")

    def _index_results(self, query: str, results: list[WebResult]) -> None:
        """Embed and index web results into Qdrant for future RAG retrieval."""
        from qdrant_client.models import PointStruct, SparseVector

        texts = [r.full_content or r.snippet for r in results]
        embeddings = self._embedder.embed_batch(texts, mode="hybrid")

        points: list[PointStruct] = []
        for result, emb in zip(results, embeddings):
            vectors: dict = {}
            if "dense" in emb:
                vectors["dense"] = emb["dense"]
            if emb.get("sparse_indices"):
                vectors["sparse"] = SparseVector(
                    indices=emb["sparse_indices"],
                    values=emb["sparse_values"],
                )

            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors,
                payload={
                    "source":         "web",
                    "url":            result.url,
                    "title":          result.title,
                    "domain":         result.domain,
                    "text":           (result.full_content or result.snippet)[:8000],
                    "score":          result.score,
                    "content_hash":   result.content_hash,
                    "query":          query,
                    "published_date": result.published_date,
                },
            ))

        if points:
            self._qdrant.upsert(collection_name=self.collection, points=points)
            log.debug(f"Indexed {len(points)} web results")

    def _ensure_collection(self):
        from qdrant_client.models import VectorParams, SparseVectorParams, Distance
        collections = [c.name for c in self._qdrant.get_collections().collections]
        if self.collection not in collections:
            self._qdrant.create_collection(
                collection_name=self.collection,
                vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams()},
            )
            log.info(f"Created Qdrant collection: {self.collection}")

    def close(self):
        self._tavily_client.close()
        if self._extractor:
            self._extractor.close()
        if self._embedder:
            self._embedder.close()
        if self._qdrant:
            self._qdrant.close()