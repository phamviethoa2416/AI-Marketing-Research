from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchAny,
    NamedVector,
    NamedSparseVector,
    SparseVector,
)

log = logging.getLogger("retrieval")


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    page_num: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def to_source_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "score": round(self.final_score, 4),
            "page_num": self.page_num,
        }

@dataclass
class SearchResult:
    query: str
    chunks: list[RetrievedChunk]
    total_retrieved: int
    elapsed_ms: float
    recall_at_k: Optional[dict[int, float]] = None


class EmbeddingClient:
    def __init__(self, url: str, timeout: float = 30.0):
        self._client = httpx.Client(timeout=timeout)
        self.url = url.rstrip("/")

    def embed_query(self, query: str) -> dict:
        r = self._client.post(
            f"{self.url}/embed",
            json={"texts": [query], "mode": "hybrid"},
        )
        r.raise_for_status()
        return r.json()["embeddings"][0]

    def close(self):
        self._client.close()


class RerankerClient:
    def __init__(self, url: str, timeout: float = 30.0):
        self._client = httpx.Client(timeout=timeout)
        self.url = url.rstrip("/")

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        pairs = [{"query": query, "passage": p} for p in passages]
        r = self._client.post(
            f"{self.url}/rerank",
            json={"pairs": pairs, "normalize": True},
        )
        r.raise_for_status()
        results = r.json()["results"]
        score_map = {item["index"]: item["normalized_score"] for item in results}
        return [score_map.get(i, 0.0) for i in range(len(passages))]

    def close(self):
        self._client.close()

def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    weights: list[float] | None = None,
    k: int = 60,
) -> dict[str, float]:
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return scores


class HybridSearcher:
    def __init__(
            self,
            qdrant_url: str,
            embedding_url: str,
            reranker_url: str,
            collection: str = "documents",
            qdrant_api_key: str = "",
    ):
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)
        self.embedder = EmbeddingClient(embedding_url)
        self.reranker = RerankerClient(reranker_url)
        self.collection = collection

    def search(
            self,
            query: str,
            top_k_initial: int = 20,
            top_k_final: int = 8,
            min_score: float = 0.0,
            document_ids: list[str] | None = None,
            alpha: float = 0.7,
    ) -> SearchResult:
        t0 = time.time()

        # ── 1. Embed query ─────────────────────────────────────────────────────
        emb = self.embedder.embed_query(query)
        dense_vec: list[float] = emb.get("dense", [])
        sparse_indices: list[int] = emb.get("sparse_indices", [])
        sparse_values: list[float] = emb.get("sparse_values", [])

        # ── 2. Build optional filter ────────────────────────────────────────────
        qdrant_filter: Filter | None = None
        if document_ids:
            qdrant_filter = Filter(
                must=[FieldCondition(
                    key="document_id",
                    match=MatchAny(any=document_ids),
                )]
            )

        # ── 3. Dense search ─────────────────────────────────────────────────────
        dense_hits = []
        if dense_vec:
            dense_hits = self.qdrant.search(
                collection_name=self.collection,
                query_vector=NamedVector(name="dense", vector=dense_vec),
                limit=top_k_initial,
                query_filter=qdrant_filter,
                with_payload=True,
                score_threshold=None,
            )

        # ── 4. Sparse search ────────────────────────────────────────────────────
        sparse_hits = []
        if sparse_indices:
            sparse_hits = self.qdrant.search(
                collection_name=self.collection,
                query_vector=NamedSparseVector(
                    name="sparse",
                    vector=SparseVector(indices=sparse_indices, values=sparse_values),
                ),
                limit=top_k_initial,
                query_filter=qdrant_filter,
                with_payload=True,
            )

        # ── 5. Build chunk map ─────────────────────────────────────────────────
        chunk_map: dict[str, RetrievedChunk] = {}

        for hit in dense_hits:
            cid = str(hit.id)
            payload = hit.payload or {}
            chunk_map[cid] = RetrievedChunk(
                chunk_id=cid,
                document_id=payload.get("document_id", ""),
                text=payload.get("text", ""),
                chunk_index=payload.get("chunk_index", 0),
                dense_score=hit.score,
                page_num=payload.get("page_num"),
                metadata={k: v for k, v in payload.items()
                          if k not in ("text", "document_id", "chunk_index", "page_num")},
            )

        for hit in sparse_hits:
            cid = str(hit.id)
            if cid in chunk_map:
                chunk_map[cid].sparse_score = hit.score
            else:
                payload = hit.payload or {}
                chunk_map[cid] = RetrievedChunk(
                    chunk_id=cid,
                    document_id=payload.get("document_id", ""),
                    text=payload.get("text", ""),
                    chunk_index=payload.get("chunk_index", 0),
                    sparse_score=hit.score,
                    page_num=payload.get("page_num"),
                    metadata={},
                )

        # ── 6. Reciprocal Rank Fusion ───────────────────────────────────────────
        dense_ranked = [str(h.id) for h in dense_hits]
        sparse_ranked = [str(h.id) for h in sparse_hits]

        rrf_scores = reciprocal_rank_fusion(
            [dense_ranked, sparse_ranked],
            weights=[alpha, 1 - alpha],
        )

        for cid, score in rrf_scores.items():
            if cid in chunk_map:
                chunk_map[cid].rrf_score = score

        # Sort by RRF, keep top_k_initial for reranking
        candidates = sorted(chunk_map.values(), key=lambda c: c.rrf_score, reverse=True)
        candidates = candidates[:top_k_initial]

        total_retrieved = len(candidates)

        if not candidates:
            return SearchResult(
                query=query,
                chunks=[],
                total_retrieved=0,
                elapsed_ms=round((time.time() - t0) * 1000, 1),
            )

        # ── 7. Rerank ───────────────────────────────────────────────────────────
        passages = [c.text for c in candidates]
        try:
            rerank_scores = self.reranker.rerank(query, passages)
            for chunk, score in zip(candidates, rerank_scores):
                chunk.rerank_score = score
                chunk.final_score = score
        except Exception as e:
            log.warning(f"Reranker failed, using RRF scores: {e}")
            for chunk in candidates:
                chunk.final_score = chunk.rrf_score

        # ── 8. Final sort & filter ─────────────────────────────────────────────
        final = sorted(candidates, key=lambda c: c.final_score, reverse=True)
        final = [c for c in final if c.final_score >= min_score]
        final = final[:top_k_final]

        elapsed_ms = round((time.time() - t0) * 1000, 1)
        log.info(
            f"Search '{query[:50]}': "
            f"{total_retrieved} retrieved → {len(final)} final "
            f"({elapsed_ms}ms)"
        )

        return SearchResult(
            query=query,
            chunks=final,
            total_retrieved=total_retrieved,
            elapsed_ms=elapsed_ms,
        )

    def close(self):
        self.embedder.close()
        self.reranker.close()

@dataclass
class EvalQuery:
    query: str
    relevant_chunk_ids: set[str]
    relevant_doc_ids: set[str] | None = None


@dataclass
class RecallReport:
    num_queries: int
    recall_at: dict[int, float]
    mrr: float
    avg_latency_ms: float
    per_query: list[dict]

    def summary(self) -> str:
        lines = [
            f"Queries: {self.num_queries}",
            f"MRR:     {self.mrr:.3f}",
            f"Latency: {self.avg_latency_ms:.0f}ms avg",
        ]
        for k, v in sorted(self.recall_at.items()):
            lines.append(f"Recall@{k}: {v:.3f}")
        return "\n".join(lines)


class RecallEvaluator:
    def __init__(self, searcher: HybridSearcher):
        self.searcher = searcher

    def evaluate(
            self,
            eval_queries: list[EvalQuery],
            k_values: list[int] | None = None,
            top_k_retrieve: int = 20,
            top_k_rerank: int = 10,
    ) -> RecallReport:
        k_values = k_values or [1, 3, 5, 10]
        max_k = max(k_values)

        recall_sums = {k: 0.0 for k in k_values}
        mrr_sum = 0.0
        latencies: list[float] = []
        per_query: list[dict] = []

        for eq in eval_queries:
            result = self.searcher.search(
                query=eq.query,
                top_k_initial=top_k_retrieve,
                top_k_final=max_k,
                min_score=0.0,
            )
            latencies.append(result.elapsed_ms)

            retrieved_ids = [c.chunk_id for c in result.chunks]
            retrieved_doc_ids = [c.document_id for c in result.chunks]

            for k in k_values:
                top_k_ids = set(retrieved_ids[:k])
                top_k_doc_ids = set(retrieved_doc_ids[:k])

                if eq.relevant_chunk_ids:
                    hits = len(eq.relevant_chunk_ids & top_k_ids)
                    recall = hits / len(eq.relevant_chunk_ids)
                elif eq.relevant_doc_ids:
                    hits = len(eq.relevant_doc_ids & top_k_doc_ids)
                    recall = hits / len(eq.relevant_doc_ids)
                else:
                    recall = 0.0

                recall_sums[k] += recall

            mrr = 0.0
            for rank, (cid, did) in enumerate(
                    zip(retrieved_ids, retrieved_doc_ids), start=1
            ):
                if cid in eq.relevant_chunk_ids or (
                        eq.relevant_doc_ids and did in eq.relevant_doc_ids
                ):
                    mrr = 1.0 / rank
                    break
            mrr_sum += mrr

            per_query.append({
                "query": eq.query,
                "mrr": round(mrr, 3),
                "retrieved": len(retrieved_ids),
                "latency_ms": result.elapsed_ms,
            })

        n = len(eval_queries)
        return RecallReport(
            num_queries=n,
            recall_at={k: round(s / n, 3) for k, s in recall_sums.items()},
            mrr=round(mrr_sum / n, 3),
            avg_latency_ms=round(sum(latencies) / len(latencies), 1) if latencies else 0,
            per_query=per_query,
        )