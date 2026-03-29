from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import httpx
import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, SparseVector

log = logging.getLogger("ingestion")

@dataclass
class TextChunk:
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    page_num: int | None = None
    section: str | None = None
    token_count: int = 0
    chunk_hash: str = ""

    def __post_init__(self):
        self.chunk_hash = hashlib.sha256(self.text.encode()).hexdigest()[:16]

@dataclass
class IngestedDocument:
    document_id: str
    filename: str
    file_type: str
    raw_text: str
    page_count: int
    chunks: list[TextChunk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

@dataclass
class IngestionResult:
    document_id: str
    chunk_count: int
    indexed_count: int
    elapsed_sec: float
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.indexed_count > 0 and not self.errors

class PDFExtractor:
    def extract(self, path: Path) -> tuple[str, int]:
        import pdfplumber

        pages_text: list[str] = []
        with pdfplumber.open(str(path)) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                parts: list[str] = []

                txt = page.extract_text(x_tolerance=3, y_tolerance=3)
                if txt:
                    parts.append(txt.strip())

                tables = page.extract_tables()
                for table in tables:
                    rows = []
                    for row in table:
                        clean = [str(c or "").strip() for c in row]
                        rows.append(" | ".join(clean))
                    if rows:
                        parts.append("\n".join(rows))

                pages_text.append("\n".join(parts))

        full_text = "\n\n".join(p for p in pages_text if p.strip())
        return full_text, page_count

class DOCXExtractor:
    def extract(self, path: Path) -> tuple[str, int]:
        from docx import Document

        doc = Document(str(path))
        parts: list[str] = []
        page_count = 1

        for element in doc.element.body:
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

            if tag == "p":
                ns = element.nsmap.get('w', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main')
                runs = element.findall(f".//{{{ns}}}t")
                text = "".join(r.text or "" for r in runs)
                if text.strip():
                    parts.append(text.strip())

            if tag == "tbl":
                rows_text: list[str] = []
                for row in element.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tr"):
                    cells = []
                    for cell in row.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tc"):
                        t_elements = cell.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
                        cell_text = " ".join(t.text or "" for t in t_elements).strip()
                        cells.append(cell_text)
                    if cells:
                        rows_text.append(" | ".join(cells))
                if rows_text:
                    parts.append("\n".join(rows_text))

        try:
            breaks = doc.element.findall(
                ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}lastRenderedPageBreak")
            if breaks:
                page_count = len(breaks) + 1
        except Exception:
            pass

        return "\n\n".join(parts), page_count

class TXTExtractor:
    def extract(self, path: Path) -> tuple[str, int]:
        import chardet
        raw = path.read_bytes()
        enc = chardet.detect(raw).get("encoding") or "utf-8"
        text = raw.decode(enc, errors="replace")
        return text, 1

EXTRACTORS = {
    "pdf":  PDFExtractor(),
    "docx": DOCXExtractor(),
    "txt":  TXTExtractor(),
    "md":   TXTExtractor(),
}

def clean_text(text: str) -> str:
    # Normalize unicode whitespace
    text = re.sub(r"[\u00a0\u200b\u202f\u2060\ufeff]", " ", text)
    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove null bytes
    text = text.replace("\x00", "")
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing whitespace on each line
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()

class TokenChunker:
    def __init__(
            self,
            chunk_size: int = 512,
            overlap: int = 64,
            encoding_name: str = "cl100k_base",
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._enc = tiktoken.get_encoding(encoding_name)

    def _count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))

    def _split_paragraphs(self, text: str) -> list[str]:
        return [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    def chunk(self, text: str) -> Iterator[TextChunk]:
        paragraphs = self._split_paragraphs(text)
        buffer: list[str] = []
        buffer_tokens = 0
        char_offset = 0
        chunk_idx = 0
        para_offsets: list[int] = []

        # Pre-compute paragraph start offsets
        current = 0
        for p in paragraphs:
            idx = text.find(p, current)
            para_offsets.append(idx if idx >= 0 else current)
            current = (idx if idx >= 0 else current) + len(p)

        def flush(buf: list[str], start_char: int, end_char: int) -> TextChunk:
            nonlocal chunk_idx
            joined = "\n\n".join(buf)
            tc = TextChunk(
                text=joined,
                chunk_index=chunk_idx,
                start_char=start_char,
                end_char=end_char,
                token_count=self._count(joined),
            )
            chunk_idx += 1
            return tc

        buf_start_char = 0

        for para, p_start in zip(paragraphs, para_offsets):
            p_tokens = self._count(para)

            if p_tokens > self.chunk_size:
                if buffer:
                    yield flush(buffer, buf_start_char, p_start)
                    buffer = []
                    buffer_tokens = 0

                sentences = re.split(r"(?<=[.!?。！？])\s+", para)
                sent_buf: list[str] = []
                sent_tokens = 0
                sent_start = p_start

                for sent in sentences:
                    st = self._count(sent)
                    if sent_tokens + st > self.chunk_size and sent_buf:
                        yield flush(sent_buf, sent_start, p_start + len(para))
                        overlap_sents = []
                        overlap_t = 0
                        for s in reversed(sent_buf):
                            overlap_t += self._count(s)
                            overlap_sents.insert(0, s)
                            if overlap_t >= self.overlap:
                                break
                        sent_buf = overlap_sents + [sent]
                        sent_tokens = sum(self._count(s) for s in sent_buf)
                    else:
                        sent_buf.append(sent)
                        sent_tokens += st

                if sent_buf:
                    buf_start_char = p_start
                    buffer = sent_buf
                    buffer_tokens = sent_tokens
                continue

            if buffer_tokens + p_tokens > self.chunk_size and buffer:
                yield flush(buffer, buf_start_char, p_start)
                overlap_buf: list[str] = []
                overlap_t = 0
                for p_prev in reversed(buffer):
                    overlap_t += self._count(p_prev)
                    overlap_buf.insert(0, p_prev)
                    if overlap_t >= self.overlap:
                        break
                buffer = overlap_buf
                buffer_tokens = sum(self._count(p) for p in buffer)
                buf_start_char = p_start

            if not buffer:
                buf_start_char = p_start
            buffer.append(para)
            buffer_tokens += p_tokens

        if buffer:
            yield flush(buffer, buf_start_char, len(text))


class EmbeddingClient:
    def __init__(self, url: str, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def embed_batch(self, texts: list[str], mode: str = "hybrid") -> list[dict]:
        r = self._client.post(
            f"{self.url}/embed",
            json={"texts": texts, "mode": mode},
        )
        r.raise_for_status()
        return r.json()["embeddings"]

    def close(self):
        self._client.close()


def upsert_chunks(
        client: QdrantClient,
        collection: str,
        document_id: str,
        chunks: list[TextChunk],
        embeddings: list[dict],
        extra_payload: dict | None = None,
) -> int:
    points: list[PointStruct] = []

    for chunk, emb in zip(chunks, embeddings):
        payload = {
            "document_id": document_id,
            "chunk_index": chunk.chunk_index,
            "chunk_hash": chunk.chunk_hash,
            "text": chunk.text,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "token_count": chunk.token_count,
            "page_num": chunk.page_num,
            "section": chunk.section,
            **(extra_payload or {}),
        }

        vectors: dict = {}
        if "dense" in emb:
            vectors["dense"] = emb["dense"]

        sparse: SparseVector | None = None
        if "sparse_indices" in emb and emb["sparse_indices"]:
            sparse = SparseVector(
                indices=emb["sparse_indices"],
                values=emb["sparse_values"],
            )
            vectors["sparse"] = sparse

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors,
            payload=payload,
        )
        points.append(point)

    if not points:
        return 0

    batch_size = 50
    total = 0
    for i in range(0, len(points), batch_size):
        batch = points[i: i + batch_size]
        client.upsert(collection_name=collection, points=batch)
        total += len(batch)

    return total

class IngestionPipeline:
    def __init__(
            self,
            embedding_url: str,
            qdrant_url: str,
            qdrant_api_key: str = "",
            collection: str = "documents",
            chunk_size: int = 512,
            chunk_overlap: int = 64,
    ):
        self.embedding = EmbeddingClient(embedding_url)
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)
        self.collection = collection
        self.chunker = TokenChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self._ensure_collection()

    def _ensure_collection(self):
        from qdrant_client.models import VectorParams, SparseVectorParams, Distance
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection not in collections:
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams()},
            )
            log.info(f"Created Qdrant collection: {self.collection}")

    def ingest(
            self,
            file_path: Path,
            document_id: str,
            file_type: str | None = None,
            extra_payload: dict | None = None
    ) -> IngestionResult:
        t0 = time.time()
        errors: list[str] = []

        # ── 1. Detect file type ────────────────────────────────────────────────
        if file_type is None:
            suffix = file_path.suffix.lower().lstrip(".")
            file_type = suffix if suffix in EXTRACTORS else "txt"

        extractor = EXTRACTORS.get(file_type)
        if extractor is None:
            return IngestionResult(
                document_id=document_id,
                chunk_count=0,
                indexed_count=0,
                elapsed_sec=time.time() - t0,
                errors=[f"Unsupported file type: {file_type}"],
            )

        # ── 2. Extract text ────────────────────────────────────────────────────
        try:
            raw_text, page_count = extractor.extract(file_path)
            log.info(f"[{document_id}] Extracted {len(raw_text):,} chars, {page_count} pages")
        except Exception as e:
            log.exception(f"[{document_id}] Extraction failed")
            return IngestionResult(
                document_id=document_id,
                chunk_count=0,
                indexed_count=0,
                elapsed_sec=time.time() - t0,
                errors=[f"Extraction error: {e}"],
            )

        # ── 3. Clean ───────────────────────────────────────────────────────────
        clean = clean_text(raw_text)
        if len(clean) < 50:
            return IngestionResult(
                document_id=document_id,
                chunk_count=0,
                indexed_count=0,
                elapsed_sec=time.time() - t0,
                errors=["Extracted text too short — document may be image-only or corrupt"],
            )

        # ── 4. Chunk ───────────────────────────────────────────────────────────
        chunks = list(self.chunker.chunk(clean))
        log.info(f"[{document_id}] Created {len(chunks)} chunks")

        # ── 5. Embed in batches ────────────────────────────────────────────────
        EMBED_BATCH = 16
        all_embeddings: list[dict] = []
        try:
            for i in range(0, len(chunks), EMBED_BATCH):
                batch_texts = [c.text for c in chunks[i: i + EMBED_BATCH]]
                batch_emb = self.embedding.embed_batch(batch_texts, mode="hybrid")
                all_embeddings.extend(batch_emb)
                log.debug(f"[{document_id}] Embedded batch {i // EMBED_BATCH + 1}")
        except Exception as e:
            log.exception(f"[{document_id}] Embedding failed")
            errors.append(f"Embedding error: {e}")
            return IngestionResult(
                document_id=document_id,
                chunk_count=len(chunks),
                indexed_count=0,
                elapsed_sec=time.time() - t0,
                errors=errors,
            )

        # ── 6. Index in Qdrant ─────────────────────────────────────────────────
        payload = {
            "filename": file_path.name,
            "file_type": file_type,
            "page_count": page_count,
            **(extra_payload or {}),
        }
        try:
            indexed = upsert_chunks(
                self.qdrant,
                self.collection,
                document_id,
                chunks,
                all_embeddings,
                extra_payload=payload,
            )
            log.info(f"[{document_id}] Indexed {indexed} chunks in {time.time() - t0:.1f}s")
        except Exception as e:
            log.exception(f"[{document_id}] Qdrant upsert failed")
            errors.append(f"Index error: {e}")
            indexed = 0

        return IngestionResult(
            document_id=document_id,
            chunk_count=len(chunks),
            indexed_count=indexed,
            elapsed_sec=round(time.time() - t0, 2),
            errors=errors,
        )

    def delete_document(self, document_id: str) -> bool:
        from qdrant_client.models import Filter, FieldCondition, MatchValue, UpdateStatus
        result = self.qdrant.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))],
            ),
        )
        return result.status == UpdateStatus.COMPLETED if result else False
