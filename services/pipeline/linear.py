from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic

log = logging.getLogger("pipeline")

# ─── Timing utility ───────────────────────────────────────────────────────────

@dataclass
class StepTimer:
    name: str
    start: float = field(default_factory=time.time)
    end: float = 0.0

    def stop(self) -> "StepTimer":
        self.end = time.time()
        return self

    @property
    def elapsed_ms(self) -> float:
        return round((self.end - self.start) * 1000, 1)


# ─── Pipeline data classes ────────────────────────────────────────────────────

@dataclass
class PipelineSource:
    url: str
    title: str
    snippet: str
    source_type: str     # "web" | "document"
    relevance_score: float
    document_id: Optional[str] = None


@dataclass
class PipelineOutput:
    query: str
    report_title: str
    markdown_content: str
    sources: list[PipelineSource]
    tavily_answer: Optional[str]

    # Metrics
    step_timings: dict[str, float]   # step_name → elapsed_ms
    total_elapsed_ms: float
    llm_input_tokens: int
    llm_output_tokens: int
    rag_chunks_used: int
    web_results_used: int

    # Quality signals
    word_count: int = 0
    section_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.markdown_content.split())
        self.section_count = self.markdown_content.count("\n## ")

    @property
    def quality_summary(self) -> dict:
        return {
            "word_count":         self.word_count,
            "section_count":      self.section_count,
            "source_count":       len(self.sources),
            "rag_chunks":         self.rag_chunks_used,
            "web_results":        self.web_results_used,
            "llm_input_tokens":   self.llm_input_tokens,
            "llm_output_tokens":  self.llm_output_tokens,
            "total_latency_ms":   self.total_elapsed_ms,
            "step_timings_ms":    self.step_timings,
        }


# ─── Prompt templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a bilingual (Vietnamese/English) research analyst producing professional reports.

Your reports follow a strict bilingual structure:
- **Vietnamese (Tiếng Việt)** is the primary language for all main content, headings, and analysis.
- **English** appears as: section subtitles in parentheses, technical terms inline, and a dedicated English Executive Summary section.

Core principles:
- Always cite sources when stating specific facts (use [Source: title] inline)
- Clearly distinguish confirmed information from inference or analysis
- Never fabricate data — if information is absent from sources, say so explicitly
- Use precise technical vocabulary; provide Vietnamese translation for English terms on first use
- Format: clean Markdown, ## for h2, ### for h3

Bilingual structure template:
## Tóm tắt điều hành (Executive Summary)
[Vietnamese summary — 120-150 words]

### English Executive Summary
[English summary — 100-120 words, same content]

## [Vietnamese heading] ([English heading])
[Vietnamese content with English technical terms inline where appropriate]
"""


def build_report_prompt(
    query: str,
    web_context: str,
    rag_context: str,
    tavily_answer: str | None,
    language: str = "bilingual",
) -> str:
    sections: list[str] = [
        f"# Research Query / Yêu cầu nghiên cứu\n{query}"
    ]

    if tavily_answer:
        sections.append(
            f"# Direct Answer / Tóm tắt nhanh\n{tavily_answer}"
        )
    if web_context:
        sections.append(
            f"# Web Sources / Nguồn web\n{web_context[:8000]}"
        )
    if rag_context:
        sections.append(
            f"# Internal Documents / Tài liệu nội bộ (RAG)\n{rag_context[:4000]}"
        )

    context_block = "\n\n---\n\n".join(sections)

    return f"""{context_block}

---

# Report Writing Instructions / Hướng dẫn viết báo cáo

Produce a **bilingual research report** (Vietnamese primary, English secondary).

Required structure — follow this exactly:

## Tóm tắt điều hành (Executive Summary)
Viết 120-150 từ tiếng Việt tóm tắt toàn bộ báo cáo.

### English Executive Summary
Write 100-120 words summarising the same content in English.

## Bối cảnh & Định nghĩa (Background & Definitions)
Giải thích bối cảnh, các khái niệm và thuật ngữ quan trọng.
Provide English term in parentheses on first use: e.g. "học máy (machine learning)".

## Phân tích chi tiết (Detailed Analysis)
Trình bày các phát hiện chính, dữ liệu, xu hướng từ các nguồn.
Use ### subsections for sub-topics.

## Đánh giá & Nhận định (Assessment & Insights)
Phân tích sâu, so sánh quan điểm, rủi ro và cơ hội.

## Kết luận & Khuyến nghị (Conclusions & Recommendations)
Tóm tắt các điểm hành động cụ thể, ưu tiên theo thứ tự quan trọng.

## Tài liệu tham khảo (References)
List all sources used: [n] Title — URL

Target length: 900-1400 words total (Vietnamese + English combined).
Format: Markdown only. No HTML tags.
"""


def build_rag_context(chunks: list) -> str:
    if not chunks:
        return ""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.text if hasattr(chunk, "text") else str(chunk)
        doc_id = chunk.document_id if hasattr(chunk, "document_id") else "unknown"
        parts.append(f"[Tài liệu {i} | doc:{doc_id[:8]}]\n{text}")
    return "\n\n".join(parts)


def build_web_context(results: list) -> str:
    if not results:
        return ""
    parts: list[str] = []
    for r in results:
        content = (r.full_content or r.snippet)[:3000]
        parts.append(f"**{r.title}** ({r.url})\n{content}")
    return "\n\n---\n\n".join(parts)


# ─── Linear Pipeline ──────────────────────────────────────────────────────────

class LinearPipeline:
    def __init__(
        self,
        anthropic_api_key: str,
        web_searcher,           # WebSearchPipeline instance
        rag_searcher,           # HybridSearcher instance
        llm_model: str = "claude-sonnet-4-5",
        llm_max_tokens: int = 4096,
        llm_temperature: float = 0.3,
        rag_top_k: int = 6,
        web_max_results: int = 6,
        fetch_web_content: bool = True,
    ):
        self._llm = anthropic.Anthropic(api_key=anthropic_api_key)
        self._web = web_searcher
        self._rag = rag_searcher
        self.model = llm_model
        self.max_tokens = llm_max_tokens
        self.temperature = llm_temperature
        self.rag_top_k = rag_top_k
        self.web_max_results = web_max_results
        self.fetch_web = fetch_web_content

    def run(
        self,
        query: str,
        document_ids: list[str] | None = None,   # restrict RAG to specific docs
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> PipelineOutput:
        t_total = time.time()
        timings: dict[str, float] = {}

        log.info(f"Linear pipeline START: '{query[:80]}'")

        # ── Step 1: Web search ─────────────────────────────────────────────────
        t_web = StepTimer("web_search")
        web_result = self._web.search(
            query=query,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            include_answer=True,
            fetch_content=self.fetch_web,
        )
        timings["web_search_ms"] = t_web.stop().elapsed_ms
        log.info(f"  [web] {len(web_result.results)} results in {timings['web_search_ms']}ms")

        # ── Step 2: RAG retrieval ──────────────────────────────────────────────
        t_rag = StepTimer("rag_retrieval")
        try:
            rag_result = self._rag.search(
                query=query,
                top_k_initial=self.rag_top_k * 3,
                top_k_final=self.rag_top_k,
                document_ids=document_ids,
            )
            rag_chunks = rag_result.chunks
        except Exception as e:
            log.warning(f"RAG retrieval failed: {e} — continuing without RAG")
            rag_chunks = []
        timings["rag_retrieval_ms"] = t_rag.stop().elapsed_ms
        log.info(f"  [rag] {len(rag_chunks)} chunks in {timings['rag_retrieval_ms']}ms")

        # ── Step 3: Build context ──────────────────────────────────────────────
        web_ctx = build_web_context(web_result.results[:self.web_max_results])
        rag_ctx = build_rag_context(rag_chunks)

        # ── Step 4: LLM write ──────────────────────────────────────────────────
        t_llm = StepTimer("llm_write")
        user_prompt = build_report_prompt(
            query=query,
            web_context=web_ctx,
            rag_context=rag_ctx,
            tavily_answer=web_result.tavily_answer,
            language="bilingual",
        )

        try:
            response = self._llm.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            markdown_content = response.content[0].text
            input_tokens  = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
        except Exception as e:
            log.exception("LLM write failed")
            raise RuntimeError(f"LLM generation failed: {e}") from e

        timings["llm_write_ms"] = t_llm.stop().elapsed_ms
        log.info(f"  [llm] {output_tokens} tokens in {timings['llm_write_ms']}ms")

        # ── Step 5: Assemble output ────────────────────────────────────────────
        sources: list[PipelineSource] = []
        for r in web_result.results:
            sources.append(PipelineSource(
                url=r.url,
                title=r.title,
                snippet=r.snippet[:200],
                source_type="web",
                relevance_score=r.score,
            ))
        for chunk in rag_chunks:
            sources.append(PipelineSource(
                url="",
                title=f"Tài liệu nội bộ [{chunk.document_id[:8]}]",
                snippet=chunk.text[:200],
                source_type="document",
                relevance_score=chunk.final_score,
                document_id=chunk.document_id,
            ))

        report_title = query[:80]
        for line in markdown_content.split("\n"):
            if line.startswith("# ") and len(line) > 3:
                report_title = line[2:].strip()
                break

        total_ms = round((time.time() - t_total) * 1000, 1)
        timings["total_ms"] = total_ms

        output = PipelineOutput(
            query=query,
            report_title=report_title,
            markdown_content=markdown_content,
            sources=sources,
            tavily_answer=web_result.tavily_answer,
            step_timings=timings,
            total_elapsed_ms=total_ms,
            llm_input_tokens=input_tokens,
            llm_output_tokens=output_tokens,
            rag_chunks_used=len(rag_chunks),
            web_results_used=len(web_result.results),
        )

        log.info(
            f"Pipeline DONE: {output.word_count} words, "
            f"{output.section_count} sections, "
            f"{total_ms}ms total"
        )
        return output


# ─── Baseline Benchmarker ─────────────────────────────────────────────────────

@dataclass
class BaselineResult:
    query: str
    word_count: int
    section_count: int
    source_count: int
    rag_chunks: int
    web_results: int
    llm_input_tokens: int
    llm_output_tokens: int
    step_timings: dict[str, float]
    total_ms: float
    success: bool
    error: str = ""


class BaselineBenchmarker:
    def __init__(self, pipeline: LinearPipeline):
        self.pipeline = pipeline
        self.results: list[BaselineResult] = []

    def run(self, queries: list[str]) -> list[BaselineResult]:
        for query in queries:
            log.info(f"Benchmark: '{query[:60]}'")
            try:
                output = self.pipeline.run(query)
                result = BaselineResult(
                    query=query,
                    word_count=output.word_count,
                    section_count=output.section_count,
                    source_count=len(output.sources),
                    rag_chunks=output.rag_chunks_used,
                    web_results=output.web_results_used,
                    llm_input_tokens=output.llm_input_tokens,
                    llm_output_tokens=output.llm_output_tokens,
                    step_timings=output.step_timings,
                    total_ms=output.total_elapsed_ms,
                    success=True,
                )
            except Exception as e:
                log.error(f"Benchmark query failed: {e}")
                result = BaselineResult(
                    query=query,
                    word_count=0, section_count=0, source_count=0,
                    rag_chunks=0, web_results=0,
                    llm_input_tokens=0, llm_output_tokens=0,
                    step_timings={}, total_ms=0,
                    success=False, error=str(e),
                )
            self.results.append(result)
        return self.results

    def summary_table(self) -> str:
        header = (
            "| Query | Words | §§ | Sources | RAG | Web | "
            "LLM-in | LLM-out | Total(ms) | OK |\n"
            "|-------|-------|-----|---------|-----|-----|"
            "--------|---------|-----------|----|\n"
        )
        rows: list[str] = []
        for r in self.results:
            q = r.query[:40] + ("..." if len(r.query) > 40 else "")
            ok = "✓" if r.success else "✗"
            rows.append(
                f"| {q} | {r.word_count} | {r.section_count} | "
                f"{r.source_count} | {r.rag_chunks} | {r.web_results} | "
                f"{r.llm_input_tokens} | {r.llm_output_tokens} | "
                f"{int(r.total_ms)} | {ok} |"
            )

        if not self.results:
            return "No results yet."

        successful = [r for r in self.results if r.success]
        if successful:
            avg_words = sum(r.word_count for r in successful) / len(successful)
            avg_ms    = sum(r.total_ms for r in successful) / len(successful)
            summary   = (
                f"\n**Trung bình:** {avg_words:.0f} từ/báo cáo, "
                f"{avg_ms:.0f}ms/pipeline, "
                f"{len(successful)}/{len(self.results)} thành công"
            )
        else:
            summary = "\n**Không có kết quả thành công.**"

        return header + "\n".join(rows) + summary