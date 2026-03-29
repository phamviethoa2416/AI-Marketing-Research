from __future__ import annotations

import re
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    HRFlowable,
    NextPageTemplate,
)
from reportlab.platypus.tableofcontents import TableOfContents

# ─── Color palette ────────────────────────────────────────────────────────────

NAVY     = colors.HexColor("#1B2B4B")
BLUE     = colors.HexColor("#2563EB")
ACCENT   = colors.HexColor("#0EA5E9")
GRAY     = colors.HexColor("#64748B")
LIGHTGRAY= colors.HexColor("#F1F5F9")
DIVIDER  = colors.HexColor("#E2E8F0")
WHITE    = colors.white
BLACK    = colors.black

PAGE_W, PAGE_H = A4
MARGIN_L = 2.5 * cm
MARGIN_R = 2.5 * cm
MARGIN_T = 2.5 * cm
MARGIN_B = 2.0 * cm

# ─── Vietnamese font registration ─────────────────────────────────────────────

FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_ITALIC = "Helvetica-Oblique"
FONT_BOLD_ITALIC = "Helvetica-BoldOblique"

def _register_vietnamese_fonts():
    global FONT_REGULAR, FONT_BOLD, FONT_ITALIC, FONT_BOLD_ITALIC
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import os

        font_dir = os.getenv("FONT_DIR", "/usr/share/fonts/truetype/dejavu")
        font_map = {
            "DejaVuSans":            "DejaVuSans.ttf",
            "DejaVuSans-Bold":       "DejaVuSans-Bold.ttf",
            "DejaVuSans-Oblique":    "DejaVuSans-Oblique.ttf",
            "DejaVuSans-BoldOblique": "DejaVuSans-BoldOblique.ttf",
        }

        registered = 0
        for name, filename in font_map.items():
            path = os.path.join(font_dir, filename)
            if os.path.exists(path):
                pdfmetrics.registerFont(TTFont(name, path))
                registered += 1

        if registered == len(font_map):
            FONT_REGULAR = "DejaVuSans"
            FONT_BOLD = "DejaVuSans-Bold"
            FONT_ITALIC = "DejaVuSans-Oblique"
            FONT_BOLD_ITALIC = "DejaVuSans-BoldOblique"
    except Exception:
        pass  # Fall back to Helvetica

_register_vietnamese_fonts()


# ─── Style sheet ──────────────────────────────────────────────────────────────

def build_styles() -> dict:
    base = getSampleStyleSheet()

    custom: dict[str, ParagraphStyle] = {
        "title": ParagraphStyle(
            "ReportTitle",
            fontName=FONT_BOLD,
            fontSize=26,
            leading=32,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            fontName=FONT_REGULAR,
            fontSize=13,
            leading=18,
            textColor=GRAY,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "meta": ParagraphStyle(
            "Meta",
            fontName=FONT_REGULAR,
            fontSize=10,
            textColor=GRAY,
            alignment=TA_CENTER,
        ),
        "h1": ParagraphStyle(
            "H1",
            fontName=FONT_BOLD,
            fontSize=16,
            leading=22,
            textColor=NAVY,
            spaceBefore=18,
            spaceAfter=8,
            borderPad=0,
        ),
        "h2": ParagraphStyle(
            "H2",
            fontName=FONT_BOLD,
            fontSize=13,
            leading=18,
            textColor=BLUE,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "H3",
            fontName=FONT_BOLD_ITALIC,
            fontSize=11,
            leading=16,
            textColor=NAVY,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            fontName=FONT_REGULAR,
            fontSize=10,
            leading=15,
            textColor=BLACK,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            fontName=FONT_REGULAR,
            fontSize=10,
            leading=14,
            textColor=BLACK,
            leftIndent=16,
            bulletIndent=0,
            spaceAfter=3,
        ),
        "source": ParagraphStyle(
            "Source",
            fontName=FONT_ITALIC,
            fontSize=8.5,
            leading=12,
            textColor=GRAY,
            leftIndent=8,
            spaceAfter=2,
        ),
        "caption": ParagraphStyle(
            "Caption",
            fontName=FONT_ITALIC,
            fontSize=9,
            leading=12,
            textColor=GRAY,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
    }
    return custom


# ─── Page templates ───────────────────────────────────────────────────────────

class ReportDocument(BaseDocTemplate):
    def __init__(self, filename: str, report_title: str, **kwargs):
        self.report_title = report_title[:70]
        self._page_num = 0

        super().__init__(filename, pagesize=A4, **kwargs)

        # Cover page: no header/footer
        cover_frame = Frame(
            MARGIN_L, MARGIN_B,
            PAGE_W - MARGIN_L - MARGIN_R,
            PAGE_H - MARGIN_T - MARGIN_B,
            id="cover",
        )

        # Content pages: with header + footer
        content_frame = Frame(
            MARGIN_L, MARGIN_B + 1.2 * cm,
            PAGE_W - MARGIN_L - MARGIN_R,
            PAGE_H - MARGIN_T - MARGIN_B - 1.8 * cm,
            id="content",
        )

        self.addPageTemplates([
            PageTemplate(id="Cover",   frames=[cover_frame]),
            PageTemplate(id="Content", frames=[content_frame],
                         onPage=self._draw_content_page),
        ])

    def _draw_content_page(self, canvas, doc):
        canvas.saveState()
        page_w, page_h = A4

        # ── Header line ──
        canvas.setStrokeColor(DIVIDER)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN_L, page_h - MARGIN_T + 3, page_w - MARGIN_R, page_h - MARGIN_T + 3)

        canvas.setFillColor(GRAY)
        canvas.setFont(FONT_REGULAR, 8)
        canvas.drawString(MARGIN_L, page_h - MARGIN_T + 5, self.report_title)
        canvas.drawRightString(page_w - MARGIN_R, page_h - MARGIN_T + 5,
                               datetime.now().strftime("%d/%m/%Y"))

        # ── Footer ──
        canvas.line(MARGIN_L, MARGIN_B + 0.8 * cm, page_w - MARGIN_R, MARGIN_B + 0.8 * cm)
        canvas.drawCentredString(page_w / 2, MARGIN_B + 3 * mm,
                                 f"Trang {doc.page}")
        canvas.setFillColor(BLUE)
        canvas.drawString(MARGIN_L, MARGIN_B + 3 * mm, "Multi-Agent Research System")

        canvas.restoreState()

    def afterFlowable(self, flowable):
        """Register headings for TOC."""
        if hasattr(flowable, "style"):
            style = flowable.style
            if style.name == "H1":
                self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))
            elif style.name == "H2":
                self.notify("TOCEntry", (1, flowable.getPlainText(), self.page))


# ─── Markdown → Platypus converter ───────────────────────────────────────────

def markdown_to_story(md: str, styles: dict) -> list:
    story: list = []
    lines = md.split("\n")
    i = 0
    bullet_buffer: list[str] = []

    def flush_bullets():
        nonlocal bullet_buffer
        for b in bullet_buffer:
            story.append(Paragraph(f"• {b}", styles["bullet"]))
        bullet_buffer = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ── Headings ────────────────────────────────────────────────────────────
        if stripped.startswith("### "):
            flush_bullets()
            text = _md_inline(stripped[4:])
            story.append(Paragraph(text, styles["h3"]))

        elif stripped.startswith("## "):
            flush_bullets()
            text = _md_inline(stripped[3:])
            story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER,
                                    spaceAfter=4, spaceBefore=8))
            story.append(Paragraph(text, styles["h2"]))

        elif stripped.startswith("# "):
            flush_bullets()
            text = _md_inline(stripped[2:])
            story.append(Paragraph(text, styles["h1"]))

        # ── Bullets ────────────────────────────────────────────────────────────
        elif stripped.startswith("- ") or stripped.startswith("* "):
            bullet_text = _md_inline(stripped[2:])
            bullet_buffer.append(bullet_text)

        # ── Horizontal rule ────────────────────────────────────────────────────
        elif stripped in ("---", "***", "___"):
            flush_bullets()
            story.append(HRFlowable(width="100%", thickness=1, color=DIVIDER,
                                    spaceBefore=6, spaceAfter=6))

        # ── Empty line = paragraph break ────────────────────────────────────────
        elif not stripped:
            flush_bullets()
            story.append(Spacer(1, 4))

        # ── Normal paragraph ────────────────────────────────────────────────────
        else:
            flush_bullets()
            text = _md_inline(stripped)
            story.append(Paragraph(text, styles["body"]))

        i += 1

    flush_bullets()
    return story


def _md_inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Italic
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    # Inline code
    text = re.sub(r"`(.+?)`", r'<font name="Courier">\1</font>', text)
    # Escape raw & and < that aren't tags
    return text


# ─── Sources table ────────────────────────────────────────────────────────────

def build_sources_table(sources: list[dict], styles: dict) -> list:
    """Build a formatted table of sources."""
    if not sources:
        return []

    story: list = [
        Paragraph("Tài liệu tham khảo (References)", styles["h1"]),
        Spacer(1, 4),
    ]

    for i, src in enumerate(sources, 1):
        source_type = src.get("source_type", "web")
        icon = "🌐" if source_type == "web" else "📄"
        title = src.get("title", "Không có tiêu đề")
        url = src.get("url", "")
        snippet = src.get("snippet", "")[:150]
        score = src.get("relevance_score", 0)

        text = f'<b>[{i}] {title}</b>'
        if url:
            text += f'<br/><font color="#2563EB">{url[:80]}</font>'
        if snippet:
            text += f'<br/><i>{snippet}...</i>'
        text += f'  <font color="#64748B">(relevance / độ liên quan: {score:.2f})</font>'

        story.append(Paragraph(text, styles["source"]))
        story.append(Spacer(1, 3))

    return story


# ─── Main formatter ───────────────────────────────────────────────────────────

class PDFFormatter:
    def generate(self, pipeline_result: dict, output_path: str) -> str:
        """Returns output_path on success, raises on error."""
        styles = build_styles()

        title     = pipeline_result.get("report_title", "Báo cáo Nghiên cứu")
        query     = pipeline_result.get("query", "")
        content   = pipeline_result.get("markdown_content", "")
        sources   = pipeline_result.get("sources", [])
        quality   = pipeline_result.get("quality", {})
        report_id = pipeline_result.get("report_id", "")

        doc = ReportDocument(
            filename=output_path,
            report_title=title,
            leftMargin=MARGIN_L,
            rightMargin=MARGIN_R,
            topMargin=MARGIN_T,
            bottomMargin=MARGIN_B,
        )

        story: list = [Spacer(1, 4 * cm), Paragraph("RESEARCH REPORT / BÁO CÁO NGHIÊN CỨU", styles["subtitle"]),
                       Spacer(1, 0.5 * cm), Paragraph(title, styles["title"]), Spacer(1, 0.8 * cm),
                       HRFlowable(width="60%", thickness=2, color=BLUE,
                                  hAlign="CENTER", spaceBefore=4, spaceAfter=4), Spacer(1, 0.5 * cm)]

        # ── Cover page ─────────────────────────────────────────────────────────

        if query:
            story.append(Paragraph(
                f"<i>Research query / Câu hỏi nghiên cứu:</i><br/>{query}",
                styles["subtitle"],
            ))

        story.append(Spacer(1, 1.5 * cm))

        # Meta info table
        now = datetime.now().strftime("%d/%m/%Y %H:%M")
        word_count = quality.get("word_count", 0)
        source_count = quality.get("source_count", 0)
        total_ms = quality.get("total_latency_ms", 0)

        meta_data = [
            ["Date / Ngày tạo:",          now],
            ["Word count / Số từ:",        f"{word_count:,}"],
            ["Sources / Số nguồn:",        str(source_count)],
            ["Generation time / Thời gian:", f"{total_ms/1000:.1f}s"],
            ["Report ID:",                 report_id[:16] + "..."],
            ["Language / Ngôn ngữ:",       "Bilingual — VI / EN"],
        ]
        meta_table = Table(meta_data, colWidths=[4 * cm, 10 * cm])
        meta_table.setStyle(TableStyle([
            ("FONTNAME",    (0, 0), (-1, -1), FONT_REGULAR),
            ("FONTSIZE",    (0, 0), (-1, -1), 10),
            ("FONTNAME",    (0, 0), (0, -1), FONT_BOLD),
            ("TEXTCOLOR",   (0, 0), (0, -1), NAVY),
            ("TEXTCOLOR",   (1, 0), (1, -1), GRAY),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, LIGHTGRAY]),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING",   (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 1 * cm))
        story.append(Paragraph(
            "Được tạo bởi <b>Multi-Agent Research System</b>",
            styles["meta"],
        ))

        # ── Switch to content template ──────────────────────────────────────────
        story.append(NextPageTemplate("Content"))
        story.append(PageBreak())

        # ── Table of Contents ──────────────────────────────────────────────────
        toc = TableOfContents()
        toc.levelStyles = [
            ParagraphStyle("TOC1", fontName=FONT_BOLD, fontSize=11,
                           leftIndent=0, textColor=NAVY, spaceAfter=3),
            ParagraphStyle("TOC2", fontName=FONT_REGULAR, fontSize=10,
                           leftIndent=16, textColor=GRAY, spaceAfter=2),
        ]
        story.append(Paragraph("Mục lục", styles["h1"]))
        story.append(toc)
        story.append(PageBreak())

        # ── Main content ────────────────────────────────────────────────────────
        content_story = markdown_to_story(content, styles)
        story.extend(content_story)

        # ── Sources ────────────────────────────────────────────────────────────
        story.append(PageBreak())
        sources_story = build_sources_table(sources, styles)
        story.extend(sources_story)

        # ── Quality appendix ───────────────────────────────────────────────────
        story.append(Spacer(1, 0.5 * cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER))
        story.append(Paragraph("Thông tin kỹ thuật", styles["h2"]))

        q_rows = [["Chỉ số", "Giá trị"]]
        for k, v in quality.items():
            if k != "step_timings_ms":
                q_rows.append([k.replace("_", " ").title(), str(v)])

        q_table = Table(q_rows, colWidths=[8 * cm, 8 * cm])
        q_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
            ("FONTNAME",   (0, 0), (-1, 0), FONT_BOLD),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHTGRAY]),
            ("GRID",       (0, 0), (-1, -1), 0.5, DIVIDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING",  (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ]))
        story.append(q_table)

        # ── Build ──────────────────────────────────────────────────────────────
        doc.multiBuild(story)
        return output_path