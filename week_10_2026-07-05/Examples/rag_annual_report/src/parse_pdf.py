"""
PDF 解析脚本：将原始年报 PDF 转换为结构化文本

教学重点（企业级 RAG 的真实挑战）：
  1. 数字 PDF vs 扫描件：处理方式完全不同
  2. 表格提取：年报里大量财务报表，直接按文字流提取会乱序
  3. 页眉/页脚噪声：每页都有公司名、页码，必须去除
  4. 章节识别：利用字体大小/加粗猜测标题层级
  5. 输出格式：保留元信息（页码、章节路径），供后续溯源用

依赖安装：
  pip install pdfplumber pymupdf pytesseract pillow
  # tesseract-ocr 需要单独安装并配置 PATH
  # Windows: https://github.com/UB-Mannheim/tesseract/wiki
"""

import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import pdfplumber          # 擅长表格提取
import fitz                # PyMuPDF，擅长文字+图片提取

# OCR 依赖可选（需要同时安装 pytesseract 包 + tesseract-ocr 二进制）
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR    = Path(__file__).parent.parent / "data" / "raw_pdf"
PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)

# 如果 tesseract 不在 PATH，手动指定（Windows 常见路径）
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class ParsedBlock:
    """
    一个解析块 = 年报里的一段连续内容（文字段落 or 表格）

    保留 page_num 和 section_path 非常重要——
    RAG 答案引用时能告诉用户"来自第38页，财务报告/资产负债表"
    """
    block_type:   str            # "text" | "table" | "title"
    content:      str            # 文字内容（表格转为 markdown）
    page_num:     int
    section_path: list[str]      # ["第三章 管理层讨论", "一、经营情况概述"]
    is_ocr:       bool = False   # 是否经过 OCR，质量可能有误
    raw_table:    Optional[list] = field(default=None, repr=False)  # 原始表格数据


# ── 工具函数 ──────────────────────────────────────────────────────────────────

# 年报里常见的章节标题模式（粗略匹配，不求完美）
CHAPTER_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百]+[章节]"),     # 第一章、第三节
    re.compile(r"^[一二三四五六七八九十]、"),               # 一、二、
    re.compile(r"^\d+\.\s"),                                # 1. 2.
]

NOISE_PATTERNS = [
    re.compile(r"^.{1,40}年度报告\s*$"),    # 页眉：公司名+年度报告
    re.compile(r"^\d+\s*$"),                # 独立页码
    re.compile(r"^—\s*\d+\s*—$"),          # — 38 —
]


def is_noise_line(line: str) -> bool:
    line = line.strip()
    if len(line) < 2:
        return True
    return any(p.match(line) for p in NOISE_PATTERNS)


def is_title_line(line: str, fontsize: Optional[float] = None, is_bold: bool = False) -> bool:
    """
    判断一行是否是标题。
    有字体信息时用字体大小，没有时用文字规律。
    """
    if fontsize and fontsize >= 14:
        return True
    if is_bold and len(line.strip()) < 50:
        return True
    return any(p.match(line.strip()) for p in CHAPTER_PATTERNS)


def table_to_markdown(table: list[list]) -> str:
    """把 pdfplumber 提取的表格转成 markdown 格式，方便 LLM 理解。"""
    if not table:
        return ""

    # 清洗单元格：None 变空字符串，去掉换行
    rows = []
    for row in table:
        cleaned = [str(cell or "").replace("\n", " ").strip() for cell in row]
        rows.append(cleaned)

    if not rows:
        return ""

    # 构建 markdown 表格
    header = rows[0]
    lines  = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[1:]:
        # 对齐列数（有些 PDF 表格行列不整齐）
        while len(row) < len(header):
            row.append("")
        lines.append("| " + " | ".join(row[:len(header)]) + " |")

    return "\n".join(lines)


def detect_if_scanned(page: fitz.Page, text: str) -> bool:
    """
    启发式判断：文字极少但图片多 → 很可能是扫描页。
    年报中扫描件多见于附件（审计报告原件）。
    """
    if len(text.strip()) > 50:
        return False
    image_list = page.get_images(full=True)
    return len(image_list) > 0


def ocr_page(page: fitz.Page, dpi: int = 200) -> str:
    """对扫描页做 OCR（中文）。需要 pytesseract + tesseract-ocr 二进制。"""
    if not OCR_AVAILABLE:
        return "[扫描页，OCR 不可用（未安装 pytesseract/tesseract），内容跳过]"
    try:
        mat  = fitz.Matrix(dpi / 72, dpi / 72)
        clip = page.rect
        pix  = page.get_pixmap(matrix=mat, clip=clip)
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang="chi_sim+eng")
        return text
    except Exception as e:
        logger.warning(f"  OCR 失败，跳过此页: {e}")
        return "[扫描页，OCR 失败，内容跳过]"


# ── 主解析逻辑 ────────────────────────────────────────────────────────────────

class AnnualReportParser:
    """
    年报 PDF 解析器。

    策略：
      - 用 pdfplumber 提取表格（它的表格算法更准）
      - 用 PyMuPDF (fitz) 提取带字体信息的文字（用于判断标题）
      - 对扫描页降级为 OCR
    """

    def __init__(self, pdf_path: Path, meta: dict = None):
        self.pdf_path = pdf_path
        self.meta     = meta or {}
        self.blocks: list[ParsedBlock] = []
        self._section_stack: list[str] = []

    def _update_section(self, title: str):
        """维护章节栈：根据缩进/编号层级推断层次。"""
        if re.match(r"^第[一二三四五六七八九十]+章", title):
            self._section_stack = [title]           # 顶级章
        elif re.match(r"^第[一二三四五六七八九十]+节", title):
            self._section_stack = self._section_stack[:1] + [title]  # 二级节
        elif re.match(r"^[一二三四五六七八九十]、", title):
            self._section_stack = self._section_stack[:2] + [title]  # 三级
        else:
            self._section_stack = self._section_stack[:3] + [title]

    def parse(self) -> list[ParsedBlock]:
        logger.info(f"开始解析: {self.pdf_path.name}")

        # 同时打开两个解析器
        plumber_doc = pdfplumber.open(self.pdf_path)
        fitz_doc    = fitz.open(str(self.pdf_path))

        for page_num in range(len(fitz_doc)):
            fitz_page   = fitz_doc[page_num]
            plumb_page  = plumber_doc.pages[page_num]

            # ── 1. 先用 PyMuPDF 获取带字体信息的文字 ──
            raw_text = fitz_page.get_text("text")
            is_scanned = detect_if_scanned(fitz_page, raw_text)

            if is_scanned:
                logger.debug(f"  第{page_num+1}页：检测到扫描件，启动 OCR")
                ocr_text = ocr_page(fitz_page)
                self.blocks.append(ParsedBlock(
                    block_type="text",
                    content=ocr_text,
                    page_num=page_num + 1,
                    section_path=list(self._section_stack),
                    is_ocr=True,
                ))
                continue

            # ── 2. 提取表格（用 pdfplumber）──
            table_bboxes = []
            for table in plumb_page.extract_tables():
                if table:
                    md = table_to_markdown(table)
                    if md:
                        self.blocks.append(ParsedBlock(
                            block_type="table",
                            content=md,
                            page_num=page_num + 1,
                            section_path=list(self._section_stack),
                            raw_table=table,
                        ))
            # 记录表格所在区域（后续跳过这些区域的文字提取）
            for table_obj in plumb_page.find_tables():
                table_bboxes.append(table_obj.bbox)

            # ── 3. 提取文字（跳过表格区域，逐行处理）──
            page_dict = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            current_para_lines = []

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:   # 0=文字，1=图片
                    continue

                for line in block.get("lines", []):
                    line_text = "".join(
                        span["text"] for span in line.get("spans", [])
                    ).strip()

                    if not line_text or is_noise_line(line_text):
                        continue

                    # 判断是否标题
                    spans    = line.get("spans", [])
                    fontsize = spans[0].get("size", 0) if spans else 0
                    is_bold  = any("Bold" in span.get("font", "") for span in spans)

                    if is_title_line(line_text, fontsize, is_bold):
                        # 先把积累的段落存起来
                        if current_para_lines:
                            self.blocks.append(ParsedBlock(
                                block_type="text",
                                content="\n".join(current_para_lines),
                                page_num=page_num + 1,
                                section_path=list(self._section_stack),
                            ))
                            current_para_lines = []

                        self._update_section(line_text)
                        self.blocks.append(ParsedBlock(
                            block_type="title",
                            content=line_text,
                            page_num=page_num + 1,
                            section_path=list(self._section_stack),
                        ))
                    else:
                        current_para_lines.append(line_text)

            # 最后一段
            if current_para_lines:
                self.blocks.append(ParsedBlock(
                    block_type="text",
                    content="\n".join(current_para_lines),
                    page_num=page_num + 1,
                    section_path=list(self._section_stack),
                ))

        plumber_doc.close()
        fitz_doc.close()

        logger.info(f"  解析完成: {len(self.blocks)} 个块")
        return self.blocks

    def save(self):
        """将解析结果保存为 JSON，保留所有元信息。"""
        stem     = self.pdf_path.stem
        out_path = PARSED_DIR / f"{stem}.json"

        output = {
            "meta":   self.meta,
            "source": str(self.pdf_path),
            "blocks": [asdict(b) for b in self.blocks],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"  已保存 → {out_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    manifest_path = RAW_DIR.parent / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        # 没有 manifest 就直接扫目录
        manifest = [
            {"filename": p.name, "stock_code": "", "year": ""}
            for p in RAW_DIR.glob("*.pdf")
        ]

    if not manifest:
        logger.error("没有找到任何 PDF，请先运行 download_reports.py")
        return

    for item in manifest:
        pdf_path = RAW_DIR / item["filename"]
        if not pdf_path.exists():
            logger.warning(f"文件不存在，跳过: {pdf_path}")
            continue

        parser = AnnualReportParser(pdf_path, meta=item)
        parser.parse()
        parser.save()

    logger.info(f"\n全部解析完成，结果在 {PARSED_DIR}")


if __name__ == "__main__":
    main()
