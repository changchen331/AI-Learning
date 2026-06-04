"""
下载中文 Wikipedia 文章，存为 jsonl 格式

数据来源：dumps.wikimedia.org（Wikimedia 官方 dump，国内可直连）
  文件：zhwiki-20251201-pages-articles1.xml-p1p187712.bz2（234MB，含 ~18万篇完整文章）
  备用：data/raw/*.txt 本地文件

教学重点：
  1. 流式解压+解析：BZ2Decompressor 分块解压 → XMLPullParser 增量解析
     内存中同时只存一篇文章，不受文件总大小限制
  2. Wikitext 清理：用正则去除模板/链接/标题标记，保留纯文本
  3. 数据格式：每行一个 JSON 对象，text 字段存清洗后的文章文本

使用方式：
  python download_data.py                        # 默认下载前 50000 篇
  python download_data.py --max_articles 10000   # 快速验证
  python download_data.py --from_local           # 用 data/raw/*.txt 本地文件

依赖：
  pip install requests（其余均为 Python 内置）
"""

import argparse
import bz2
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

DUMP_URL = (
    "https://dumps.wikimedia.org/zhwiki/20251201/"
    "zhwiki-20251201-pages-articles1.xml-p1p187712.bz2"
)
MEDIAWIKI_NS = "{http://www.mediawiki.org/xml/export-0.11/}"


# ── Wikitext 清理 ──────────────────────────────────────────────────────────────

def clean_wikitext(text: str) -> str:
    """
    去除 Wikipedia 标记语言中的格式标记，保留纯中文文本

    处理顺序：
      1. 去除 {{模板}} ：多轮去除，处理嵌套
      2. 去除 [[链接|显示文字]] → 保留显示文字
      3. 去除 [外链 显示] → 保留显示文字
      4. 去除 HTML 标签 <ref>...</ref>、<br/>等
      5. 去除 ==章节标题==
      6. 去除 '''粗体''' ''斜体''
      7. 去除表格（{|...|} 块）
      8. 清理多余空白
    """
    # 去 HTML 注释
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # 去 <ref>...</ref> 脚注
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*/>', '', text)
    # 去 {{模板}}（多轮展开，处理嵌套，最多 5 轮）
    for _ in range(5):
        prev = text
        text = re.sub(r'\{\{[^{}]*\}\}', '', text)
        if text == prev:
            break
    # 去表格
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
    # 去 [[文件:xxx]] [[File:xxx]] [[Image:xxx]] 等非正文链接
    text = re.sub(r'\[\[(?:文件|File|Image|Category|分类|Special|特殊):[^\]]+\]\]', '',
                  text, flags=re.IGNORECASE)
    # 展开 [[链接|显示文字]] → 显示文字
    text = re.sub(r'\[\[(?:[^|\]]+\|)?([^\]]+)\]\]', r'\1', text)
    # 去 [外链 显示] → 显示
    text = re.sub(r'\[https?://\S+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://\S+\]', '', text)
    # 去 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去 == 章节标题 ==
    text = re.sub(r'={2,}[^=\n]*={2,}', '\n', text)
    # 去 '''粗体''' ''斜体''
    text = re.sub(r"'{2,3}", '', text)
    # 去行首的表格分隔符（单独一行的 |、!、|-）
    text = re.sub(r'^\s*[|!].*$', '', text, flags=re.MULTILINE)
    # 清理多余空白
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


# ── 流式下载 + 解析 ──────────────────────────────────────────────────────────

def download_wiki_articles(max_articles: int = 50000) -> Path:
    """
    流式下载 bz2 压缩的 Wikipedia XML dump，边下载边解析

    流程：
      HTTP stream → 64KB chunks → BZ2Decompressor → XMLPullParser
                                   (分块解压)        (增量解析)
    内存中同时最多存一篇文章的 XML 元素，与文件总大小无关。
    """
    try:
        import requests
    except ImportError:
        raise ImportError("请先安装 requests：pip install requests")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "wiki_zh.jsonl"

    if out_path.exists():
        logger.info(f"数据文件已存在：{out_path}，跳过下载")
        return out_path

    logger.info(f"下载源：{DUMP_URL}")
    logger.info(f"目标：前 {max_articles} 篇文章 → {out_path}")
    logger.info("流式下载中（无需等待全部下载完成）...")

    decompressor = bz2.BZ2Decompressor()
    # XMLPullParser：只关注 end 事件，page 结束时再处理整个元素
    parser = ET.XMLPullParser(["end"])

    written = 0
    downloaded_bytes = 0
    chunk_size = 65536  # 64KB

    try:
        resp = requests.get(
            DUMP_URL,
            stream=True,
            headers={"User-Agent": "Mozilla/5.0 (educational project)"},
            timeout=60,
        )
        resp.raise_for_status()

        with open(out_path, "w", encoding="utf-8") as fout:
            for raw_chunk in resp.iter_content(chunk_size=chunk_size):
                if not raw_chunk:
                    continue

                # BZ2 解压此块
                try:
                    xml_chunk = decompressor.decompress(raw_chunk)
                except EOFError:
                    break  # BZ2 流正常结束

                if not xml_chunk:
                    continue

                downloaded_bytes += len(raw_chunk)
                parser.feed(xml_chunk)

                # 处理所有已完整解析的元素
                for event, elem in parser.read_events():
                    # 去命名空间前缀，方便判断
                    tag = elem.tag.replace(MEDIAWIKI_NS, "")

                    if tag != "page":
                        continue

                    # 只取主命名空间（ns=0），跳过 Talk/User/Template 等页
                    ns_elem = elem.find(f"{MEDIAWIKI_NS}ns")
                    if ns_elem is None or ns_elem.Data != "0":
                        elem.clear()
                        continue

                    title_elem = elem.find(f"{MEDIAWIKI_NS}title")
                    title = (title_elem.Data or "").strip() if title_elem is not None else ""

                    # 找 text 节点（在 revision 下）
                    text_elem = elem.find(f".//{MEDIAWIKI_NS}text")
                    raw_text = (text_elem.Data or "").strip() if text_elem is not None else ""

                    # 跳过重定向
                    if raw_text.upper().startswith("#REDIRECT") or \
                            raw_text.startswith("#重定向"):
                        elem.clear()
                        continue

                    # 清理 wikitext 标记
                    clean = clean_wikitext(raw_text)

                    # 过滤太短的文章（清洗后有效文本不足 50 字符）
                    if len(clean) < 50:
                        elem.clear()
                        continue

                    fout.write(json.dumps({
                        "id": str(written),
                        "title": title,
                        "text": clean,
                    }, ensure_ascii=False) + "\n")
                    written += 1

                    # 释放内存
                    elem.clear()

                    if written % 5000 == 0:
                        mb = downloaded_bytes / 1024 / 1024
                        logger.info(f"  已写入 {written} 篇，下载 {mb:.0f}MB...")

                    if written >= max_articles:
                        logger.info(f"已达到 {max_articles} 篇上限，停止下载")
                        resp.close()
                        break

                if written >= max_articles:
                    break

    except requests.RequestException as e:
        if out_path.exists():
            out_path.unlink()
        raise RuntimeError(
            f"网络下载失败：{e}\n\n"
            "备用方案：将中文 .txt 文件放入 data/raw/ 后运行：\n"
            "  python download_data.py --from_local"
        ) from e

    logger.info(f"完成！共写入 {written} 篇 → {out_path}")
    size_mb = out_path.stat().st_size / 1024 / 1024
    logger.info(f"文件大小：{size_mb:.1f} MB")
    return out_path


# ── 本地文件兜底 ──────────────────────────────────────────────────────────────

def from_local_txt(raw_dir: Path) -> Path:
    """
    从本地 .txt 文件构建 wiki_zh.jsonl（网络不通时的兜底）
    把任意中文 .txt 文件放到 data/raw/ 目录后运行。
    """
    txt_files = list(raw_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"未在 {raw_dir} 找到 .txt 文件。\n"
            "请将中文文本文件（任意来源）放入该目录后重试。"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "wiki_zh.jsonl"

    logger.info(f"从本地读取 {len(txt_files)} 个文件：{[f.name for f in txt_files]}")
    written = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for txt_path in txt_files:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as fin:
                buffer = []
                for line in fin:
                    line = line.strip()
                    if line:
                        buffer.append(line)
                    elif buffer:
                        text = "".join(buffer)
                        if len(text) >= 30:
                            fout.write(json.dumps({
                                "id": str(written),
                                "title": txt_path.stem,
                                "text": text,
                            }, ensure_ascii=False) + "\n")
                            written += 1
                        buffer = []
                if buffer:
                    text = "".join(buffer)
                    if len(text) >= 30:
                        fout.write(json.dumps({
                            "id": str(written),
                            "title": txt_path.stem,
                            "text": text,
                        }, ensure_ascii=False) + "\n")
                        written += 1

    logger.info(f"完成！从本地文件生成 {written} 段 → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_articles", type=int, default=50000)
    parser.add_argument("--from_local", action="store_true",
                        help="使用 data/raw/*.txt 本地文件，跳过网络下载")
    args = parser.parse_args()

    if args.from_local:
        raw_dir = DATA_DIR / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        from_local_txt(raw_dir)
    else:
        download_wiki_articles(args.max_articles)


if __name__ == "__main__":
    main()
