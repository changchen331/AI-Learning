"""
rag_backend.py — A股年报 RAG 检索后端（三种方式共享的业务逻辑）

教学重点：
  1. 这一层是"纯业务逻辑"，不感知被哪种方式调用（Function Call / MCP / CLI 都复用它）
  2. 模块级加载 FAISS 索引一次，进程内所有调用复用（CLI 子进程每次启动加载一次）
  3. L2 归一化 + IndexFlatIP 内积 = 余弦相似度
  4. 元数据过滤（stock_code / year）在检索后做，过滤条件越多越大搜回数

使用方式（作为模块）：
  from src.rag_backend import search_annual_report, list_companies
  print(search_annual_report("宁德时代2023年营收", stock_code="300750", year="2023", top_k=3))

依赖：
  pip install faiss-cpu numpy openai
  向量数据位于 vectorstore/（运行 scripts/copy_data.py 复制）
  环境变量：DASHSCOPE_API_KEY（Embedding 用）

知识库说明：
  公司（stock_code）：贵州茅台(600519) / 五粮液(000858) / 宁德时代(300750)
                      海康威视(002415) / 中国平安(601318)
  年份：2021 / 2022 / 2023
  规模：15 份年报，共 10353 个语义分块
"""

import json
import os
import sys
from pathlib import Path

# Windows 上 torch 与 numpy 各自链接 OpenMP 会冲突，必须打开此开关
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from openai import OpenAI

# ── 常量 ──────────────────────────────────────────────────────────────────

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_MODEL = "text-embedding-v3"
EMBED_DIM = 1024

# 用 __file__ 定位项目根目录，无论从哪个工作目录启动都能找到 vectorstore/
BASE_DIR = Path(__file__).parent.parent
FAISS_INDEX_PATH = BASE_DIR / "vectorstore" / "faiss_index.bin"
FAISS_META_PATH = BASE_DIR / "vectorstore" / "faiss_meta.json"

# 公司信息表（用于 list_companies 和参数说明）
COMPANIES = [
    {"name": "贵州茅台", "stock_code": "600519", "years": ["2021", "2022", "2023"]},
    {"name": "五粮液", "stock_code": "000858", "years": ["2021", "2022", "2023"]},
    {"name": "宁德时代", "stock_code": "300750", "years": ["2021", "2022", "2023"]},
    {"name": "海康威视", "stock_code": "002415", "years": ["2021", "2022", "2023"]},
    {"name": "中国平安", "stock_code": "601318", "years": ["2021", "2022", "2023"]},
]

# ── 初始化（模块导入时执行一次）────────────────────────────────────────────

if not DASHSCOPE_API_KEY:
    print("错误：未设置环境变量 DASHSCOPE_API_KEY", file=sys.stderr)
    sys.exit(1)

_embed_client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

try:
    import faiss
except ImportError:
    print("错误：未安装 faiss-cpu，请运行 pip install faiss-cpu", file=sys.stderr)
    sys.exit(1)

if not FAISS_INDEX_PATH.exists() or not FAISS_META_PATH.exists():
    print(f"错误：向量索引文件不存在，请先运行 scripts/copy_data.py", file=sys.stderr)
    print(f"  期望路径：{FAISS_INDEX_PATH}", file=sys.stderr)
    sys.exit(1)

_index = faiss.read_index(str(FAISS_INDEX_PATH))
with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
    _meta_list: list[dict] = json.load(f)

print(
    f"[rag_backend] 就绪：{_index.ntotal} 个向量，{len(_meta_list)} 条元数据",
    file=sys.stderr,
)


# ── 辅助函数 ──────────────────────────────────────────────────────────────

def get_embedding(text: str) -> np.ndarray:
    """
    调用 DashScope Embedding API，返回 L2 归一化后的 float32 向量。
    FAISS 使用 IndexFlatIP（内积），预先 L2 归一化后内积等价于余弦相似度。
    """
    response = _embed_client.embeddings.create(model=EMBED_MODEL, input=[text])
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ── 对外接口 ──────────────────────────────────────────────────────────────

def search_annual_report(
        query: str,
        stock_code: str | None = None,
        year: str | None = None,
        top_k: int = 5,
) -> str:
    """
    在A股年报语料库中检索与问题最相关的段落。

    Args:
        query:      检索问题，自然语言，例如 "宁德时代2023年营收和净利润"
        stock_code: 可选，按公司过滤。600519(茅台)/000858(五粮液)/
                    300750(宁德时代)/002415(海康威视)/601318(中国平安)
        year:       可选，按年份过滤。"2021" / "2022" / "2023"
        top_k:      返回段落数，默认5，建议不超过10

    Returns:
        按相关度排序的段落列表，每段含来源（公司、年份、章节、页码）
    """
    try:
        query_vec = get_embedding(query)
    except Exception as e:
        return f"Embedding 调用失败：{e}"

    # 有过滤条件时多搜几倍，再过滤；无过滤时搜略多一点
    search_k = min(top_k * 10 if (stock_code or year) else top_k * 3, _index.ntotal)
    distances, indices = _index.search(query_vec.reshape(1, -1), search_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_meta_list):
            continue
        meta = _meta_list[idx]
        if stock_code and meta.get("stock_code") != stock_code:
            continue
        if year and str(meta.get("year")) != str(year):
            continue
        results.append({
            "score": float(dist),
            "content": meta.get("content", ""),
            "stock_code": meta.get("stock_code", ""),
            "year": str(meta.get("year", "")),
            "section": meta.get("section", ""),
            "page_num": meta.get("page_num", ""),
        })
        if len(results) >= top_k:
            break

    if not results:
        filter_parts = []
        if stock_code:
            filter_parts.append(f"股票代码={stock_code}")
        if year:
            filter_parts.append(f"年份={year}")
        filter_str = f"（过滤条件：{', '.join(filter_parts)}）" if filter_parts else ""
        return f"未找到相关内容{filter_str}，请尝试换一种问法或去掉过滤条件"

    lines = [f"检索到 {len(results)} 条相关段落：\n"]
    for i, r in enumerate(results, 1):
        company_name = next(
            (c["name"] for c in COMPANIES if c["stock_code"] == r["stock_code"]),
            r["stock_code"],
        )
        lines.append(
            f"【{i}】{company_name}（{r['stock_code']}）{r['year']}年报"
            f" | 第{r['page_num']}页 | 相关度：{r['score']:.3f}"
        )
        lines.append(f"章节：{r['section']}")
        lines.append(r["content"])
        lines.append("")

    return "\n".join(lines)


def list_companies() -> str:
    """
    列出年报知识库中包含的所有公司及可查询的年份范围。

    Returns:
        公司列表，含名称、股票代码、可查年份
    """
    lines = ["年报知识库收录公司列表：\n"]
    for c in COMPANIES:
        years_str = " / ".join(c["years"])
        lines.append(f"  {c['name']}  股票代码：{c['stock_code']}  年份：{years_str}")
    lines.append("\n共 5 家公司，每家 3 年，合计 15 份年报")
    return "\n".join(lines)


if __name__ == "__main__":
    # 自检：直接运行看检索结果
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("search")
    p1.add_argument("--query", required=True)
    p1.add_argument("--stock-code", default=None)
    p1.add_argument("--year", default=None)
    p1.add_argument("--top-k", type=int, default=5)
    sub.add_parser("list-companies")
    args = parser.parse_args()

    if args.cmd == "search":
        print(search_annual_report(args.query, args.stock_code, args.year, args.top_k))
    else:
        print(list_companies())
