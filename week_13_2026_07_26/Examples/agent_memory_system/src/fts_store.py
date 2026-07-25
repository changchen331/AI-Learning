"""
Layer 4 关键词检索：SQLite FTS5 全文索引（BM25）

教学重点：
  1. 与 FAISS 语义检索互补——FTS5/BM25 擅长精确关键词命中
     （代码符号、ID、环境变量名、专有名词），向量擅长"措辞不同但语义相近"
  2. API 与 VectorStore 对齐：add_entries / rebuild_from_entries / search
     ——这样 MemoryFlusher 可以用同一套调用同步两套索引
  3. 优雅降级：FTS5 不可用时 available=False，search 返回 []，
     HybridRetriever 会自动退化为纯向量（对应 OpenClaw "扩展不可用就暴力计算"的可用性保证）

中文处理（关键）：
  SQLite FTS5 默认 unicode61 分词器把整段中文当作一个 token，
  导致 '咖啡' 无法命中标题 '用户爱喝美式咖啡'。本模块对中文逐字空格分词
  （'用户爱喝美式咖啡' → '用 户 爱 喝 美 式 咖 啡'），使每个汉字成为独立 token，
  查询 '咖啡' 被解析为 '咖' AND '啡'，召回含这两字的文档。
  代价是丢失词序/邻接精度（'咖啡牛奶' 与 '牛奶咖啡' 互相命中）——
  教学场景重召回，可接受；需要词序精度可换 ICU/jieba 分词器。

对齐 OpenClaw：
  - OpenClaw 用 chunks_fts(FTS5) + chunks_vec(sqlite-vec) 做 BM25+向量混合
  - 本项目向量走 FAISS（已有），FTS5 走本模块；混合在 retrieval.py 组合
  - BM25 归一化：SQLite bm25() 返回负值（越负越相关），取反后 min-max 归一到 [0,1]
    ⚠️ OpenClaw 文章写的 1/(1+bm25Score) 对 SQLite 负值会除零/负数，语义不通，
    故改用 min-max——权重 0.7/0.3 与并集语义严格照搬

使用方式：
  from src.fts_store import FTSStore
  fts = FTSStore()
  fts.add_entries([{"id":"entry_1","title":"...","content":"...","category":"fact"}])
  results = fts.search("关键词", top_k=3)

依赖：仅 stdlib sqlite3（需编译 FTS5，Python 3.9+ Windows 通常自带）
"""

import re
import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent / "outputs" / "sessions" / "memory.db"

# 匹配单个 CJK 统一表意文字（中日韩），用于逐字分词
_CJK_CHAR = re.compile(r"([一-鿿㐀-䶿])")


def _tokenize_zh(text: str) -> str:
    """
    把每个中文字符单独空格分隔，ASCII 词整体保留。
    '用户爱喝美式咖啡' → '用 户 爱 喝 美 式 咖 啤'
    '用 PostgreSQL 存数据' → '用 PostgreSQL 存 数 据'
    使 unicode61 能逐字索引中文，从而支持任意中文子串召回。
    """
    if not text:
        return ""
    spaced = _CJK_CHAR.sub(r" \1 ", text)
    return re.sub(r"\s+", " ", spaced).strip()


def _build_match_query(text: str) -> str:
    """
    把分词后的文本转成 FTS5 安全的 MATCH 表达式：
    每个 token 用双引号包裹、空格连接（FTS5 隐含 AND）。
    双引号使其成为 phrase，避免 'OR'/'NOT'/'*' 等被当成操作符。
    '"咖" "啡"' = 文档须同时含 '咖' 和 '啡'（不要求邻接）。
    """
    tokens = _tokenize_zh(text).split()
    return " ".join(f'"{t}"' for t in tokens) if tokens else ""


class FTSStore:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.available = False
        try:
            self._init_db()
            self.available = True
        except sqlite3.OperationalError as e:
            # FTS5 扩展未编译进来——禁用 BM25 这条腿，混合检索自动退化为纯向量
            logger.warning(f"FTS5 不可用，BM25 检索禁用（退化为纯向量）：{e}")
            self.available = False

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._connect()
        try:
            # 若存在旧版 memory_fts（缺 title_raw 列），drop 重建为新 schema
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(memory_fts)").fetchall()}
            if cols and "content_raw" not in cols:
                logger.info("检测到旧版 memory_fts schema，重建中（FTS 索引下次 flush/compaction 会补回）")
                conn.execute("DROP TABLE memory_fts")
            conn.executescript("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    entry_id UNINDEXED,
                    title,
                    content,
                    category UNINDEXED,
                    title_raw UNINDEXED,
                    content_raw UNINDEXED
                );
            """)
            conn.commit()
        finally:
            conn.close()

    # ── 写入（与 VectorStore 同名，便于 MemoryFlusher 统一调度）──────────────

    def add_entries(self, entries: list[dict]) -> int:
        """增量插入新条目。返回插入条数。"""
        if not self.available or not entries:
            return 0
        conn = self._connect()
        try:
            for e in entries:
                title = e.get("title", "")
                content = e.get("content", "")
                conn.execute(
                    "INSERT INTO memory_fts "
                    "(entry_id, title, content, category, title_raw, content_raw) "
                    "VALUES (?,?,?,?,?,?)",
                    (
                        str(e.get("id", "")),
                        _tokenize_zh(title), _tokenize_zh(content),
                        e.get("category", ""),
                        title, content,
                    ),
                )
            conn.commit()
            return len(entries)
        finally:
            conn.close()

    def rebuild_from_entries(self, entries: list[dict]):
        """清空后从全量条目重建（Compaction 后与 VectorStore.rebuild_from_entries 同步调用）"""
        if not self.available:
            return
        conn = self._connect()
        try:
            conn.execute("DELETE FROM memory_fts")
            for e in entries:
                title = e.get("title", "")
                content = e.get("content", "")
                conn.execute(
                    "INSERT INTO memory_fts "
                    "(entry_id, title, content, category, title_raw, content_raw) "
                    "VALUES (?,?,?,?,?,?)",
                    (
                        str(e.get("id", "")),
                        _tokenize_zh(title), _tokenize_zh(content),
                        e.get("category", ""),
                        title, content,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    # ── 查询 ──────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        BM25 关键词检索。返回 [{"id","title","content","category","score"}]，
        score 已 min-max 归一到 [0,1]（1 = 本批最相关）。
        """
        if not self.available:
            return []
        match_query = _build_match_query(query)
        if not match_query:
            return []
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT entry_id, title_raw, content_raw, category, "
                    "bm25(memory_fts) AS bm25 "
                    "FROM memory_fts WHERE memory_fts MATCH ? "
                    "ORDER BY bm25 LIMIT ?",
                    (match_query, top_k),
                ).fetchall()
            finally:
                conn.close()
        except sqlite3.OperationalError as e:
            # MATCH 语法对某些 query 仍可能抛错（罕见）——按降级处理
            logger.warning(f"FTS 查询失败（query={query!r}, match={match_query!r}）：{e}")
            return []

        if not rows:
            return []

        # bm25() 返回负值，越负越相关 → raw = -bm25（越大越好），再 min-max 归一
        raws = [-r["bm25"] for r in rows]
        lo, hi = min(raws), max(raws)
        span = hi - lo
        results = []
        for r, raw in zip(rows, raws):
            norm = 1.0 if span == 0 else (raw - lo) / span
            results.append({
                "id": r["entry_id"],
                "title": r["title_raw"],
                "content": r["content_raw"],
                "category": r["category"],
                "score": norm,
            })
        return results

    # ── 统计 ──────────────────────────────────────────────────────────────────

    @property
    def total_entries(self) -> int:
        if not self.available:
            return 0
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM memory_fts").fetchone()
            return row["cnt"]
        finally:
            conn.close()
