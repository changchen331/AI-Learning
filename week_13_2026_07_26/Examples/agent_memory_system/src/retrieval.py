"""
混合检索：FAISS 向量语义 + SQLite FTS5/BM25 关键词

教学重点：
  1. 对齐 OpenClaw memory_search 的混合策略——BM25 + 向量取并集，权重 0.7/0.3
  2. 两条腿互补：向量找"措辞不同但语义相近"，BM25 找"精确符号/ID/专名命中"
  3. 降级：FTS 不可用或无命中时，自动退化为纯向量结果，不报错

得分公式（对齐 OpenClaw 的并集 + 加权）：
  final = vec_score * 0.7 + bm25_score * 0.3
  - 两侧都命中：两项相加
  - 只命中一侧：另一侧记 0
  - 取并集后按 final 降序，截 top_k

使用方式：
  from src.retrieval import HybridRetriever
  retriever = HybridRetriever(vs, fts)
  results = retriever.search("用户问句", top_k=3)
  # 每条结果多一个 "source" 字段：vector | bm25 | both
"""

import logging

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, vs, fts, vec_weight: float = 0.7, bm25_weight: float = 0.3):
        self.vs = vs
        self.fts = fts
        self.vec_weight = vec_weight
        self.bm25_weight = bm25_weight

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        vec_results = self.vs.search(query, top_k=top_k)
        bm25_results = self.fts.search(query, top_k=top_k) if self.fts.available else []

        # 双侧都空
        if not vec_results and not bm25_results:
            return []

        # 降级：某一侧无结果时直接返回另一侧
        if not bm25_results:
            for r in vec_results:
                r["source"] = "vector"
            return vec_results[:top_k]
        if not vec_results:
            for r in bm25_results:
                r["source"] = "bm25"
            return bm25_results[:top_k]

        # 并集：按 id 合并两侧结果
        merged: dict[str, dict] = {}
        for r in vec_results:
            key = r.get("id") or r.get("title", "")
            merged[key] = {
                "id": r.get("id"),
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "category": r.get("category", ""),
                "vec_score": float(r.get("score", 0.0)),
                "bm25_score": 0.0,
            }
        for r in bm25_results:
            key = r.get("id") or r.get("title", "")
            if key in merged:
                merged[key]["bm25_score"] = float(r.get("score", 0.0))
            else:
                merged[key] = {
                    "id": r.get("id"),
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "category": r.get("category", ""),
                    "vec_score": 0.0,
                    "bm25_score": float(r.get("score", 0.0)),
                }

        # 加权打分 + 来源标注
        for item in merged.values():
            item["score"] = (
                    item["vec_score"] * self.vec_weight
                    + item["bm25_score"] * self.bm25_weight
            )
            if item["vec_score"] > 0 and item["bm25_score"] > 0:
                item["source"] = "both"
            elif item["vec_score"] > 0:
                item["source"] = "vector"
            else:
                item["source"] = "bm25"

        ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]
