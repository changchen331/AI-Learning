"""
GraphRAG 在线检索：Local Search 与 Global Search

教学重点（对应 PPT Part 4.4）：
  1. Local Search（局部检索）：适合「具体实体/关系」查询
     实体链接 → 邻居扩展（1~2 跳）→ 子图提取 → 拼上下文 → LLM 合成答案
     例：「宁德时代的子公司有哪些」「茅台董事长是谁」
  2. Global Search（全局检索）：适合「主题/趋势」查询
     社区摘要分发 → Map 并行为每个社区生成部分答案 → Reduce 聚合 → LLM 合成
     例：「各家公司研发投入对比」「白酒板块的整体情况」

两条路径都返回结构化的中间步骤，供 serve.py 的 /query/debug 逐步展示。

实体链接用 Embedding 相似度（而非字符串子串匹配）：因为抽取时实体名写法不一
（茅台/贵州茅台酒股份有限公司），embedding 能跨越字面差异捕捉同一实体。

使用方式：
  from retrieve import GraphRAGLocal, GraphRAGGlobal
  local = GraphRAGLocal()           # 加载 neo4j + 节点名向量
  res = local.search("宁德时代的子公司", debug=True)   # debug=True 返回各步骤

依赖：
  pip install neo4j numpy
  export DEEPSEEK_API_KEY=...  export DASHSCOPE_API_KEY=...
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import llm_chat, embed_query, embed_texts, EMBED_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
COMMUNITIES_PATH = BASE_DIR / "outputs" / "communities.json"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = None

LLM_MODEL = "deepseek-chat"  # deepseek-v4-flash

# Local Search 参数
LINK_TOPK = 3  # 实体链接取 top-3 候选实体
HOP = 2  # 邻居扩展跳数
MAX_SUBGRAPH = 40  # 子图最多保留节点数（控制喂给 LLM 的上下文长度）

# Global Search 参数
COMM_TOPK = 3  # 取最相关的 top-3 社区

ANSWER_SYSTEM = """你是基于知识图谱的问答助手。根据【参考资料】中的实体、关系、数据回答问题。

规则：
1. 只用参考资料中的事实，不得编造
2. 引用数据时标注来源（如[宁德时代年报]）
3. 资料不足时直接说明，不要猜"""


# ── 共用：节点名向量缓存（实体链接用）──────────────────────────────────────

class NodeNameIndex:
    """缓存所有实体节点的名字向量，供实体链接做余弦相似度查询。
    同一进程内只构建一次（serve.py lifespan 复用）。"""

    def __init__(self, driver):
        with driver.session() as s:
            rows = list(s.run("MATCH (n:Entity) RETURN n.uid AS uid, n.name AS name, n.type AS type"))
        self.uids = [r["uid"] for r in rows]
        self.names = [r["name"] for r in rows]
        self.types = [r["type"] for r in rows]
        if not self.uids:
            self.vecs = np.zeros((0, EMBED_DIM), dtype="float32")
            logger.warning("图库无节点，NodeNameIndex 为空")
            return
        logger.info(f"为 {len(self.names)} 个实体名编码向量...")
        self.vecs = embed_texts(self.names)
        logger.info("节点名向量就绪")

    def link(self, query_vec: np.ndarray, top_k: int = LINK_TOPK) -> list[dict]:
        """查询向量 → top-K 最相似的实体节点。"""
        if self.vecs.shape[0] == 0:
            return []
        sims = self.vecs @ query_vec.reshape(-1)  # 已归一化，内积即余弦；[N,1024]@[1024]=[N]
        top = np.argsort(-sims)[:top_k]
        return [{"uid": self.uids[i], "name": self.names[i], "type": self.types[i],
                 "score": float(sims[i])} for i in top]


# ── Local Search ──────────────────────────────────────────────────────────────

class GraphRAGLocal:
    """局部检索：实体链接 → 邻居扩展 → 子图 → 上下文 → LLM 合成。"""

    def __init__(self, uri=NEO4J_URI):
        self.driver = GraphDatabase.driver(uri, auth=NEO4J_AUTH)
        self.driver.verify_connectivity()
        self.node_index = NodeNameIndex(self.driver)

    def close(self):
        self.driver.close()

    def _expand_subgraph(self, linked_uids: list[str]) -> dict:
        """对每个链接到的实体，取邻居，返回子图节点+边（带截断）。
        截断策略：非 Indicator 邻居优先保留（子公司/人物/产品/地区信息量更高，
        Indicator 数量多占 649 节点的 63%，不优先会被挤出去），再用 Indicator 补足。
        始终保留 linked 节点本身。"""
        if not linked_uids:
            return {"nodes": [], "edges": []}
        linked_set = set(linked_uids)
        with self.driver.session() as s:
            # ① 直接邻居 + 其类型（一次性取回，Python 侧按类型排序截断）
            neighbors = list(s.run(
                "MATCH (n:Entity) WHERE n.uid IN $uids "
                "MATCH (n)-[]-(m:Entity) "
                "RETURN collect(DISTINCT [m.uid, m.type]) AS neighbors", uids=linked_uids
            ))[0]["neighbors"]
            # 非指标优先排序（Indicator 排后），保留核心实体类
            neighbors.sort(key=lambda x: 1 if x[1] == "Indicator" else 0)
            keep = list(linked_uids)
            for uid, _ in neighbors:
                if uid not in linked_set:
                    keep.append(uid)
                if len(keep) >= MAX_SUBGRAPH:
                    break
            keep_set = set(keep)
            # ② 节点 + 子图内边
            nodes = [dict(r) for r in s.run(
                "MATCH (n:Entity) WHERE n.uid IN $uids "
                "RETURN n.uid AS uid, n.name AS name, n.type AS type", uids=keep)]
            edges = [dict(r) for r in s.run(
                "MATCH (a:Entity)-[r]->(b:Entity) "
                "WHERE a.uid IN $uids AND b.uid IN $uids "
                "RETURN a.uid AS src, b.uid AS dst, a.name AS src_name, b.name AS dst_name, "
                "type(r) AS rel, r.year AS year, r.role AS role, r.ratio AS ratio",
                uids=keep)]
        return {"nodes": nodes, "edges": edges}

    def _build_context(self, subgraph: dict) -> str:
        """把子图格式化为喂给 LLM 的文本上下文。"""
        lines = ["[实体]"]
        for n in subgraph["nodes"]:
            lines.append(f"- {n['name']} ({n['type']})")
        lines.append("\n[关系]")
        for e in subgraph["edges"]:
            extra = []
            if e.get("year"):
                extra.append(str(e["year"]))
            if e.get("role"):
                extra.append(f"职务={e['role']}")
            if e.get("ratio"):
                extra.append(f"持股={e['ratio']}")
            tag = f" [{', '.join(extra)}]" if extra else ""
            lines.append(f"- ({e['src_name']}) -[{e['rel']}]-> ({e['dst_name']}){tag}")
        return "\n".join(lines)

    def search(self, question: str, debug: bool = False) -> dict:
        # ① 实体链接
        qvec = embed_query(question)
        linked = self.node_index.link(qvec, top_k=LINK_TOPK)
        linked_uids = [x["uid"] for x in linked]

        # ② 邻居扩展 + 子图提取
        subgraph = self._expand_subgraph(linked_uids)

        # ③ 构建上下文
        context = self._build_context(subgraph)

        # ④ LLM 合成答案
        user_msg = f"【参考资料】\n{context}\n\n【问题】\n{question}\n\n请根据资料回答。"
        answer = llm_chat(ANSWER_SYSTEM, user_msg, temperature=0.1, max_tokens=1024)

        result = {"answer": answer, "context": context,
                  "linked_entities": linked, "subgraph": subgraph}
        return result


# ── Global Search ─────────────────────────────────────────────────────────────

class GraphRAGGlobal:
    """全局检索：社区摘要分发 → Map → Reduce → LLM 合成。"""

    def __init__(self, uri=NEO4J_URI):
        self.driver = GraphDatabase.driver(uri, auth=NEO4J_AUTH)
        self.driver.verify_connectivity()
        if not COMMUNITIES_PATH.exists():
            raise FileNotFoundError(f"社区摘要不存在，请先运行 community_detect.py: {COMMUNITIES_PATH}")
        self.communities = json.load(open(COMMUNITIES_PATH, encoding="utf-8"))
        # 对每个社区摘要编码向量，供相关性检索
        texts = [c["summary"] for c in self.communities]
        logger.info(f"为 {len(texts)} 个社区摘要编码向量...")
        self.comm_vecs = embed_texts(texts)

    def close(self):
        self.driver.close()

    def _pick_communities(self, qvec: np.ndarray, top_k: int = COMM_TOPK) -> list[dict]:
        """选与查询最相关的 top-K 社区。"""
        sims = self.comm_vecs @ qvec.reshape(-1)  # [C,1024]@[1024]=[C]
        top = np.argsort(-sims)[:top_k]
        picked = []
        for i in top:
            c = dict(self.communities[i])
            c["score"] = float(sims[i])
            picked.append(c)
        return picked

    def _map_step(self, question: str, comm: dict) -> str:
        """Map：用单个社区摘要为问题生成部分答案。"""
        ents = ", ".join(f"{e['name']}({e['type']})" for e in comm["key_entities"])
        user = (f"【社区摘要】{comm['summary']}\n【关键实体】{ents}\n\n"
                f"【问题】{question}\n\n根据该社区信息，给出与问题相关的部分要点（若无相关则说'无相关'）。")
        return llm_chat("你是社区信息提取助手，根据社区摘要简明回答。", user,
                        temperature=0.0, max_tokens=256)

    def _reduce_step(self, question: str, partials: list[str]) -> str:
        """Reduce：把各社区部分答案聚合成最终答案。"""
        joined = "\n\n".join(f"[社区{i + 1}] {p}" for i, p in enumerate(partials))
        user = f"【各社区部分答案】\n{joined}\n\n【问题】{question}\n\n综合上述信息给出最终答案。"
        return llm_chat(ANSWER_SYSTEM, user, temperature=0.1, max_tokens=1024)

    def search(self, question: str, debug: bool = False) -> dict:
        # ① 选相关社区
        qvec = embed_query(question)
        picked = self._pick_communities(qvec)

        # ② Map：每个社区并行-ish（串行调用，简单清晰）生成部分答案
        map_results = [{"community_id": c["community_id"], "score": c["score"],
                        "summary": c["summary"],
                        "partial": self._map_step(question, c)} for c in picked]

        # ③ Reduce：聚合
        answer = self._reduce_step(question, [m["partial"] for m in map_results])

        return {"answer": answer, "picked_communities": picked, "map_results": map_results}


if __name__ == "__main__":
    # 自测：Local + Global 各跑一个问题
    import logging as _l

    _l.getLogger().setLevel(_l.INFO)
    q1 = "宁德时代有哪些子公司"
    q2 = "各家公司研发投入情况对比"
    print(f"\n===== Local Search: {q1} =====")
    r1 = GraphRAGLocal().search(q1)
    print(r1["answer"])
    print(f"\n  链接实体: {[e['name'] for e in r1['linked_entities']]}")
    print(f"  子图: {len(r1['subgraph']['nodes'])} 节点 / {len(r1['subgraph']['edges'])} 边")

    print(f"\n===== Global Search: {q2} =====")
    r2 = GraphRAGGlobal().search(q2)
    print(r2["answer"])
    print(f"  选中社区: {[(c['community_id'], round(c['score'], 3)) for c in r2['picked_communities']]}")
