"""
社区检测 + 社区摘要：把图谱切成主题社区，供 Global Search 用

教学重点（对应 PPT Part 4.3）：
  1. Leiden 社区检测：把密集相连的实体归为一组（一个"主题社区"）。
     例：茅台的子公司+高管+产品聚成一块，宁德的另一块——社区天然对应公司边界
  2. 社区摘要：每个社区用 LLM 生成一段结构化摘要（key entities/themes），
     Global Search 时把相关社区摘要直接喂给 LLM 做主题问答
  3. 优雅降级：本地 Neo4j 没装 GDS 插件，改用 Python networkx+igraph+leidenalg
     算法透明可见、零插件依赖（CLAUDE.md 4.4 降级规范）。GDS 装了的话可换，
     但教学上 Python 路线更清晰

流程：
  1. 从 Neo4j 读出所有实体节点 + 关系边
  2. networkx 构无向图 → igraph 转换 → Leiden 分社区
     （⚠️ leidenalg 0.12 + igraph 1.0 的 find_partition 有 bug，必须用
        ModularityVertexPartition(H) + Optimiser().optimise_partition(part) 显式路径）
  3. 把 community id 写回 Neo4j 节点属性
  4. 每个社区抽 top-N 代表实体 → LLM 生成摘要 → 存 outputs/communities.json

使用方式：
  python community_detect.py                # 检测 + 写回 + 摘要
  python community_detect.py --no-summary   # 只检测写回，不生成摘要（省 LLM 调用）

依赖：
  pip install neo4j networkx igraph leidenalg
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import llm_chat  # 社区摘要用 DeepSeek

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
COMMUNITIES_PATH = BASE_DIR / "outputs" / "communities.json"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = None

TOP_ENTITIES_PER_COMM = 15  # 每个社区取最多 15 个代表实体做摘要


# ── 1. 从 Neo4j 读图 ──────────────────────────────────────────────────────────

def load_graph_from_neo4j(driver):
    """读出所有 Entity 节点和关系，返回 (nodes, edges)。
    nodes: [{uid, name, type}], edges: [(src_uid, dst_uid), ...]"""
    with driver.session() as s:
        nodes = [{"uid": r["uid"], "name": r["name"], "type": r["type"]}
                 for r in s.run("MATCH (n:Entity) RETURN n.uid AS uid, n.name AS name, n.type AS type")]
        edges = [(r["src"], r["dst"]) for r in s.run(
            "MATCH (a:Entity)-[r]->(b:Entity) RETURN a.uid AS src, b.uid AS dst")]
    return nodes, edges


# ── 2. Leiden 社区检测 ───────────────────────────────────────────────────────

def detect_communities(nodes, edges):
    """networkx 建图 → igraph 转换 → Leiden。返回 {uid: community_id}。"""
    import networkx as nx
    from igraph import Graph as IGraph
    import leidenalg

    if not nodes:
        return {}

    G = nx.Graph()
    for n in nodes:
        G.add_node(n["uid"])
    for src, dst in edges:
        G.add_edge(src, dst)

    # 孤立节点 igraph TupleList 会丢，先记下；后面归到独立社区
    isolated = [n["uid"] for n in nodes if G.degree(n["uid"]) == 0]

    H = IGraph.TupleList(G.edges(), directed=False)
    name_to_idx = {H.vs[i]["name"]: i for i in range(H.vcount())}

    # ⚠️ leidenalg 0.12 + igraph 1.0.0 的 find_partition 有 bug（报 missing graph），
    #    必须用显式 partition + Optimiser 路径（见项目 ARCHITECTURE 踩坑表）
    part = leidenalg.ModularityVertexPartition(H)
    leidenalg.Optimiser().optimise_partition(part)
    logger.info(f"Leiden 完成: {len(part.membership)} 节点 → {len(set(part.membership))} 社区, "
                f"modularity={part.quality():.4f}")

    comm_of = {uid: part.membership[i] for uid, i in name_to_idx.items()}
    # 孤立节点各给一个独立社区 id（从已有最大 id 之后递增）
    next_id = max(part.membership) + 1 if part.membership else 0
    for uid in isolated:
        comm_of[uid] = next_id
        next_id += 1
    return comm_of


# ── 3. 社区 id 写回 Neo4j ─────────────────────────────────────────────────────

def write_communities_back(driver, comm_of):
    """把 community id 写回每个节点的 community 属性。"""
    with driver.session() as s:
        for uid, cid in comm_of.items():
            s.run("MATCH (n:Entity {uid:$uid}) SET n.community=$c", uid=uid, c=cid)
    logger.info(f"已写回 {len(comm_of)} 个节点的 community 属性")


# ── 4. 社区摘要（LLM）────────────────────────────────────────────────────────

def build_community_summaries(driver, comm_of, nodes):
    """每个社区取 top-N 代表实体（按度数），LLM 生成摘要。存 communities.json。"""
    # 按社区聚合实体
    comm_entities = defaultdict(list)
    node_by_uid = {n["uid"]: n for n in nodes}
    for uid, cid in comm_of.items():
        comm_entities[cid].append(node_by_uid[uid])

    # 从 Neo4j 读每个实体的度数，用于挑代表
    with driver.session() as s:
        deg = {r["uid"]: r["deg"] for r in s.run(
            "MATCH (n:Entity) OPTIONAL MATCH (n)-[r]-() "
            "WITH n, count(r) AS d RETURN n.uid AS uid, d AS deg")}

    summaries = []
    for cid, members in sorted(comm_entities.items()):
        # 按度数降序取 top
        members.sort(key=lambda n: -deg.get(n["uid"], 0))
        reps = members[:TOP_ENTITIES_PER_COMM]
        rep_desc = ", ".join(f"{m['name']}({m['type']})" for m in reps)
        summary = summarize_community(cid, reps)
        summaries.append({
            "community_id": cid,
            "size": len(members),
            "key_entities": [{"name": m["name"], "type": m["type"]} for m in reps],
            "summary": summary,
        })
        logger.info(f"社区 {cid} ({len(members)} 实体) 摘要: {summary[:60]}...")

    with open(COMMUNITIES_PATH, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    logger.info(f"社区摘要写入 {COMMUNITIES_PATH}（共 {len(summaries)} 社区）")
    return summaries


def summarize_community(cid, reps):
    """单社区 LLM 摘要。输入代表实体，输出一段主题描述。"""
    if not reps:
        return ""
    ent_list = "\n".join(f"- {m['name']}（{m['type']}）" for m in reps)
    system = ("你是知识图谱社区分析专家。根据给定的实体列表，用一段话概括这个社区的主题和关键关系，"
              "不超过100字。直接输出概括，不要分点。")
    user = f"社区编号 {cid} 包含以下实体：\n{ent_list}\n\n请概括这个社区的主题。"
    return llm_chat(system, user, temperature=0.2, max_tokens=256)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    from neo4j import GraphDatabase
    parser = argparse.ArgumentParser(description="Leiden 社区检测 + 摘要")
    parser.add_argument("--uri", default=NEO4J_URI)
    parser.add_argument("--no-summary", action="store_true", help="不生成 LLM 摘要")
    args = parser.parse_args()

    driver = GraphDatabase.driver(args.uri, auth=NEO4J_AUTH)
    driver.verify_connectivity()

    nodes, edges = load_graph_from_neo4j(driver)
    logger.info(f"从 Neo4j 读出 {len(nodes)} 节点 / {len(edges)} 边")
    if not nodes:
        logger.error("图库为空，请先运行 build_graph.py")
        return

    comm_of = detect_communities(nodes, edges)
    write_communities_back(driver, comm_of)

    if not args.no_summary:
        build_community_summaries(driver, comm_of, nodes)

    # 统计
    dist = Counter(comm_of.values())
    print(f"\n{'=' * 50}\n社区检测完成：{len(dist)} 个社区")
    print(f"最大社区 {max(dist.values())} 节点，最小 {min(dist.values())} 节点，平均 {len(comm_of) / len(dist):.1f}\n")
    driver.close()


if __name__ == "__main__":
    main()
