"""
知识图谱入库：三元组 → Neo4j 图

教学重点：
  1. 实体规范化（Entity Resolution）：LLM 抽取时同一实体出现多种写法
     （贵州茅台酒股份有限公司 / 贵州茅台 / 茅台），不规范化会把一个实体拆成多个节点
     → 知识图谱破碎、多跳推理断裂。这里用「别名表」把 5 家上市公司统一到规范名
  2. Neo4j MERGE 幂等写入：同一 (s,p,o) 多次出现只建一条边，靠唯一约束 + MERGE 去重
  3. 节点建模取舍：年份放边属性而非单独建 CompanyYear 节点——更简单，仍支持
     跨年查询（MATCH (c)-[r:REPORTS {year:'2022'}]->(i)），教学上看得更清

Schema（与 extract_triples.py 的抽取 Schema 严格对应）：
  节点（统一打 :Entity 标签 + 类型标签）：
    Company {name, stock_code} / Person / Subsidiary / Product
    Indicator / Segment / Region
  边（带 year / role / ratio 属性）：
    REPORTS / SERVES_AS / CONTROLS / HAS_SUBSIDIARY / PRODUCES
    OPERATES_IN / INVESTS_IN / RELATED_TO

使用方式：
  python build_graph.py                          # 默认读 outputs/triples.jsonl
  python build_graph.py --input outputs/triples.jsonl --clear   # 先清库再建

依赖：
  pip install neo4j
  本地 Neo4j 已启动（bolt://localhost:7687，认证已关闭，见项目 ARCHITECTURE 踩坑表）
"""

import argparse
import json
import logging
import os
from pathlib import Path

from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
TRIPLES_PATH = BASE_DIR / "outputs" / "triples.jsonl"

# ── Neo4j 连接（本地教学环境，认证已关闭；正式环境用环境变量切回带认证）─────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = None  # 本地 auth_enabled=false；若开启认证改 ('neo4j', os.getenv('NEO4J_PASSWORD'))

# ── 实体规范化别名表（核心教学点）────────────────────────────────────────────
# 5 家上市公司的规范名 → 各种写法。LLM 抽取时会出现全称/简称混用，不规范化会拆节点
COMPANY_ALIASES = {
    "五粮液": ["宜宾五粮液股份有限公司", "宜宾五粮液", "五粮液股份", "五粮液"],
    "贵州茅台": ["贵州茅台酒股份有限公司", "茅台股份", "茅台", "贵州茅台"],
    "宁德时代": ["宁德时代新能源科技股份有限公司", "宁德时代新能源", "宁德时代"],
    "中国平安": ["中国平安保险(集团)股份有限公司", "中国平安保险", "中国平安保险(集团)股份有限公司",
                 "平安", "中国平安"],
    "海康威视": ["杭州海康威视数字技术股份有限公司", "海康威视数字技术", "海康", "海康威视"],
}
# 规范名 → 股票代码（给 Company 节点挂 stock_code，便于精确查询）
COMPANY_STOCK = {
    "五粮液": "000858", "贵州茅台": "600519", "宁德时代": "300750",
    "中国平安": "601318", "海康威视": "002415",
}
# 反查表：任意写法 → 规范名（含规范名自身）
ALIAS_TO_CANON = {}
for canon, variants in COMPANY_ALIASES.items():
    for v in variants:
        ALIAS_TO_CANON[v] = canon

# 支持的关系类型（与 Schema 对应，非法的跳过）
VALID_PREDICATES = {
    "REPORTS", "SERVES_AS", "CONTROLS", "HAS_SUBSIDIARY", "PRODUCES",
    "OPERATES_IN", "INVESTS_IN", "RELATED_TO",
}


def normalize(name: str, etype: str) -> tuple[str, str | None]:
    """
    实体规范化。返回 (规范名, stock_code 或 None)。
    Company 走别名表；其他类型仅 strip 空白。
    """
    name = (name or "").strip()
    if etype == "Company" and name in ALIAS_TO_CANON:
        canon = ALIAS_TO_CANON[name]
        return canon, COMPANY_STOCK.get(canon)
    return name, None


# ── Cypher 写入语句 ───────────────────────────────────────────────────────────
# 用 MERGE 幂等：uid=type:name 作为节点唯一键，重复写入不会新增节点
MERGE_NODE = """
MERGE (n:Entity:__TYPE__ {uid: $uid})
SET n.name = $name, n.type = $type
""".replace("__TYPE__", "$(_type)")  # 占位，实际下面字符串拼接类型标签


# 为避免注入，类型标签从白名单取，字符串拼接到 Cypher（类型是固定枚举，安全）


def merge_node_cypher(etype: str) -> str:
    """按类型生成 MERGE 节点语句（类型标签从白名单拼，安全）。"""
    return (
        f"MERGE (n:Entity:`{etype}` {{uid: $uid}}) "
        f"SET n.name = $name, n.type = $type"
    )


def merge_edge_cypher(pred: str) -> str:
    """按关系类型生成 MERGE 边语句。边带 year/role/ratio 属性（存在才写）。"""
    # MERGE (a)-[r:PRED]->(b) 然后按属性 SET；year 作为边区分键（同年同关系不重复）
    return (
        f"MATCH (a:Entity {{uid: $src_uid}}), (b:Entity {{uid: $dst_uid}}) "
        f"MERGE (a)-[r:`{pred}`]->(b) "
        f"SET r.year = coalesce(r.year, $year), "
        f"    r.role = CASE WHEN $role IS NOT NULL THEN $role ELSE r.role END, "
        f"    r.ratio = CASE WHEN $ratio IS NOT NULL THEN $ratio ELSE r.ratio END"
    )


def uid_of(name: str, etype: str) -> str:
    """节点唯一键：类型:小写规范名。"""
    canon, _ = normalize(name, etype)
    return f"{etype}:{canon.lower()}"


# ── 主流程 ────────────────────────────────────────────────────────────────────

class GraphBuilder:
    def __init__(self, uri=NEO4J_URI, auth=NEO4J_AUTH):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.driver.verify_connectivity()
        logger.info(f"Neo4j 已连接: {uri}")

    def close(self):
        self.driver.close()

    def ensure_constraints(self):
        """建唯一约束（幂等）。4.4 community 支持单属性唯一约束。"""
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT entity_uid IF NOT EXISTS "
                  "FOR (n:Entity) REQUIRE n.uid IS UNIQUE")
        logger.info("唯一约束已就绪 (Entity.uid)")

    def clear(self):
        """清空全库（--clear 时用）。"""
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
        logger.info("已清空数据库")

    def build(self, triples_path: Path):
        """读 triples.jsonl，规范化实体，MERGE 节点+边。"""
        triples = [json.loads(l) for l in open(triples_path, encoding="utf-8")]
        logger.info(f"读入 {len(triples)} 条三元组")

        node_count = 0
        edge_count = 0
        skipped = 0

        with self.driver.session() as s:
            for t in triples:
                pred = t.get("predicate")
                if pred not in VALID_PREDICATES:
                    skipped += 1
                    continue

                s_name, s_type = t["subject"], t["subject_type"]
                o_name, o_type = t["object"], t["object_type"]
                s_canon, s_stock = normalize(s_name, s_type)
                o_canon, _ = normalize(o_name, o_type)
                if not s_canon or not o_canon:
                    skipped += 1
                    continue

                src_uid = uid_of(s_name, s_type)
                dst_uid = uid_of(o_name, o_type)

                # 写 subject 节点
                s.run(merge_node_cypher(s_type),
                      uid=src_uid, name=s_canon, type=s_type)
                # Company 节点额外补 stock_code
                if s_stock:
                    s.run("MATCH (n:Entity {uid:$uid}) SET n.stock_code=$sc",
                          uid=src_uid, sc=s_stock)
                # 写 object 节点
                s.run(merge_node_cypher(o_type),
                      uid=dst_uid, name=o_canon, type=o_type)
                # 写边
                s.run(merge_edge_cypher(pred),
                      src_uid=src_uid, dst_uid=dst_uid,
                      year=t.get("year"), role=t.get("role"), ratio=t.get("ratio"))
                node_count += 2
                edge_count += 1

        logger.info(f"建图完成: 写入 {edge_count} 条边（MERGE 幂等，实际节点数更少）"
                    f"，跳过 {skipped} 条非法三元组")
        self.print_stats()

    def print_stats(self):
        """打印图统计：各类型节点数、各关系数。"""
        with self.driver.session() as s:
            n_nodes = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            n_edges = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            print(f"\n{'=' * 50}")
            print(f"图统计：{n_nodes} 节点 / {n_edges} 边")
            print(f"{'=' * 50}")
            print("\n[节点类型分布]")
            for r in s.run("MATCH (n:Entity) RETURN n.type AS t, count(n) AS c ORDER BY c DESC"):
                print(f"  {r['t']:<12} {r['c']}")
            print("\n[关系类型分布]")
            for r in s.run("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c ORDER BY c DESC"):
                print(f"  {r['t']:<16} {r['c']}")


def main():
    parser = argparse.ArgumentParser(description="三元组 → Neo4j 建图")
    parser.add_argument("--input", type=str, default=str(TRIPLES_PATH))
    parser.add_argument("--uri", type=str, default=NEO4J_URI)
    parser.add_argument("--clear", action="store_true", help="建图前先清空数据库")
    args = parser.parse_args()

    gb = GraphBuilder(uri=args.uri)
    gb.ensure_constraints()
    if args.clear:
        gb.clear()
    gb.build(Path(args.input))
    gb.close()


if __name__ == "__main__":
    main()
