"""
LLM 驱动的知识抽取：年报文本 → 三元组 (Subject, Predicate, Object)

教学重点（对应 PPT Part 3.3）：
  1. 传统 NER+RE 需要专项标注、CRF/BiLSTM，迁移难；LLM 用 Prompt + 结构化输出
     (JSON) 零标注跨域即用，关系自由定义
  2. 实体类型 + 关系类型即 Schema，在 Prompt 里显式约束，输出可直接入库
  3. 抽取成本控制：15 份×1500 blocks 全抽太贵，按关键 section 抽样前 N 块
     ——「索引成本比向量 RAG 高 5~10×」本身是教学点

抽取的 Schema（实体类型 / 关系类型，与 build_graph.py 入库时严格对应）：
  实体类型: Company 公司 / Person 人物 / Subsidiary 子公司 / Product 产品
            Indicator 财务指标 / Segment 业务板块 / Region 地区
  关系类型:
    REPORTS        (公司, 指标值)        —— 财务数据，object 带数值+单位
    SERVES_AS      (人物, 公司)          —— 董监高任职，object 带 role
    CONTROLS       (人物/公司, 公司)     —— 实控人/控股股东
    HAS_SUBSIDIARY (公司, 子公司)         —— object 带持股比例
    PRODUCES       (公司, 产品)           —— 主要产品
    OPERATES_IN    (公司, 地区)           —— 经营地区
    INVESTS_IN     (公司, 板块/项目)      —— 研发/投资方向
    RELATED_TO     (通用兜底)

使用方式：
  # 全量抽取（15 份年报，每份前 25 个关键块，约 6 分钟）
  python extract_triples.py

  # 快速演示（只抽 1 份报告前 5 块）
  python extract_triples.py --max-reports 1 --max-blocks-per-report 5

  # 自定义输入/输出路径
  python extract_triples.py --parsed-dir ../../rag_annual_report/data/parsed --output outputs/triples.jsonl

依赖：
  pip install openai
  export DEEPSEEK_API_KEY="sk-xxx"   # LLM 用 DeepSeek
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# 同目录 import（脚本直接运行时也能 import 兄弟模块）
sys.path.insert(0, str(Path(__file__).parent))
from llm_client import llm_chat

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
PARSED_DIR = BASE_DIR.parent / "rag_annual_report" / "data" / "parsed"
OUTPUT_PATH = BASE_DIR / "outputs" / "triples.jsonl"
STATS_PATH = BASE_DIR / "outputs" / "extract_stats.json"

# 关键 section 关键词：只从这些章节抽，避开目录页/格式噪声，提升抽取质量、控制成本
KEY_SECTION_KEYWORDS = [
    "公司简介", "业务概要", "经营情况", "主要会计数据", "财务指标",
    "股东", "董事", "监事", "高级管理", "实际控制人", "控股股东",
    "子公司", "参股", "主要产品", "营业收入", "研发", "员工", "地区",
]

# ── 抽取 Prompt（核心教学载体，学生重点看这里）─────────────────────────────
SYSTEM_PROMPT = """你是知识图谱抽取专家。从给定的上市公司年报文本片段中抽取结构化三元组，用于构建知识图谱。

【实体类型】（只能用以下 7 种）
- Company       公司（上市公司本身）
- Person        人物（董事/监事/高管/实控人/股东）
- Subsidiary    子公司或参股公司
- Product       主要产品
- Indicator     财务指标（营业收入/净利润/研发投入等，object 写具体数值+单位）
- Segment       业务板块/项目
- Region        地区

【关系类型】（只能用以下 8 种）
- REPORTS        Company -> Indicator   例:(贵州茅台, REPORTS, 营业收入1240亿元)
- SERVES_AS      Person -> Company       例:(丁雄军, SERVES_AS, 贵州茅台) object_type固定Company, role填职务
- CONTROLS       Person/Company -> Company  实控人/控股股东
- HAS_SUBSIDIARY Company -> Subsidiary  例:(宁德时代, HAS_SUBSIDIARY, 宁德新能源科技) ratio填持股比例
- PRODUCES       Company -> Product
- OPERATES_IN    Company -> Region
- INVESTS_IN     Company -> Segment     研发投入方向
- RELATED_TO     通用兜底，尽量少用

【输出格式】严格 JSON，不要 markdown 代码块，不要解释：
{
  "triples": [
    {"subject":"贵州茅台","subject_type":"Company","predicate":"REPORTS","object":"营业收入1240.99亿元","object_type":"Indicator","year":"2022","role":null,"ratio":null},
    {"subject":"丁雄军","subject_type":"Person","predicate":"SERVES_AS","object":"贵州茅台","object_type":"Company","year":"2022","role":"董事长","ratio":null}
  ]
}

【规则】
1. 只抽取文本中明确出现的事实，不得编造或推断
2. 数值尽量精确，保留原文的金额和单位（亿元/万元/元）
3. 实体名用规范化简称（贵州茅台酒股份有限公司→贵州茅台；宁德时代新能源科技股份有限公司→宁德时代）
4. 没有可抽取事实时返回 {"triples": []}
5. 一段文本通常抽 1~6 条三元组，宁缺毋滥"""


def make_user_prompt(block_content: str, section: str, stock: str, year: str) -> str:
    """组装单次抽取的 user message。把来源元信息也给模型，辅助规范化。"""
    return (
        f"【来源】{stock} {year}年年报 · 章节：{section}\n"
        f"【文本】\n{block_content}\n\n"
        f"请抽取上述文本中的三元组，严格按指定 JSON 格式输出。"
    )


# ── 块选择：按关键 section 抽样前 N 块 ────────────────────────────────────────

def is_key_section(section_path: list[str]) -> bool:
    """section_path 形如 ['1 / 127', '第二节', '第四节 > 二、公司控股股东...']。
    只要任一段含关键词就算关键 section。"""
    joined = " > ".join(section_path or [])
    return any(kw in joined for kw in KEY_SECTION_KEYWORDS)


def select_blocks(report_json: dict, max_blocks: int) -> list[dict]:
    """从单份报告里挑出值得抽取的块：text/table 型、内容够长、属关键 section，取前 max_blocks 个。"""
    picked = []
    for b in report_json.get("blocks", []):
        if b.get("block_type") not in ("text", "table"):
            continue
        content = b.get("content", "") or ""
        if len(content) < 80:  # 太短的块多是页眉/目录/格式噪声
            continue
        if not is_key_section(b.get("section_path", [])):
            continue
        picked.append(b)
        if len(picked) >= max_blocks:
            break
    return picked


# ── 单块抽取 ──────────────────────────────────────────────────────────────────

def extract_block_triples(block: dict, stock: str, year: str) -> list[dict]:
    """对单个 block 调一次 LLM 抽取，返回三元组列表。解析失败返回空列表。"""
    content = block.get("content", "")
    section = " > ".join(block.get("section_path") or [])
    user_msg = make_user_prompt(content, section, stock, year)

    raw = llm_chat(SYSTEM_PROMPT, user_msg, temperature=0.0, max_tokens=2048,
                   response_format_json=True)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"JSON 解析失败 (stock={stock} p{block.get('page_num')}): {raw[:80]}")
        return []

    triples = data.get("triples", [])
    # 给每条三元组打上来源溯源信息（入库后可查到出自哪份年报哪一页）
    src = {"stock": stock, "year": year, "page": block.get("page_num"),
           "section": section, "source_file": block.get("source_file", "")}
    for t in triples:
        t["source"] = src
    return triples


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="年报三元组抽取（LLM 驱动）")
    parser.add_argument("--parsed-dir", type=str, default=str(PARSED_DIR),
                        help="parsed JSON 所在目录")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH),
                        help="输出 jsonl 路径")
    parser.add_argument("--max-reports", type=int, default=0,
                        help="最多处理几份报告，0=全部（15 份）")
    parser.add_argument("--max-blocks-per-report", type=int, default=25,
                        help="每份报告最多抽几个块（成本控制）")
    args = parser.parse_args()

    parsed_dir = Path(args.parsed_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    report_files = sorted(parsed_dir.glob("*.json"))
    if args.max_reports:
        report_files = report_files[:args.max_reports]
    logger.info(f"待处理报告 {len(report_files)} 份，每份最多 {args.max_blocks_per_report} 块")

    total_triples = 0
    total_blocks = 0
    t0 = time.time()

    with open(output, "w", encoding="utf-8") as fout:
        for ri, rf in enumerate(report_files):
            # 文件名约定：600519_2022_贵州茅台_...json → 解析出 stock/year
            name_parts = rf.stem.split("_")
            stock = name_parts[0] if name_parts else ""
            year = name_parts[1] if len(name_parts) > 1 else ""
            try:
                report = json.load(open(rf, encoding="utf-8"))
            except Exception as e:
                logger.warning(f"读取 {rf.name} 失败: {e}")
                continue

            blocks = select_blocks(report, args.max_blocks_per_report)
            logger.info(f"[{ri + 1}/{len(report_files)}] {rf.name} 选中 {len(blocks)} 块")

            for bi, block in enumerate(blocks):
                triples = extract_block_triples(block, stock, year)
                for t in triples:
                    fout.write(json.dumps(t, ensure_ascii=False) + "\n")
                total_triples += len(triples)
                total_blocks += 1
                # 进度日志：每 50 块打印一次，避免长时间无输出（CLAUDE.md 4.3）
                if total_blocks % 50 == 0:
                    elapsed = time.time() - t0
                    logger.info(f"进度: {total_blocks} 块, {total_triples} 三元组, 用时 {elapsed:.0f}s")

    elapsed = time.time() - t0
    stats = {
        "reports": len(report_files),
        "blocks_processed": total_blocks,
        "triples_extracted": total_triples,
        "elapsed_seconds": round(elapsed, 1),
        "avg_triples_per_block": round(total_triples / max(total_blocks, 1), 2),
        "model": "deepseek-chat (deepseek-v4-flash)",
    }
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"抽取完成: {total_triples} 条三元组 / {total_blocks} 块 / {len(report_files)} 份报告")
    logger.info(f"用时 {elapsed:.0f}s  |  写入 {output}")
    logger.info(f"统计 {stats}")


if __name__ == "__main__":
    main()
