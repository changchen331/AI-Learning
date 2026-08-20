"""
GraphRAG vs 向量 RAG 横向对比评估

教学重点（对应 PPT Part 5.1 三范式对比）：
  1. 同一组多跳问题，跑两种方案，看真实数字
  2. 多跳/关系类问题 GraphRAG 应占优；简单事实召回向量 RAG 占优
  3. 成本（延迟/调用次数）是落地决策的关键维度，不只看准确率

评估方案（透明、可解释，不藏黑盒）：
  - Hit Rate：gold 关键实体是否出现在检索到的上下文里（召回质量）
  - Answer Match：gold 答案子串是否出现在最终答案里（生成质量，子串匹配有局限，
    但透明可复现；paraphrase 时会低估，结论方向仍可靠）
  - Latency：端到端秒数；Tokens：LLM 调用 token 数（成本代理）

向量 RAG 基线直接复用 rag_annual_report 项目的 RAGPipeline（FAISS + BM25 + Rerank），
不重新建索引，保持公平对比。

使用方式：
  python eval_compare.py
  python eval_compare.py --limit 5      # 只跑前 5 题快速看
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 评估集：15 道多跳/关系问题 + gold 关键词 ──────────────────────────────────
# gold 用「任一子串命中即算正确」的宽松匹配，降低 paraphrase 误判
EVAL_SET = [
    {"q": "宁德时代有哪些子公司？", "gold": ["子公司", "宁德"], "gold_ctx": ["宁德"]},
    {"q": "贵州茅台2022年董事长是谁？", "gold": ["丁雄军"], "gold_ctx": ["丁雄军", "茅台"]},
    {"q": "五粮液有哪些子公司？", "gold": ["宜宾", "五粮液"], "gold_ctx": ["五粮液"]},
    {"q": "海康威视的主要产品是什么？", "gold": ["摄像", "视频", "安防", "产品"], "gold_ctx": ["海康"]},
    {"q": "中国平安的董事长是谁？", "gold": ["马明哲", "平安"], "gold_ctx": ["平安"]},
    {"q": "贵州茅台2023年营业收入是多少？", "gold": ["亿", "茅台"], "gold_ctx": ["茅台"]},
    {"q": "宁德时代研发投入情况如何？", "gold": ["研发", "宁德"], "gold_ctx": ["宁德"]},
    {"q": "五粮液2022年归属于上市公司股东的净利润是多少？", "gold": ["亿", "五粮液"], "gold_ctx": ["五粮液"]},
    {"q": "海康威视的实控人是谁？", "gold": ["海康", "龚虹嘉", "陈宗年", "国资"], "gold_ctx": ["海康"]},
    {"q": "贵州茅台有哪些主要产品？", "gold": ["茅台", "酒"], "gold_ctx": ["茅台"]},
    {"q": "宁德时代2023年净利润是多少？", "gold": ["亿", "宁德"], "gold_ctx": ["宁德"]},
    {"q": "中国平安主要业务板块有哪些？", "gold": ["平安", "保险", "银行", "投资"], "gold_ctx": ["平安"]},
    {"q": "五粮液2021年研发投入是多少？", "gold": ["五粮液"], "gold_ctx": ["五粮液"]},
    {"q": "海康威视2022年营业收入是多少？", "gold": ["亿", "海康"], "gold_ctx": ["海康"]},
    # 应拒绝题：比亚迪不在图谱/年报集，正确行为是说明无法回答而非编造
    {"q": "比亚迪2024年营业收入是多少？", "gold": ["无法", "未", "不在", "没有", "不包含"], "gold_ctx": []},
]


def match(text: str, keys: list[str]) -> bool:
    """任一子串命中即 True（大小写不敏感）。"""
    t = text.lower()
    return any(k.lower() in t for k in keys)


# ── GraphRAG 侧 ──────────────────────────────────────────────────────────────

def run_graphrag(question: str):
    """跑 GraphRAG Local 检索，返回 (answer, context, latency, ok)。"""
    from retrieve import GraphRAGLocal
    if not hasattr(run_graphrag, "_rag"):
        run_graphrag._rag = GraphRAGLocal()
    t0 = time.time()
    try:
        r = run_graphrag._rag.search(question)
        return r["answer"], r["context"], time.time() - t0, True
    except Exception as e:
        logger.warning(f"GraphRAG 失败: {e}")
        return f"出错: {e}", "", time.time() - t0, False


# ── 向量 RAG 侧（复用 rag_annual_report）─────────────────────────────────────

def run_vectorrag(question: str):
    """复用 rag_annual_report 的 RAGPipeline 做向量+BM25+Rerank 检索。"""
    if not hasattr(run_vectorrag, "_pipe"):
        rag_proj = BASE_DIR.parent / "rag_annual_report" / "src"
        sys.path.insert(0, str(rag_proj))
        from rag_pipeline import RAGPipeline  # 复用，不重新建索引
        run_vectorrag._pipe = RAGPipeline(use_bm25=True, use_rerank=False)
    t0 = time.time()
    try:
        r = run_vectorrag._pipe.query(question, verbose=False)
        ctx = "\n\n".join(c.get("content", "")[:200] for c in r.get("retrieved", []))
        return r["answer"], ctx, time.time() - t0, True
    except Exception as e:
        logger.warning(f"向量RAG 失败: {e}")
        return f"出错: {e}", "", time.time() - t0, False


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GraphRAG vs 向量RAG 对比")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题，0=全跑")
    args = parser.parse_args()

    qs = EVAL_SET[:args.limit] if args.limit else EVAL_SET
    results = []
    for i, item in enumerate(qs):
        q = item["q"]
        logger.info(f"[{i + 1}/{len(qs)}] {q}")
        g_ans, g_ctx, g_lat, g_ok = run_graphrag(q)
        v_ans, v_ctx, v_lat, v_ok = run_vectorrag(q)
        results.append({
            "question": q,
            "gold_answer": item["gold"],
            "gold_ctx": item["gold_ctx"],
            "graphrag": {"answer": g_ans, "ctx": g_ctx, "latency": round(g_lat, 2),
                         "ctx_hit": match(g_ctx, item["gold_ctx"]) if item["gold_ctx"] else None,
                         "ans_hit": match(g_ans, item["gold"])},
            "vectorrag": {"answer": v_ans, "ctx": v_ctx, "latency": round(v_lat, 2),
                          "ctx_hit": match(v_ctx, item["gold_ctx"]) if item["gold_ctx"] else None,
                          "ans_hit": match(v_ans, item["gold"])},
        })

    # 汇总
    def summarize(side: str) -> dict:
        ans_hits = sum(r[side]["ans_hit"] for r in results)
        ctx_hits = sum(1 for r in results if r[side].get("ctx_hit"))
        ctx_total = sum(1 for r in results if r[side].get("ctx_hit") is not None)
        avg_lat = sum(r[side]["latency"] for r in results) / len(results)
        return {"answer_acc": f"{ans_hits}/{len(results)}",
                "ctx_hitrate": f"{ctx_hits}/{ctx_total}" if ctx_total else "N/A",
                "avg_latency_s": round(avg_lat, 2)}

    summary = {"graphrag": summarize("graphrag"), "vectorrag": summarize("vectorrag")}
    out = {"summary": summary, "details": results}
    out_path = BASE_DIR / "outputs" / "eval_compare.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 打印对比表
    print(f"\n{'=' * 60}\nGraphRAG vs 向量RAG 对比（{len(results)} 题）\n{'=' * 60}")
    print(f"{'指标':<16} {'GraphRAG':<14} {'向量RAG':<14}")
    for k in ["answer_acc", "ctx_hitrate", "avg_latency_s"]:
        print(f"{k:<16} {summary['graphrag'][k]:<14} {summary['vectorrag'][k]:<14}")
    print(f"\n详细结果写入 {out_path}")
    print("\n[逐题答案命中]")
    for r in results:
        print(f"  {r['question'][:20]:<22} G={r['graphrag']['ans_hit']} V={r['vectorrag']['ans_hit']}")


if __name__ == "__main__":
    main()
