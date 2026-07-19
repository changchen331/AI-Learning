"""
对比评估：手写 Prompt 解析版 vs Function Calling 版

评估维度：
  1. 格式稳定性（手写版专属：成功解析率）
  2. 步骤数（完成同一问题用了几步）
  3. 耗时（秒）
  4. 是否给出 Final Answer（成功率）

使用方式：
  python evaluate.py
  python evaluate.py --output ../evaluation/compare_result.json
"""

import argparse
import json
import logging
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.WARNING)

# 评估用问题集：覆盖不同工具组合和难度
EVAL_QUESTIONS = [
    {
        "id": "Q1",
        "question": "贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？",
        "expected_tools": ["rag_search", "rag_search", "calculator"],
        "note": "跨公司对比 + 计算",
    },
    {
        "id": "Q2",
        "question": "宁德时代2021年到2023年的营业收入分别是多少？复合增长率是多少？",
        "expected_tools": ["rag_search", "rag_search", "rag_search", "calculator"],
        "note": "跨年度趋势 + CAGR计算",
    },
    {
        "id": "Q3",
        "question": "海康威视2023年年报中提到了哪些主要风险因素？",
        "expected_tools": ["rag_search"],
        "note": "单次检索，定性问题",
    },
    {
        "id": "Q4",
        "question": "贵州茅台过去一年（2023年全年）的股价涨跌幅是多少？",
        "expected_tools": ["company_lookup", "stock_price", "calculator"],
        "note": "股价查询 + 计算",
    },
    {
        "id": "Q5",
        "question": "请预测明年A股市场的走势",
        "expected_tools": [],
        "note": "超出能力边界，应拒绝作答",
    },
]


def _run_single(mode: str, question: str, max_steps: int = 10) -> dict:
    """运行单个问题，收集评估指标"""
    if mode == "manual":
        from react_manual import run as react_run
    else:
        from react_function_calling import run as react_run

    steps = []
    parse_errors = 0
    start = time.time()
    final_answer = None
    success = False

    for step_data in react_run(question, max_steps=max_steps):
        steps.append(step_data)
        if step_data["type"] == "unparseable":
            parse_errors += 1
        if step_data["type"] == "final":
            final_answer = step_data["answer"]
            success = True

    elapsed = time.time() - start
    action_steps = [s for s in steps if s["type"] == "action"]
    tools_used = [s["action"] for s in action_steps]

    return {
        "mode": mode,
        "total_steps": len(action_steps),
        "elapsed_s": round(elapsed, 1),
        "success": success,
        "parse_errors": parse_errors,  # 手写版专属指标
        "tools_used": tools_used,
        "final_answer": (final_answer or "")[:200],
    }


def evaluate(output_path: str | None = None, max_steps: int = 10):
    results = []

    for q in EVAL_QUESTIONS:
        print(f"\n{'─' * 60}")
        print(f"[{q['id']}] {q['question']}")
        print(f"  预期工具: {q['expected_tools']}  ({q['note']})")

        for mode in ["manual", "fc"]:
            print(f"  运行 [{mode}]...", end=" ", flush=True)
            r = _run_single(mode, q["question"], max_steps)
            print(f"步骤:{r['total_steps']}  耗时:{r['elapsed_s']}s  成功:{r['success']}")
            results.append({**q, **r})

    # 汇总对比表
    print(f"\n{'=' * 60}")
    print(f"{'ID':<4} {'Mode':<8} {'Steps':<7} {'Time(s)':<9} {'Success':<8} {'ParseErr'}")
    print(f"{'─' * 60}")
    for r in results:
        print(
            f"{r['id']:<4} {r['mode']:<8} {r['total_steps']:<7} "
            f"{r['elapsed_s']:<9} {str(r['success']):<8} {r['parse_errors']}"
        )

    # 按维度汇总
    for mode in ["manual", "fc"]:
        mode_results = [r for r in results if r["mode"] == mode]
        avg_steps = sum(r["total_steps"] for r in mode_results) / len(mode_results)
        avg_time = sum(r["elapsed_s"] for r in mode_results) / len(mode_results)
        success_rate = sum(r["success"] for r in mode_results) / len(mode_results)
        total_errors = sum(r["parse_errors"] for r in mode_results)
        print(f"\n[{mode}] 平均步数:{avg_steps:.1f}  平均耗时:{avg_time:.1f}s  "
              f"成功率:{success_rate:.0%}  解析错误总数:{total_errors}")

    if output_path:
        # numpy scalar 安全转换
        def _safe(v):
            return v.item() if hasattr(v, "item") else v

        safe_results = [{k: _safe(v) for k, v in r.items()} for r in results]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(safe_results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存至 {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="保存 JSON 结果的路径")
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()
    evaluate(args.output, args.max_steps)
