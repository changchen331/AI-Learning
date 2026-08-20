"""
Parallel vs Serial 量化对比（凸显 subagent 并行优势）

教学重点：
  同一组调研问题，主 agent 派发的 subagent 分别用「并行(ThreadPool)」和
  「串行(for 循环)」两种方式执行，对比 wall-clock，量化并行加速。

  并行的意义不是少做事，而是把 N 个独立子任务的墙钟时间从 sum 压到 max。
  本项目的 dispatch_subagents 用 ThreadPoolExecutor 实现并行，
  serial=True 时退化为串行（eval 基线）。

使用方式：
  python eval_compare.py            # 默认 4 题，parallel vs serial
  python eval_compare.py --limit 2  # 快速版
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EVAL_QUESTIONS = [
    "2024年中国新能源汽车市场调研：销量规模、主要厂商竞争格局、政策趋势",
    "中国咖啡市场调研：市场规模、主要品牌、消费趋势",
    "中国扫地机器人市场调研：市场规模、主要品牌、技术趋势",
    "中国宠物经济调研：市场规模、主要品类、消费趋势",
]


def run_one(question, serial):
    """跑一次调研，返回 (wall_clock, n_subagents, dispatched)。
    serial=True/False 控制子 agent 执行方式。"""
    import agents
    t0 = time.time()
    r = agents.run_research(question, serial=serial)
    wall = time.time() - t0
    ps = r["parallel_stats"][-1] if r["parallel_stats"] else None
    return {
        "wall": round(wall, 2),
        "n_subagents": ps["n_subagents"] if ps else 0,
        "dispatch_wall": ps["wall_clock"] if ps else 0,
        "serial_sum": ps["serial_sum"] if ps else 0,
        "speedup": ps["speedup"] if ps else 0,
        "dispatched": len(r["dispatches"]) > 0,
    }


def main():
    parser = argparse.ArgumentParser(description="parallel vs serial 对比")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    qs = EVAL_QUESTIONS[:args.limit] if args.limit else EVAL_QUESTIONS

    results = []
    for i, q in enumerate(qs):
        logger.warning(f"[{i + 1}/{len(qs)}] {q}")
        p = run_one(q, serial=False)
        s = run_one(q, serial=True)
        results.append({"question": q, "parallel": p, "serial": s})
        print(f"  {q[:22]:<24} 并行 {p['wall']}s vs 串行 {s['wall']}s "
              f"(subagent {p['n_subagents']}, 加速 {p['speedup']}×)")

    avg_p = sum(r["parallel"]["wall"] for r in results) / len(results)
    avg_s = sum(r["serial"]["wall"] for r in results) / len(results)
    avg_spd = sum(r["parallel"]["speedup"] for r in results) / len(results)

    print(f"\n{'=' * 60}\nParallel vs Serial 对比（{len(results)} 题）\n{'=' * 60}")
    print(f"{'指标':<16} {'并行(ThreadPool)':<18} {'串行(for循环)':<18}")
    print(f"{'平均墙钟(s)':<16} {avg_p:<18.2f} {avg_s:<18.2f}")
    print(f"{'平均加速':<16} {avg_spd:<18.2f}× {'—':<18}")
    print(f"\n结论：subagent 并行把 N 个独立子任务的墙钟从 sum 压到 ≈max，"
          f"平均加速 {avg_spd:.2f}×")

    out = {"summary": {"avg_parallel_s": round(avg_p, 2),
                       "avg_serial_s": round(avg_s, 2),
                       "avg_speedup": round(avg_spd, 2)},
           "details": results}
    (Path(__file__).parent.parent / "outputs" / "eval_compare.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
