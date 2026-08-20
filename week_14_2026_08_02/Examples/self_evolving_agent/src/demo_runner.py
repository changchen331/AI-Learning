"""
演示主程序：运行 Skill 自进化实验，输出多版本对比数据。

教学重点：
  完整展示"无进化基线 → 运行时 Nudge 多轮进化"的量化效果对比。
  每次 Nudge 后跑 30 题 probe eval，记录各版本对同一问题的回答差异。

使用方式：
  cd self_evolving_agent
  python src/demo_runner.py

依赖：
  pip install openai
  set DEEPSEEK_API_KEY=your_key
"""

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from skill_manager import SkillManager
from evaluator import Evaluator
from agent import CustomerServiceAgent
from background_reviewer import BackgroundReviewer

SKILLS_DIR = ROOT / "skills"
SKILLS_ORIG = ROOT / "outputs" / "skills_original"  # 只建一次，永不覆盖
EVAL_SET = ROOT / "data" / "eval_set.json"
DEMO_SCRIPT = ROOT / "data" / "demo_script.json"
POLICIES = ROOT / "data" / "policies.md"
VERSIONS_DIR = ROOT / "outputs" / "skill_versions"
EVAL_RUNS_DIR = ROOT / "outputs" / "eval_runs"
EVOL_LOG = ROOT / "outputs" / "evolution_log.json"


# ── 备份 / 还原 ──────────────────────────────────────────────────────────────

def ensure_original(sm: SkillManager):
    """首次运行时创建原始备份（并保存快照 v1），之后只读不写。"""
    if not SKILLS_ORIG.exists():
        shutil.copytree(SKILLS_DIR, SKILLS_ORIG)
        print(f"✓ 首次运行：原始 Skills 备份至 {SKILLS_ORIG.name}/")
        # 把初始 Skills 也存入版本历史（v1 快照）
        for skill_name, content in sm.load_all().items():
            sm._save_version(skill_name, content, action="initial", reason="初始版本")
    else:
        print("✓ 检测到原始备份，已跳过覆盖")


def restore_from_original():
    """每次实验前从原始备份还原，保证可重复运行。"""
    if not SKILLS_ORIG.exists():
        raise RuntimeError("原始备份不存在，请删除 outputs/ 并重新运行")
    if SKILLS_DIR.exists():
        shutil.rmtree(SKILLS_DIR)
    shutil.copytree(SKILLS_ORIG, SKILLS_DIR)
    # 清空版本历史和快照，重新开始
    if VERSIONS_DIR.exists():
        shutil.rmtree(VERSIONS_DIR)
    snapshots = ROOT / "outputs" / "skill_snapshots"
    if snapshots.exists():
        shutil.rmtree(snapshots)
    if EVAL_RUNS_DIR.exists():
        shutil.rmtree(EVAL_RUNS_DIR)
    print("✓ 已还原初始 Skills，清空上次版本历史")


# ── Probe / Full Eval ─────────────────────────────────────────────────────────

def run_probe_eval(agent: CustomerServiceAgent, evaluator: Evaluator,
                   probe_ids: list[int], run_id: str, label: str,
                   sm: SkillManager) -> dict:
    """
    对 probe_ids 指定的题目运行评估，保存到 eval_runs/{run_id}.json。
    返回包含 per-question 答案的完整结果字典。
    """
    EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    total, correct = 0, 0
    by_category: dict = {}
    answers: dict = {}

    for qid in probe_ids:
        q = evaluator.questions[qid]
        answer = agent.answer(q["question"])
        ok, reason = evaluator.evaluate_answer(answer, qid)
        total += 1
        cat = q["category"]
        by_category.setdefault(cat, {"total": 0, "correct": 0})
        by_category[cat]["total"] += 1
        if ok:
            correct += 1
            by_category[cat]["correct"] += 1
        answers[str(qid)] = {"answer": answer, "correct": ok, "fail_reason": reason if not ok else ""}

    for cat in by_category.values():
        cat["accuracy"] = round(cat["correct"] / cat["total"], 3)

    result = {
        "run_id": run_id,
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "skill_versions_active": sm.get_active_versions(),
        "summary": {"total": total, "correct": correct, "accuracy": round(correct / total, 3)},
        "by_category": by_category,
        "answers": answers,
    }
    (EVAL_RUNS_DIR / f"{run_id}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def run_full_eval(agent: CustomerServiceAgent, evaluator: Evaluator,
                  run_id: str, label: str, sm: SkillManager) -> dict:
    all_ids = list(evaluator.questions.keys())
    return run_probe_eval(agent, evaluator, all_ids, run_id, label, sm)


# ── Evolution Log ─────────────────────────────────────────────────────────────

class EvolutionLog:
    """收集实验过程中所有结构化数据，实验结束后写入 evolution_log.json。"""

    def __init__(self):
        self.eval_runs: list[dict] = []
        self.nudge_events: list[dict] = []
        self.question_history: dict[str, list] = {}  # qid -> [{run_id, answer, correct}]

    def add_eval_run(self, result: dict):
        self.eval_runs.append({
            "run_id": result["run_id"],
            "label": result["label"],
            "timestamp": result["timestamp"],
            "skill_versions_active": result["skill_versions_active"],
            "accuracy": result["summary"]["accuracy"],
            "correct": result["summary"]["correct"],
            "total": result["summary"]["total"],
            "by_category": {k: v["accuracy"] for k, v in result["by_category"].items()},
        })
        # 更新 per-question 历史
        for qid_str, ans_data in result["answers"].items():
            self.question_history.setdefault(qid_str, []).append({
                "run_id": result["run_id"],
                "label": result["label"],
                "skill_versions": result["skill_versions_active"],
                "answer": ans_data["answer"],
                "correct": ans_data["correct"],
                "fail_reason": ans_data.get("fail_reason", ""),
            })

    def add_nudge_event(self, seq: int, block: str, actions_taken: list[dict],
                        accuracy_before: float, skill_versions_after: dict):
        self.nudge_events.append({
            "after_seq": seq,
            "block": block,
            "timestamp": datetime.now().isoformat(),
            "accuracy_before_this_block": round(accuracy_before, 3),
            "actions_taken": actions_taken,
            "skill_versions_after": skill_versions_after,
        })

    def save(self, sm: SkillManager, evaluator: Evaluator):
        """写入 evolution_log.json，包含 Skill 版本历史和题目对比数据。"""
        skill_snapshots = {}
        for skill_dir in SKILLS_DIR.iterdir():
            if skill_dir.is_dir():
                name = skill_dir.name
                history = sm.get_version_history(name)
                skill_snapshots[name] = [
                    {
                        "version": h["version"],
                        "time": h["time"],
                        "action": h["action"],
                        "reason": h["reason"][:120],
                        "snapshot_file": h.get("snapshot_file", ""),
                    }
                    for h in history
                ]

        # 构建 question_comparison：所有题目的完整跨版本答案对比
        question_comparison = {}
        for qid_str, history in self.question_history.items():
            qid = int(qid_str)
            q = evaluator.questions.get(qid)
            if q:
                question_comparison[qid_str] = {
                    "question": q["question"],
                    "category": q["category"],
                    "difficulty": q["difficulty"],
                    "ground_truth": q["ground_truth"],
                    "history": history,
                }

        log = {
            "generated_at": datetime.now().isoformat(),
            "skill_snapshots": skill_snapshots,
            "eval_runs": self.eval_runs,
            "nudge_events": self.nudge_events,
            "question_comparison": question_comparison,
        }
        EVOL_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✓ evolution_log.json 已保存 ({EVOL_LOG})")


# ── 主实验流程 ────────────────────────────────────────────────────────────────

def run_experiment():
    print("=" * 60)
    print("  云购商城客服 Agent 自进化实验 v2")
    print("=" * 60)

    # 准备目录
    (ROOT / "outputs").mkdir(exist_ok=True)
    sm = SkillManager(str(SKILLS_DIR), str(VERSIONS_DIR))

    # 第一步：确保原始备份存在，然后还原
    ensure_original(sm)
    restore_from_original()

    # 还原后需要重新初始化（路径没变，但文件被替换了）
    sm = SkillManager(str(SKILLS_DIR), str(VERSIONS_DIR))
    agent = CustomerServiceAgent(sm, nudge_interval=0)  # nudge 由 demo_runner 手动触发
    reviewer = BackgroundReviewer(str(POLICIES), sm)
    evaluator = Evaluator(str(EVAL_SET))
    elog = EvolutionLog()

    # 保存初始 Skill 快照（v1）
    for skill_name, content in sm.load_all().items():
        sm._save_version(skill_name, content, action="initial", reason="初始版本")

    # 加载脚本
    script_data = json.loads(DEMO_SCRIPT.read_text(encoding="utf-8"))
    demo_questions = script_data["questions"]
    nudge_interval = script_data["nudge_interval"]
    probe_ids = script_data.get("probe_question_ids", list(range(1, 31)))

    # ── 基线评估 ─────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("基线评估（初始 Skills，无进化）")
    print("─" * 60)
    baseline = run_full_eval(agent, evaluator, "baseline", "基线（初始Skills）", sm)
    elog.add_eval_run(baseline)
    _print_summary(baseline)
    agent.conversation_history.clear()

    # ── 演示脚本 ─────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"演示脚本运行（{len(demo_questions)} 题，Nudge 间隔={nudge_interval}）")
    print("─" * 60)

    # iters 按题数计数（与 demo_script 的 "nudge_interval" 对齐）
    # block_failed_turns 仅累积本块内的失败样本，只有这些会送入 Reviewer
    iters = 0
    block_correct = 0
    block_total = 0
    block_failed_turns: list[dict] = []

    for item in demo_questions:
        seq = item["seq"]
        eval_id = item["eval_id"]
        question = item["question"]
        block = item.get("block", "")

        answer = agent.answer(question)
        ok, reason = evaluator.evaluate_answer(answer, eval_id)
        block_correct += int(ok)
        block_total += 1
        iters += 1
        if not ok:
            block_failed_turns.append({"question": question, "answer": answer, "fail_reason": reason})

        status = "✓" if ok else "✗"
        note = f"  [{item['note']}]" if item.get("note") else ""
        print(f"  Q{seq:02d} {status}  {question[:50]:<50}{note}")

        if iters >= nudge_interval:
            block_acc = block_correct / block_total if block_total else 0
            print(f"\n{'━' * 60}")
            print(f"  本块 [{block}] 完成: {block_correct}/{block_total} = {block_acc:.1%}")

            # 全对 → 直接跳过 Nudge 和 Probe（零失败无需进化）
            if not block_failed_turns:
                print(f"  ✓ 本块全对，跳过 Nudge 和 Probe eval")
                elog.add_nudge_event(seq, block, [], block_acc, sm.get_active_versions())
            else:
                print(f"  🔔 Nudge 触发（{len(block_failed_turns)} 条失败样本注入 Reviewer）")
                actions = reviewer.review(block_failed_turns)
                executed_actions = []
                for act in (actions or []):
                    try:
                        if act["action"] == "create":
                            ok_act = sm.create(act["skill_name"], act["content"], reason=act.get("reason", ""))
                        elif act["action"] == "patch":
                            ok_act = sm.patch(act["skill_name"], act["old_text"], act["new_text"],
                                              reason=act.get("reason", ""))
                        else:
                            ok_act = False
                        if ok_act:
                            executed_actions.append({"action": act["action"], "skill": act["skill_name"],
                                                     "reason": act.get("reason", "")[:80]})
                    except Exception as e:
                        print(f"  [Reviewer] 执行失败: {e}")
                print(f"  ✓ 执行了 {len(executed_actions)} 个 Skill 操作")

                probe_run_id = f"after_nudge_seq{seq}"
                probe_label = f"Nudge后（seq={seq}, block={block}）"
                probe_result = run_probe_eval(agent, evaluator, probe_ids, probe_run_id, probe_label, sm)
                elog.add_eval_run(probe_result)
                print(
                    f"  Probe eval: {probe_result['summary']['correct']}/{probe_result['summary']['total']} = {probe_result['summary']['accuracy']:.1%}")
                elog.add_nudge_event(seq, block, executed_actions, block_acc, sm.get_active_versions())

            # 重置块状态
            iters = 0
            block_correct = 0
            block_total = 0
            block_failed_turns = []
            agent.conversation_history = agent.conversation_history[-5:]
            print(f"{'━' * 60}\n")

    # ── 最终评估 ─────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("最终评估（进化后 Skills）")
    print("─" * 60)
    final = run_full_eval(agent, evaluator, "final", "最终（进化后）", sm)
    elog.add_eval_run(final)
    _print_summary(final)

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  实验结果汇总")
    print("=" * 60)
    base_acc = baseline["summary"]["accuracy"]
    final_acc = final["summary"]["accuracy"]
    print(f"  基线准确率:   {base_acc:.1%}  ({baseline['summary']['correct']}/{baseline['summary']['total']})")
    print(f"  进化后准确率: {final_acc:.1%}  ({final['summary']['correct']}/{final['summary']['total']})")
    print(f"  准确率提升:   +{(final_acc - base_acc):.1%}")

    print(f"\n  进化轨迹（probe eval 准确率）:")
    for run in elog.eval_runs:
        if run["run_id"].startswith("after_nudge"):
            print(f"    {run['label']:<35}  {run['correct']:>2}/{run['total']:>2} = {run['accuracy']:.1%}")

    print(f"\n  Skill 版本历史:")
    for name, versions in sorted(sm.get_all_version_summaries().items()):
        print(f"    {name}: {len(versions)} 个版本")
        for v in versions:
            print(f"      v{v.get('time', '')[:19]} [{v['action']}] {v['reason'][:60]}")

    elog.save(sm, evaluator)
    print(f"\n✓ 评估详情见 outputs/eval_runs/")
    print(f"✓ 各版本 Skill 见 outputs/skill_snapshots/")


def _print_summary(result: dict):
    r = result["summary"]
    print(f"总体准确率: {r['correct']}/{r['total']} = {r['accuracy']:.1%}")
    print("分类准确率:")
    for cat, stats in sorted(result["by_category"].items()):
        bar = "█" * int(stats["accuracy"] * 20)
        print(f"  {cat:<22} {stats['correct']:>2}/{stats['total']:>2}  {bar} {stats['accuracy']:.0%}")


if __name__ == "__main__":
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("错误: 请先设置 DEEPSEEK_API_KEY 环境变量")
        sys.exit(1)
    run_experiment()
