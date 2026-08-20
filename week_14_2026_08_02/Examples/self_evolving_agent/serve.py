"""
教学演示服务：步进式 Agent 自进化 UI

启动：
  cd self_evolving_agent
  uvicorn serve:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import shutil
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from skill_manager import SkillManager
from evaluator import Evaluator
from agent import CustomerServiceAgent
from background_reviewer import BackgroundReviewer

SKILLS_DIR = ROOT / "skills"
SKILLS_ORIG = ROOT / "outputs" / "skills_original"
VERSIONS_DIR = ROOT / "outputs" / "skill_versions"
EVAL_RUNS_DIR = ROOT / "outputs" / "eval_runs"
EVAL_SET = ROOT / "data" / "eval_set.json"
DEMO_SCRIPT = ROOT / "data" / "demo_script.json"
POLICIES = ROOT / "data" / "policies.md"

_s: dict = {}  # global experiment state


# ── 初始化 / 还原 ─────────────────────────────────────────────────────────────

def _restore():
    if SKILLS_ORIG.exists():
        if SKILLS_DIR.exists():
            shutil.rmtree(SKILLS_DIR)
        shutil.copytree(SKILLS_ORIG, SKILLS_DIR)
    for d in [VERSIONS_DIR, ROOT / "outputs" / "skill_snapshots", EVAL_RUNS_DIR]:
        if d.exists():
            shutil.rmtree(d)


def _init():
    global _s
    sm = SkillManager(str(SKILLS_DIR), str(VERSIONS_DIR))
    ev = Evaluator(str(EVAL_SET))
    agent = CustomerServiceAgent(sm, nudge_interval=0)
    reviewer = BackgroundReviewer(str(POLICIES), sm)
    script = json.loads(DEMO_SCRIPT.read_text(encoding="utf-8"))
    qs = script["questions"]
    blocks = [qs[i:i + 10] for i in range(0, len(qs), 10)]
    _s = {
        "phase": "idle",
        "current_block": 0,
        "sm": sm, "ev": ev, "agent": agent, "reviewer": reviewer,
        "blocks": blocks,
        "probe_ids": script.get("probe_question_ids", []),
        "eval_results": {},
        "conv_history": [],
        "nudge_count": 0,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    (ROOT / "outputs").mkdir(exist_ok=True)
    if not SKILLS_ORIG.exists():
        shutil.copytree(SKILLS_DIR, SKILLS_ORIG)
    _init()
    yield


app = FastAPI(lifespan=lifespan)

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Access-Control-Allow-Origin": "*",
}


def _evt(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── 静态 / 状态接口 ───────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(ROOT / "index.html")


@app.get("/state")
async def get_state():
    blocks = _s.get("blocks", [])
    return JSONResponse({
        "phase": _s["phase"],
        "current_block": _s["current_block"],
        "total_blocks": len(blocks),
        "nudge_count": _s["nudge_count"],
        "block_names": [b[0]["block"] for b in blocks],
        "eval_results": {
            k: {"accuracy": v["accuracy"], "correct": v["correct"], "total": v["total"],
                "by_category": v["by_category"]}
            for k, v in _s["eval_results"].items()
        },
    })


@app.get("/skills")
async def get_skills():
    sm = _s["sm"]
    result = {}
    for name, content in sorted(sm.load_all().items()):
        history = sm.get_version_history(name)
        result[name] = {
            "content": content,
            "version_count": len(history),
            "history": [
                {"version": h.get("version", i + 1), "time": h["time"][:19],
                 "action": h["action"], "reason": h["reason"][:120]}
                for i, h in enumerate(history)
            ],
        }
    return JSONResponse(result)


@app.get("/skill_version/{name}/{version}")
async def get_skill_version(name: str, version: int):
    history = _s["sm"].get_version_history(name)
    for h in history:
        if h.get("version") == version:
            return JSONResponse({
                "content": h["content"], "action": h["action"],
                "reason": h["reason"], "time": h["time"][:19],
            })
    return JSONResponse({"error": "not found"}, status_code=404)


@app.post("/reset")
async def reset():
    _restore()
    _init()
    return {"status": "ok"}


# ── 工具函数 ──────────────────────────────────────────────────────────────────

async def _probe(ev, agent, sm, probe_ids: list[int], run_id: str) -> dict:
    """在线程里跑 probe eval，返回结果 dict（不逐题流式）。"""

    def _sync():
        by_cat: dict = {}
        correct = 0
        for qid in probe_ids:
            q = ev.questions[qid]
            answer = agent.answer(q["question"])
            ok, _ = ev.evaluate_answer(answer, qid)
            if ok:
                correct += 1
            cat = q["category"]
            by_cat.setdefault(cat, {"total": 0, "correct": 0})
            by_cat[cat]["total"] += 1
            if ok:
                by_cat[cat]["correct"] += 1
        for v in by_cat.values():
            v["accuracy"] = round(v["correct"] / v["total"], 3)
        total = len(probe_ids)
        return {
            "run_id": run_id,
            "accuracy": round(correct / total, 3),
            "correct": correct,
            "total": total,
            "by_category": by_cat,
            "timestamp": datetime.now().isoformat(),
            "skill_versions_active": sm.get_active_versions(),
        }

    result = await asyncio.to_thread(_sync)
    _s["eval_results"][run_id] = result
    return result


# ── SSE 流式接口 ──────────────────────────────────────────────────────────────

@app.get("/stream/baseline")
async def stream_baseline():
    async def gen():
        sm = _s["sm"]
        ev = _s["ev"]
        agent = _s["agent"]
        run_id = "baseline"
        _s["phase"] = "baseline_running"

        # 保存初始快照
        for name, content in sm.load_all().items():
            sm._save_version(name, content, action="initial", reason="初始版本")

        all_ids = sorted(ev.questions.keys())
        by_cat: dict = {}
        correct = 0

        for qid in all_ids:
            q = ev.questions[qid]
            answer = await asyncio.to_thread(agent.answer, q["question"])
            ok, reason = ev.evaluate_answer(answer, qid)
            if ok:
                correct += 1
            cat = q["category"]
            by_cat.setdefault(cat, {"total": 0, "correct": 0})
            by_cat[cat]["total"] += 1
            if ok:
                by_cat[cat]["correct"] += 1
            yield _evt({"type": "eval_q", "run_id": run_id, "id": qid,
                        "correct": ok, "category": cat,
                        "question": q["question"][:60], "answer": answer[:200],
                        "fail_reason": reason if not ok else ""})

        for v in by_cat.values():
            v["accuracy"] = round(v["correct"] / v["total"], 3)

        result = {
            "run_id": "baseline",
            "accuracy": round(correct / len(all_ids), 3),
            "correct": correct,
            "total": len(all_ids),
            "by_category": by_cat,
            "timestamp": datetime.now().isoformat(),
            "skill_versions_active": sm.get_active_versions(),
        }
        _s["eval_results"]["baseline"] = result
        agent.conversation_history.clear()
        _s["phase"] = "baseline_done"

        yield _evt({"type": "eval_complete", "run_id": "baseline",
                    "correct": correct, "total": len(all_ids),
                    "accuracy": result["accuracy"], "by_category": by_cat})
        yield _evt({"type": "phase_change", "phase": "baseline_done"})
        yield _evt({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/stream/block/{block_index}")
async def stream_block(block_index: int):
    async def gen():
        if block_index != _s["current_block"]:
            yield _evt({"type": "error", "message": f"需要运行第 {_s['current_block']} 块"})
            return

        sm = _s["sm"]
        ev = _s["ev"]
        agent = _s["agent"]
        reviewer = _s["reviewer"]
        block_qs = _s["blocks"][block_index]
        block_name = block_qs[0].get("block", f"block_{block_index}")
        _s["phase"] = f"block_{block_index}_running"

        # ── Phase 1: 逐题回答 + 累积失败样本 ─────────────────────────────────
        block_correct = 0
        block_failed_turns: list[dict] = []
        for item in block_qs:
            q = ev.questions.get(item["eval_id"], {})
            yield _evt({"type": "question_start", "seq": item["seq"],
                        "question": item["question"], "block": block_name,
                        "eval_id": item["eval_id"]})
            answer = await asyncio.to_thread(agent.answer, item["question"])
            ok, reason = ev.evaluate_answer(answer, item["eval_id"])
            if ok:
                block_correct += 1
            else:
                block_failed_turns.append({
                    "question": item["question"], "answer": answer, "fail_reason": reason,
                })
            _s["conv_history"].append({"question": item["question"], "answer": answer})
            agent.conversation_history = _s["conv_history"][-30:]
            yield _evt({"type": "question_result", "seq": item["seq"],
                        "answer": answer, "correct": ok,
                        "fail_reason": reason if not ok else "",
                        "category": q.get("category", "")})

        block_acc = block_correct / len(block_qs)
        yield _evt({"type": "block_complete", "block": block_name,
                    "correct": block_correct, "total": len(block_qs),
                    "accuracy": round(block_acc, 3)})

        # ── Phase 2: 全对则跳过进化，否则只把失败样本送 Reviewer ─────────────
        if not block_failed_turns:
            yield _evt({"type": "nudge_skipped", "block": block_name,
                        "reason": "本块全部答对，跳过 Nudge 和 Probe eval"})
        else:
            _s["nudge_count"] += 1
            yield _evt({"type": "nudge_start",
                        "nudge_num": _s["nudge_count"],
                        "block": block_name,
                        "failure_count": len(block_failed_turns)})

            actions = await asyncio.to_thread(reviewer.review, block_failed_turns)
            analysis = getattr(reviewer, "last_analysis", "")
            yield _evt({"type": "reviewer_analysis", "analysis": analysis})

            executed = []
            for act in (actions or []):
                try:
                    if act["action"] == "create":
                        ok_act = sm.create(act["skill_name"], act["content"],
                                           reason=act.get("reason", ""))
                    elif act["action"] == "patch":
                        ok_act = sm.patch(act["skill_name"], act["old_text"],
                                          act["new_text"], reason=act.get("reason", ""))
                    else:
                        ok_act = False
                    if ok_act:
                        executed.append({
                            "action": act["action"],
                            "skill_name": act["skill_name"],
                            "reason": act.get("reason", "")[:120],
                        })
                        yield _evt({"type": "skill_action", **executed[-1]})
                except Exception as e:
                    yield _evt({"type": "skill_error", "error": str(e)[:100]})

            yield _evt({"type": "nudge_complete", "num_actions": len(executed)})

            # Phase 3: Probe eval（仅在触发了 Nudge 时跑）
            yield _evt({"type": "probe_start", "total": len(_s["probe_ids"])})
            run_id = f"after_block_{block_index}"
            result = await _probe(ev, agent, sm, _s["probe_ids"], run_id)
            yield _evt({"type": "probe_result", "run_id": run_id,
                        "correct": result["correct"], "total": result["total"],
                        "accuracy": result["accuracy"],
                        "by_category": result["by_category"],
                        "skill_versions": result["skill_versions_active"]})

        # ── 推进状态 ─────────────────────────────────────────────────────────
        _s["current_block"] += 1
        is_last = (_s["current_block"] >= len(_s["blocks"]))
        _s["phase"] = "all_blocks_done" if is_last else f"block_{block_index}_done"
        agent.conversation_history = agent.conversation_history[-5:]

        yield _evt({"type": "phase_change", "phase": _s["phase"],
                    "current_block": _s["current_block"]})
        yield _evt({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/stream/final")
async def stream_final():
    async def gen():
        sm = _s["sm"]
        ev = _s["ev"]
        agent = _s["agent"]
        run_id = "final"
        _s["phase"] = "final_running"

        all_ids = sorted(ev.questions.keys())
        by_cat: dict = {}
        correct = 0

        for qid in all_ids:
            q = ev.questions[qid]
            answer = await asyncio.to_thread(agent.answer, q["question"])
            ok, reason = ev.evaluate_answer(answer, qid)
            if ok:
                correct += 1
            cat = q["category"]
            by_cat.setdefault(cat, {"total": 0, "correct": 0})
            by_cat[cat]["total"] += 1
            if ok:
                by_cat[cat]["correct"] += 1
            yield _evt({"type": "eval_q", "run_id": run_id, "id": qid,
                        "correct": ok, "category": cat,
                        "question": q["question"][:60], "answer": answer[:200],
                        "fail_reason": reason if not ok else ""})

        for v in by_cat.values():
            v["accuracy"] = round(v["correct"] / v["total"], 3)

        result = {
            "run_id": "final",
            "accuracy": round(correct / len(all_ids), 3),
            "correct": correct,
            "total": len(all_ids),
            "by_category": by_cat,
            "timestamp": datetime.now().isoformat(),
            "skill_versions_active": sm.get_active_versions(),
        }
        _s["eval_results"]["final"] = result
        _s["phase"] = "complete"

        yield _evt({"type": "eval_complete", "run_id": "final",
                    "correct": correct, "total": len(all_ids),
                    "accuracy": result["accuracy"], "by_category": by_cat})
        yield _evt({"type": "phase_change", "phase": "complete"})
        yield _evt({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)
