"""
市场调研 Subagent HTTP 服务（FastAPI + SSE 流式）

教学重点：
  SSE 逐事件推送，前端实时看到：
  - 主 agent 的 ReAct 每步（Thought/Action/Observation）
  - 派发 subagent 时拓扑加节点
  - 各 subagent 并行 ReAct 步骤
  - 最终报告 + 并行加速统计

启动：
  uvicorn src.serve:app --host 0.0.0.0 --port 8002
  浏览器开 http://localhost:8002

依赖：pip install fastapi uvicorn
"""
import json
import logging
import os
import queue
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app):
    logger.info("市场调研 subagent 服务就绪")
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    has_tavily = bool(os.getenv("TAVILY_API_KEY"))
    has_llm = bool(os.getenv("DEEPSEEK_API_KEY"))
    return {"status": "ok", "tavily": has_tavily, "llm": has_llm}


@app.post("/query")
def query(req: QueryRequest):
    """SSE 流式：主 agent + 各 subagent 的 ReAct 步骤逐事件推。"""
    import agents

    def event_stream():
        q = queue.Queue()
        SENTINEL = object()

        def push(ev):
            q.put(ev)

        def on_main_step(step):
            push({"type": "main_step", **step})

        def on_dispatch(info):
            push({"type": "dispatch", **info})

        def on_subagent_step(sid, step):
            push({"type": "subagent_step", "subagent_id": sid, **step})

        def on_subagent_done(sid, duration, topic):
            push({"type": "subagent_done", "subagent_id": sid,
                  "duration": duration, "subtopic": topic})

        def run():
            try:
                r = agents.run_research(
                    req.question,
                    on_main_step=on_main_step,
                    on_dispatch=on_dispatch,
                    on_subagent_step=on_subagent_step,
                    on_subagent_done=on_subagent_done,
                )
                push({"type": "final", "answer": r["final_answer"],
                      "parallel_stats": r["parallel_stats"],
                      "main_trace_len": len(r["main_trace"]),
                      "subagent_count": len(r["subagents"])})
            except Exception as e:
                push({"type": "error", "message": f"{type(e).__name__}: {str(e)[:200]}"})
            finally:
                push(SENTINEL)

        threading.Thread(target=run, daemon=True).start()

        # 先发 start
        yield "data: " + json.dumps({"type": "start", "question": req.question},
                                    ensure_ascii=False) + "\n\n"
        while True:
            ev = q.get()
            if ev is SENTINEL:
                yield "data: " + json.dumps({"type": "done"}, ensure_ascii=False) + "\n\n"
                break
            yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve:app", host="0.0.0.0", port=8002, reload=False)
