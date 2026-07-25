"""
FastAPI HTTP 服务 — 带 SSE 事件流的可视化后端

教学重点：
  1. SSE 事件流将四层记忆的每个步骤实时推送到前端
  2. /flush 接口展示 Memory Flush 三个 Pass 的进度
  3. lifespan 模式：索引/DB 在启动时加载一次，请求间复用

使用方式：
  uvicorn src.serve:app --host 0.0.0.0 --port 8000

接口：
  POST /chat     SSE 流式对话（含四层记忆事件）
  POST /flush    SSE 流式 Memory Flush
  GET  /memories 查看当前记忆状态
  GET  /health   健康检查

依赖：
  pip install fastapi uvicorn openai faiss-cpu
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import asyncio
import json
import logging
import os
import sqlite3
import sys
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel

from week_13_2026_07_26.Examples.agent_memory_system.src.session_db import SessionDB
from week_13_2026_07_26.Examples.agent_memory_system.src.memory_loader import MemoryLoader
from week_13_2026_07_26.Examples.agent_memory_system.src.vector_store import VectorStore
from week_13_2026_07_26.Examples.agent_memory_system.src.fts_store import FTSStore
from week_13_2026_07_26.Examples.agent_memory_system.src.retrieval import HybridRetriever
from week_13_2026_07_26.Examples.agent_memory_system.src.memory_flush import MemoryFlusher
from week_13_2026_07_26.Examples.agent_memory_system.src.llm_config import get_chat_client, current_model_info
from week_13_2026_07_26.Examples.agent_memory_system.src.heartbeat_parser import HeartbeatParser
from week_13_2026_07_26.Examples.agent_memory_system.src.scheduler import HeartbeatScheduler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── 全局单例 ─────────────────────────────────────────────────────────────────

db: SessionDB = None
loader: MemoryLoader = None
vs: VectorStore = None
fts: FTSStore = None
retriever: HybridRetriever = None
flusher: MemoryFlusher = None
current_session_id: int = None
hb_parser: HeartbeatParser = None
hb_scheduler: HeartbeatScheduler = None

# ── SSE 广播：每个连接一个 Queue，调度器触发时 broadcast 推给所有连接 ──────────
_stream_listeners: list[asyncio.Queue] = []


async def broadcast(event_type: str, data: dict):
    payload = sse_event(event_type, data)
    logger.info(f"[broadcast] {event_type}，当前监听数：{len(_stream_listeners)}")
    for q in list(_stream_listeners):  # 拷贝一份，避免迭代时被修改
        try:
            await q.put(payload)
        except asyncio.QueueFull:
            logger.warning("[broadcast] 队列已满，丢弃一条消息")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, loader, vs, fts, retriever, flusher, current_session_id, hb_parser, hb_scheduler
    db = SessionDB()
    loader = MemoryLoader()
    vs = VectorStore()
    fts = FTSStore()
    retriever = HybridRetriever(vs, fts)
    flusher = MemoryFlusher()
    hb_parser = HeartbeatParser()
    current_session_id = db.new_session()
    logger.info(f"服务启动，会话 #{current_session_id}")
    logger.info(f"FTS5/BM25 可用：{fts.available}（{'混合检索' if fts.available else '退化为纯向量'})")

    hb_scheduler = HeartbeatScheduler()
    hb_scheduler.start(broadcast)
    logger.info("HEARTBEAT 调度器已启动")

    yield

    hb_scheduler.stop()
    if current_session_id:
        db.close_session(current_session_id)


app = FastAPI(title="Agent 记忆系统", lifespan=lifespan)


# ── 请求/响应模型 ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: int = None  # None = 使用当前会话


class FlushRequest(BaseModel):
    session_id: int = None


# ── SSE 工具函数 ──────────────────────────────────────────────────────────────

def sse_event(event_type: str, data: dict) -> str:
    payload = json.dumps({"type": event_type, **data}, ensure_ascii=False)
    return f"data: {payload}\n\n"


# ── /chat 接口 ───────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    sid = req.session_id or current_session_id

    async def stream():
        # Step 1: Layer 3 加载
        prompt_result = loader.build_system_prompt(recent_memory_limit=10)
        layers_info = [
            {"name": l.name, "source": l.source_file, "chars": l.char_count}
            for l in prompt_result.layers
        ]
        yield sse_event("memory_load", {
            "layers": layers_info,
            "total_chars": prompt_result.total_chars,
        })
        await asyncio.sleep(0)

        # Step 2: Layer 4 混合检索（向量 0.7 + BM25 0.3）
        semantic_results = retriever.search(req.message, top_k=3)
        yield sse_event("semantic_search", {
            "query": req.message,
            "results": [
                {
                    "category": r.get("category", ""),
                    "title": r.get("title", ""),
                    "content": r.get("content", "")[:120],
                    "score": round(r["score"], 3),
                    "source": r.get("source", ""),
                }
                for r in semantic_results
            ],
        })
        await asyncio.sleep(0)

        # Step 3: 组装 Context
        semantic_context = ""
        if semantic_results:
            snippets = [f"- [{r['category']}] {r.get('title', '')}: {r['content'][:100]}" for r in semantic_results]
            semantic_context = "## 语义检索到的相关记忆\n" + "\n".join(snippets)

        system_prompt = prompt_result.system_prompt
        if semantic_context:
            system_prompt += "\n\n" + semantic_context

        # Layer 2：会话历史
        history = db.get_session_messages(sid)
        history_for_api = [{"role": m["role"], "content": m["content"]} for m in history]

        yield sse_event("context_assembly", {
            "system_chars": len(system_prompt),
            "history_turns": len(history_for_api),
            "layers_used": [l["name"] for l in layers_info] + (["faiss_semantic"] if semantic_results else []),
        })
        await asyncio.sleep(0)

        # Step 4: LLM 流式生成
        api_messages = (
                [{"role": "system", "content": system_prompt}]
                + history_for_api
                + [{"role": "user", "content": req.message}]
        )
        client, model = get_chat_client()
        stream_resp = client.chat.completions.create(
            model=model, messages=api_messages, temperature=0.7, stream=True
        )
        full_response = ""
        for chunk in stream_resp:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_response += delta
                yield sse_event("token", {"text": delta})

        # Step 5: 保存到 DB
        db.add_message(sid, "user", req.message)
        db.add_message(sid, "assistant", full_response)

        msg_count = db.get_message_count(sid)
        yield sse_event("done", {
            "response": full_response,
            "session_id": sid,
            "message_count": msg_count,
            "auto_flush_threshold": 20,
        })

        # Step 6: 后台检测调度意图（不阻塞响应）
        if hb_parser and hb_parser.may_contain_schedule_intent(req.message):
            asyncio.create_task(_check_schedule_intent(req.message))

    return StreamingResponse(stream(), media_type="text/event-stream")


async def _check_schedule_intent(message: str):
    """后台任务：检测新建/取消调度意图，更新 HEARTBEAT.md 并重载调度器"""
    loop = asyncio.get_event_loop()

    # 优先检测取消意图（取消 pattern 更明确，不会误判）
    if hb_parser.may_contain_cancel_intent(message):
        task_name = await loop.run_in_executor(None, hb_parser.analyze_and_cancel, message)
        if task_name:
            hb_scheduler._load_tasks()
            await broadcast("heartbeat_task_cancelled", {
                "task_name": task_name,
                "message": f"🚫 已停止定时任务：{task_name}",
            })
            return  # 取消和新建互斥，不继续检测新建意图

    # 检测新建意图
    if hb_parser.may_contain_schedule_intent(message):
        task = await loop.run_in_executor(None, hb_parser.analyze_and_write, message)
        if task:
            hb_scheduler._load_tasks()
            await broadcast("heartbeat_task_added", {
                "task_name": task["name"],
                "trigger": task["trigger"],
                "description": task.get("description", ""),
                "message": f"✅ 已为你设置定时任务：{task.get('description', task['name'])}",
            })


# ── /flush 接口 ──────────────────────────────────────────────────────────────

@app.post("/flush")
async def flush_session(req: FlushRequest):
    sid = req.session_id or current_session_id

    async def stream():
        messages = db.get_session_messages(sid)
        yield sse_event("flush_start", {
            "session_id": sid,
            "message_count": len(messages),
        })
        await asyncio.sleep(0)

        if not messages:
            yield sse_event("flush_done", {"error": "会话为空"})
            return

        # 在线程池中跑同步的 flush（避免阻塞 event loop）
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, flusher.flush, messages, sid)

        yield sse_event("flush_pass1", {
            "user_updates": result.user_updates,
            "count": len(result.user_updates),
        })
        await asyncio.sleep(0)

        yield sse_event("flush_pass2", {
            "new_entries": [
                {"category": e.get("category", ""), "title": e.get("title", ""), "content": e.get("content", "")[:100]}
                for e in result.new_memory_entries
            ],
            "count": len(result.new_memory_entries),
        })
        await asyncio.sleep(0)

        yield sse_event("flush_pass3", {
            "vectorized": result.vectorized_count,
            "total_in_index": vs.total_entries,
        })
        await asyncio.sleep(0)

        if result.compacted:
            yield sse_event("flush_compaction", {
                "before": result.compaction_before,
                "after": result.compaction_after,
            })
            await asyncio.sleep(0)

        db.mark_flushed(sid)
        yield sse_event("flush_done", {
            "error": result.error,
            "summary": result.summary(),
        })

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── /memories 接口 ────────────────────────────────────────────────────────────

@app.get("/memories")
async def get_memories():
    mem_dir = loader.memory_dir

    def read_md(name):
        p = mem_dir / name
        return p.read_text(encoding="utf-8") if p.exists() else ""

    return JSONResponse({
        "user_md": read_md("USER.md"),
        "memory_md": read_md("MEMORY.md"),
        "soul_md": read_md("SOUL.md"),
        "agents_md": read_md("AGENTS.md"),
        "heartbeat_md": read_md("HEARTBEAT.md"),
        "entry_count": loader.get_memory_entry_count(),
        "faiss_total": vs.total_entries,
        "fts_total": fts.total_entries,
        "fts_available": fts.available,
        "recent_sessions": db.get_recent_sessions(5),
    })


# ── /stream 接口：持久 SSE，接收调度器广播 ───────────────────────────────────

@app.get("/stream")
async def stream_events():
    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    _stream_listeners.append(q)
    logger.info(f"[/stream] 新连接，当前监听数：{len(_stream_listeners)}")

    async def generate():
        try:
            tasks = hb_parser.load_tasks() if hb_parser else []
            yield sse_event("heartbeat_connected", {
                "task_count": len(tasks),
                "tasks": [{"name": t["name"], "trigger": t["trigger"],
                           "description": t.get("description", "")} for t in tasks],
            })
            while True:
                try:
                    # 最多等 20 秒，超时发 keepalive 注释，防止连接被服务端或代理关闭
                    payload = await asyncio.wait_for(q.get(), timeout=20.0)
                    yield payload
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            if q in _stream_listeners:
                _stream_listeners.remove(q)
            logger.info(f"[/stream] 连接断开，剩余监听数：{len(_stream_listeners)}")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 关闭 Nginx 缓冲（如有反向代理）
        },
    )


# ── /reset 接口 ──────────────────────────────────────────────────────────────

@app.post("/reset")
async def reset_to_factory():
    """回到出厂初始态：重置全部 md 文件、清空 FAISS、清空 SQLite、重载调度器"""
    global current_session_id
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _do_factory_reset)
    # 文件已重置，让调度器重新读取 HEARTBEAT.md，清除所有旧 job
    if hb_scheduler:
        hb_scheduler._load_tasks()
    current_session_id = db.new_session()
    return {"status": "ok", "session_id": current_session_id}


def _do_factory_reset():
    """
    完整出厂重置，直接操作，不依赖外部脚本：
      1. 重写全部 5 个 memory/*.md（含 SOUL / AGENTS）
      2. 删除 FAISS 索引文件，重置 vs 内存状态
      3. 清空 SQLite 数据（DELETE 而非 DROP，保留 schema）
      4. 重新建表（db._init_db）确保 schema 完整
    """
    from week_13_2026_07_26.Examples.agent_memory_system.src.reset import INITIAL_USER_MD, INITIAL_MEMORY_MD, \
        INITIAL_HEARTBEAT_MD

    mem_dir = loader.memory_dir

    # ① 重写 md 文件（USER / MEMORY / HEARTBEAT 用常量，SOUL / AGENTS 从 backups/initial/ 恢复）
    (mem_dir / "USER.md").write_text(INITIAL_USER_MD, encoding="utf-8")
    (mem_dir / "MEMORY.md").write_text(INITIAL_MEMORY_MD, encoding="utf-8")
    (mem_dir / "HEARTBEAT.md").write_text(INITIAL_HEARTBEAT_MD, encoding="utf-8")

    project_root = mem_dir.parent
    for fname in ("SOUL.md", "AGENTS.md"):
        backup_src = project_root / "backups" / "initial" / "memory" / fname
        if backup_src.exists():
            (mem_dir / fname).write_text(backup_src.read_text(encoding="utf-8"), encoding="utf-8")

    # ② 清空 FAISS（删文件 + 重置内存对象）
    for f in ("memory.faiss", "memory_meta.pkl"):
        p = vs.index_dir / f
        if p.exists():
            p.unlink()
    vs.index = None
    vs.metadata = []

    # ③ 清空 SQLite（不删文件，直接 DELETE 保留 schema）
    conn = sqlite3.connect(db.db_path)
    conn.executescript("DELETE FROM messages; DELETE FROM sessions;")
    try:
        conn.execute("DELETE FROM memory_fts")  # P1 引入的 FTS5 全文索引表
    except sqlite3.OperationalError:
        pass  # FTS5 不可用时该表不存在，跳过
    conn.commit()
    conn.close()

    # ④ 确保 schema 完整（防止极端情况下 schema 丢失）
    db._init_db()


# ── /session/new 接口 ─────────────────────────────────────────────────────────

@app.post("/session/new")
async def new_session():
    global current_session_id
    if current_session_id:
        db.close_session(current_session_id)
    current_session_id = db.new_session()
    return {"session_id": current_session_id}


# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "session_id": current_session_id,
        "memory_entries": loader.get_memory_entry_count(),
        "faiss_entries": vs.total_entries,
        "fts_entries": fts.total_entries,
        "fts_available": fts.available,
        "model": current_model_info(),
    }


# ── 静态文件（index.html）────────────────────────────────────────────────────

INDEX_HTML = Path(__file__).parent.parent / "index.html"


@app.get("/")
async def root():
    return FileResponse(INDEX_HTML)
