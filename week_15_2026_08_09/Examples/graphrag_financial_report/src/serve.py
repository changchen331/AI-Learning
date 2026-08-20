"""
GraphRAG HTTP 服务（FastAPI）

教学重点（对应 CLAUDE.md 第五节）：
  1. lifespan 复用：GraphRAG 检索器（含节点向量缓存）启动时加载一次，
     每次请求复用，不在请求里重建索引
  2. 两个接口：/query 标准问答、/query/debug 逐步返回每个中间环节
  3. /query/debug 必须返回 final_context（喂给 LLM 的原始文本）——
     这是教学调试接口的核心，让学生看清 LLM 收到了什么

启动：
  uvicorn serve:app --host 0.0.0.0 --port 8000
  然后浏览器开 http://localhost:8000

依赖：
  pip install fastapi uvicorn
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from retrieve import GraphRAGLocal, GraphRAGGlobal
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = None

# 全局检索器（lifespan 里初始化）
local_rag: Optional[GraphRAGLocal] = None
global_rag: Optional[GraphRAGGlobal] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载检索器一次，请求复用。"""
    global local_rag, global_rag
    logger.info("加载 GraphRAG 检索器...")
    local_rag = GraphRAGLocal()
    # Global 可选（需要 community_detect 已跑过）
    try:
        global_rag = GraphRAGGlobal()
    except FileNotFoundError as e:
        logger.warning(f"Global Search 不可用（{e}），仅 Local 模式")
    logger.info("GraphRAG 就绪")
    yield
    local_rag.close()
    if global_rag:
        global_rag.close()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    mode: str = "local"  # "local" | "global"


@app.get("/health")
def health():
    """健康检查 + 图库统计。"""
    try:
        drv = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        with drv.session() as s:
            n_nodes = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            n_edges = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        drv.close()
        return {"status": "ok", "neo4j": "connected",
                "nodes": n_nodes, "edges": n_edges,
                "global_ready": global_rag is not None}
    except Exception as e:
        return {"status": "degraded", "neo4j": str(e)[:80]}


@app.post("/query")
def query(req: QueryRequest):
    """标准问答，返回 {answer, context}。"""
    if req.mode == "global":
        if not global_rag:
            return {"answer": "Global 模式未就绪，请先运行 community_detect.py", "context": ""}
        r = global_rag.search(req.question)
        return {"answer": r["answer"], "context": r.get("picked_communities", "")}
    r = local_rag.search(req.question)
    return {"answer": r["answer"], "context": r["context"]}


@app.post("/query/debug")
def query_debug(req: QueryRequest):
    """
    教学调试接口：逐步返回每个中间环节。
    必须包含 final_context（喂给 LLM 的原始文本）——默认在页面展开。
    """
    if req.mode == "global":
        if not global_rag:
            return {"answer": "Global 模式未就绪", "final_context": "",
                    "steps": [{"step": "error", "data": "先运行 community_detect.py"}]}
        r = global_rag.search(req.question)
        return {
            "answer": r["answer"],
            "final_context": "\n\n".join(m["partial"] for m in r["map_results"]),
            "steps": [
                {"step": "pick_communities", "data": r["picked_communities"]},
                {"step": "map", "data": r["map_results"]},
                {"step": "reduce", "data": r["answer"]},
            ],
        }
    # Local
    r = local_rag.search(req.question)
    return {
        "answer": r["answer"],
        "final_context": r["context"],  # ← 喂给 LLM 的原始文本，页面默认展开
        "steps": [
            {"step": "entity_linking", "data": r["linked_entities"]},
            {"step": "subgraph", "data": r["subgraph"]},
            {"step": "context_build", "data": r["context"]},
        ],
    }


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)
