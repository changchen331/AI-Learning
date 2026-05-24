"""
多 LLM 故障转移演示服务（断路器版）

教学重点：
  1. 断路器模式（Circuit Breaker）：CLOSED / OPEN / HALF_OPEN 三态机
  2. 主动健康探针：后台 asyncio 任务，不依赖用户请求触发恢复
  3. 被动检测：请求失败时累计计数，达阈值后自动断路
  4. 快速失败（Fail Fast）：OPEN 状态直接跳过，不等待 API 超时

使用方式：
  export DASHSCOPE_API_KEY="sk-xxx"
  export DEEPSEEK_API_KEY="sk-xxx"
  uvicorn src.serve:app --host 0.0.0.0 --port 8000

依赖：
  pip install fastapi uvicorn openai
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── 断路器状态机 ───────────────────────────────────────────────────────────────

class CircuitState(str, Enum):
    CLOSED = "CLOSED"  # 正常，所有请求放行
    OPEN = "OPEN"  # 断路，直接快速失败，不发真实请求
    HALF_OPEN = "HALF_OPEN"  # 探针状态，放行一条请求试探恢复


@dataclass
class CircuitBreaker:
    """
    断路器核心实现。

    状态转移规则：
      CLOSED    --[连续失败 >= failure_threshold]──────> OPEN
      OPEN      --[冷却时间 >= recovery_timeout]────────> HALF_OPEN  ← 由 can_attempt() 触发
      HALF_OPEN --[探针成功 >= success_threshold]───────> CLOSED
      HALF_OPEN --[探针失败]────────────────────────────> OPEN（重置冷却计时）

    关键设计：OPEN 状态下请求直接被拒绝（快速失败），不等待 API 超时，
    保护下游不被雪崩请求拖垮，同时让调用方快速切换到备用 Provider。
    """
    name: str
    # 生产推荐值：failure_threshold=3~5, recovery_timeout=60~120s, success_threshold=2
    failure_threshold: int = 3  # 连续失败几次后断路
    recovery_timeout: int = 30  # 断路后等多少秒进入 HALF_OPEN（Demo 用 30s）
    success_threshold: int = 1  # HALF_OPEN 探针连续成功几次才真正关闭（Demo 用 1）

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0

    def can_attempt(self) -> bool:
        """判断是否允许发出请求。OPEN 状态时检查冷却时间，到期自动迁移到 HALF_OPEN。"""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"[{self.name}] 断路器 OPEN → HALF_OPEN（冷却 {elapsed:.0f}s）")
                return True
            return False
        # HALF_OPEN：放行探针请求
        # NOTE: 高并发时多条请求可能同时通过，生产中需加 in-flight 锁
        return True

    def on_success(self):
        """请求成功时通知断路器。"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.info(f"[{self.name}] HALF_OPEN 探针成功 {self.success_count}/{self.success_threshold}")
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"[{self.name}] 断路器 HALF_OPEN → CLOSED（已恢复）")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # 成功请求重置连续失败计数

    def on_failure(self):
        """请求失败时通知断路器。"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning(f"[{self.name}] HALF_OPEN 探针失败 → OPEN（重新冷却）")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"[{self.name}] 连续失败 {self.failure_count} 次 → OPEN（断路）")

    @property
    def recovery_in(self) -> float:
        """OPEN 状态下距离进入 HALF_OPEN 还有多少秒。"""
        if self.state != CircuitState.OPEN:
            return 0.0
        return max(0.0, self.recovery_timeout - (time.time() - self.last_failure_time))


# ── Provider 配置 ──────────────────────────────────────────────────────────────

PROVIDERS = [
    {
        "id": "dashscope",
        "name": "百炼 (DashScope)",
        "model": "qwen-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "simulated_fail": False,
        "cb": CircuitBreaker(name="dashscope"),
    },
    {
        "id": "deepseek",
        "name": "DeepSeek",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "simulated_fail": False,
        "cb": CircuitBreaker(name="deepseek"),
    },
]


# ── API 调用封装 ───────────────────────────────────────────────────────────────

def _call_sync(p: dict, messages: list, max_tokens: int = None) -> str:
    """
    同步调用单个 Provider（供 run_in_executor 使用）。
    NOTE: 生产中应缓存 OpenAI client 实例，避免每次重建。
    """
    api_key = os.environ.get(p["api_key_env"])
    if not api_key:
        raise RuntimeError(f"环境变量 {p['api_key_env']} 未设置")
    client = OpenAI(api_key=api_key, base_url=p["base_url"])
    kwargs = dict(model=p["model"], messages=messages)
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


# ── 主动健康探针 ───────────────────────────────────────────────────────────────

PROBE_INTERVAL = 15  # 秒（Demo 用 15s，生产用 30~60s）
PROBE_MESSAGES = [{"role": "user", "content": "hi"}]


async def probe_provider(p: dict):
    """对单个 Provider 发轻量探针请求，更新断路器状态。"""
    cb: CircuitBreaker = p["cb"]
    if not cb.can_attempt():
        return
    try:
        logger.info(f"[{p['name']}] 主动探针（断路器: {cb.state}）")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, partial(_call_sync, p, PROBE_MESSAGES, 5))
        cb.on_success()
    except Exception as e:
        cb.on_failure()
        logger.warning(f"[{p['name']}] 探针失败: {e}")


async def health_probe_loop():
    """
    后台常驻任务：定期对处于 OPEN 状态的 Provider 发探针。

    为什么只探 OPEN 状态，不探 CLOSED：
      - CLOSED 状态通过真实用户请求感知故障（被动检测），无需额外消耗 token
      - OPEN 状态没有用户请求到达，必须主动探针才能触发恢复
    """
    logger.info(f"健康探针后台任务已启动，间隔 {PROBE_INTERVAL}s")
    while True:
        await asyncio.sleep(PROBE_INTERVAL)
        for p in PROVIDERS:
            if p["simulated_fail"]:
                continue
            cb: CircuitBreaker = p["cb"]
            if cb.state == CircuitState.OPEN:
                await probe_provider(p)


# ── FastAPI 应用 ───────────────────────────────────────────────────────────────

HTML_PATH = Path(__file__).parent.parent / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(health_probe_loop())
    yield
    task.cancel()


app = FastAPI(title="LLM Failover Demo", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    history: list = []


@app.get("/")
async def index():
    return FileResponse(HTML_PATH)


@app.get("/providers")
async def get_providers():
    result = []
    for p in PROVIDERS:
        cb: CircuitBreaker = p["cb"]
        result.append({
            "id": p["id"],
            "name": p["name"],
            "model": p["model"],
            "has_key": bool(os.environ.get(p["api_key_env"])),
            "simulated_fail": p["simulated_fail"],
            "circuit_state": cb.state,
            "failure_count": cb.failure_count,
            "failure_threshold": cb.failure_threshold,
            "recovery_in": round(cb.recovery_in),
            # available = 不在维护模式 且 断路器不是 OPEN
            "available": not p["simulated_fail"] and cb.state != CircuitState.OPEN,
        })
    return result


@app.post("/providers/{provider_id}/simulate-fail")
async def simulate_fail(provider_id: str):
    """模拟人工维护/下线（立即跳过，不影响断路器状态）。"""
    for p in PROVIDERS:
        if p["id"] == provider_id:
            p["simulated_fail"] = True
            logger.info(f"[{p['name']}] 维护模式开启")
            return {"status": "ok"}
    return JSONResponse(status_code=404, content={"error": "not found"})


@app.post("/providers/{provider_id}/trip-circuit")
async def trip_circuit(provider_id: str):
    """
    直接将断路器置为 OPEN（演示用途）。
    等价于真实场景中连续调用失败 failure_threshold 次后的结果。
    """
    for p in PROVIDERS:
        if p["id"] == provider_id:
            cb: CircuitBreaker = p["cb"]
            cb.state = CircuitState.OPEN
            cb.failure_count = cb.failure_threshold
            cb.last_failure_time = time.time()
            logger.info(f"[{p['name']}] 手动触发断路器 OPEN（演示），{cb.recovery_timeout}s 后探针")
            return {"status": "ok", "recovery_in": cb.recovery_timeout}
    return JSONResponse(status_code=404, content={"error": "not found"})


@app.post("/providers/{provider_id}/restore")
async def restore_provider(provider_id: str):
    """恢复 Provider：清除维护模式并重置断路器为 CLOSED。"""
    for p in PROVIDERS:
        if p["id"] == provider_id:
            p["simulated_fail"] = False
            cb: CircuitBreaker = p["cb"]
            cb.state = CircuitState.CLOSED
            cb.failure_count = 0
            cb.success_count = 0
            logger.info(f"[{p['name']}] 已恢复，断路器重置 CLOSED")
            return {"status": "ok"}
    return JSONResponse(status_code=404, content={"error": "not found"})


@app.post("/providers/reset")
async def reset_all():
    for p in PROVIDERS:
        p["simulated_fail"] = False
        cb: CircuitBreaker = p["cb"]
        cb.state = CircuitState.CLOSED
        cb.failure_count = 0
        cb.success_count = 0
    logger.info("所有 Provider 已重置")
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest):
    messages = [{"role": "system", "content": "You are a helpful assistant. 请用中文回答。"}]
    for msg in req.history[-20:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": req.message})

    tried = []

    for p in PROVIDERS:
        cb: CircuitBreaker = p["cb"]

        # 检查 1：人工维护模式（立即跳过）
        if p["simulated_fail"]:
            tried.append({"provider": p["name"], "reason": "维护中"})
            continue

        # 检查 2：断路器（快速失败，不发真实请求）
        if not cb.can_attempt():
            tried.append({
                "provider": p["name"],
                "reason": f"断路器 OPEN（{round(cb.recovery_in)}s 后探针）",
            })
            continue

        # 检查 3：API Key
        api_key = os.environ.get(p["api_key_env"])
        if not api_key:
            tried.append({"provider": p["name"], "reason": "无 API Key"})
            continue

        # 发起真实调用
        try:
            logger.info(f"[{p['name']}] 调用中（断路器: {cb.state}）")
            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(None, partial(_call_sync, p, messages))
            cb.on_success()
            logger.info(f"[{p['name']}] 调用成功")
            return {
                "answer": content,
                "provider": p["name"],
                "provider_id": p["id"],
                "tried": tried,
            }
        except Exception as e:
            cb.on_failure()
            logger.error(f"[{p['name']}] 调用失败（已失败 {cb.failure_count}/{cb.failure_threshold} 次）: {e}")
            tried.append({
                "provider": p["name"],
                "reason": f"API 错误（失败 {cb.failure_count}/{cb.failure_threshold} 次）",
            })

    return JSONResponse(
        status_code=503,
        content={"error": "所有 LLM 提供商均不可用", "tried": tried},
    )
