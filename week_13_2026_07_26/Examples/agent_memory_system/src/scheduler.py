"""
HEARTBEAT 调度器 — APScheduler + 任务执行 + 广播

教学重点：
  1. APScheduler AsyncIOScheduler：在 FastAPI 的同一 event loop 内运行，无需额外线程
  2. 任务执行：每个 action 对应一段 LLM 调用逻辑，结果通过 broadcast 推送前端
  3. 热重载：HEARTBEAT.md 被写入新任务后，调度器自动重新加载（每60秒检测一次文件修改时间）

依赖：
  pip install apscheduler
"""

import asyncio
import logging
from datetime import datetime
from typing import Callable, Awaitable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from week_13_2026_07_26.Examples.agent_memory_system.src.heartbeat_parser import HeartbeatParser, HEARTBEAT_PATH
from week_13_2026_07_26.Examples.agent_memory_system.src.llm_config import get_chat_client
from week_13_2026_07_26.Examples.agent_memory_system.src.memory_loader import MemoryLoader
from week_13_2026_07_26.Examples.agent_memory_system.src.session_db import SessionDB

logger = logging.getLogger(__name__)

BroadcastFn = Callable[[str, dict], Awaitable[None]]


class HeartbeatScheduler:
    def __init__(self):
        self._scheduler = AsyncIOScheduler()
        self._parser = HeartbeatParser()
        self._broadcast: BroadcastFn | None = None
        self._last_mtime: float = 0

    # ── 启动 / 停止 ───────────────────────────────────────────────────────────

    def start(self, broadcast: BroadcastFn):
        """启动调度器，broadcast 是向前端推送消息的协程函数"""
        self._broadcast = broadcast
        self._load_tasks()
        # 每60秒检测 HEARTBEAT.md 是否被修改，有改动则热重载
        self._scheduler.add_job(
            self._check_reload,
            trigger="interval",
            seconds=60,
            id="_heartbeat_watcher",
        )
        self._scheduler.start()
        logger.info("HeartbeatScheduler 已启动")

    def stop(self):
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    # ── 任务加载 ──────────────────────────────────────────────────────────────

    def _load_tasks(self):
        """从 HEARTBEAT.md 加载任务，清除旧 jobs 后重新注册"""
        # 保留内部监控 job
        for job in self._scheduler.get_jobs():
            if not job.id.startswith("_"):
                job.remove()

        tasks = self._parser.load_tasks()
        for task in tasks:
            self._register_task(task)

        if HEARTBEAT_PATH.exists():
            self._last_mtime = HEARTBEAT_PATH.stat().st_mtime
        logger.info(f"已加载 {len(tasks)} 个 HEARTBEAT 任务")

    def _register_task(self, task: dict):
        try:
            parts = task["trigger"].split()
            if len(parts) != 5:
                raise ValueError(f"trigger 格式错误，需要5字段 cron：{task['trigger']}")
            minute, hour, day, month, day_of_week = parts
            trigger = CronTrigger(
                minute=minute, hour=hour, day=day,
                month=month, day_of_week=day_of_week,
            )
            self._scheduler.add_job(
                self._execute_task,
                trigger=trigger,
                args=[task],
                id=task["name"],
                replace_existing=True,
                misfire_grace_time=300,
            )
            logger.info(f"已注册任务：{task['name']}  trigger={task['trigger']}")
        except Exception as e:
            logger.error(f"注册任务 {task.get('name')} 失败：{e}")

    async def _check_reload(self):
        if not HEARTBEAT_PATH.exists():
            return
        mtime = HEARTBEAT_PATH.stat().st_mtime
        if mtime > self._last_mtime:
            logger.info("HEARTBEAT.md 已更新，重新加载任务...")
            self._load_tasks()
            await self._broadcast("heartbeat_reloaded", {
                "message": "HEARTBEAT.md 已更新，任务重新加载",
                "task_count": len(self._parser.load_tasks()),
            })

    # ── 任务执行 ──────────────────────────────────────────────────────────────

    async def _execute_task(self, task: dict):
        name = task["name"]
        action = task.get("action", "send_message")
        logger.info(f"执行 HEARTBEAT 任务：{name}（action={action}）")

        await self._broadcast("heartbeat_start", {
            "task_name": name,
            "action": action,
            "description": task.get("description", ""),
            "fired_at": datetime.now().isoformat(),
        })

        try:
            if action == "send_message":
                await self._action_send_message(task)
            elif action == "summarize_sessions":
                await self._action_summarize_sessions(task)
            elif action == "compact_memory":
                await self._action_compact_memory(task)
            elif action == "user_profile_refresh":
                await self._action_user_profile_refresh(task)
            else:
                logger.warning(f"未知 action：{action}")
        except Exception as e:
            logger.error(f"任务 {name} 执行失败：{e}", exc_info=True)
            await self._broadcast("heartbeat_error", {
                "task_name": name,
                "error": str(e),
            })

    # ── action 实现 ───────────────────────────────────────────────────────────

    async def _action_send_message(self, task: dict):
        """LLM 根据用户画像和 prompt 生成主动消息，推送到前端"""
        loader = MemoryLoader()
        user_md = loader.get_user_md_path().read_text(encoding="utf-8")
        prompt = task.get("prompt") or "根据用户画像，生成一条简短友好的主动问候或提醒。"

        system = (
            "你是一个有记忆的 AI 助手，现在主动向用户发送一条消息。\n"
            f"用户画像：\n{user_md}"
        )
        client, model = await asyncio.get_event_loop().run_in_executor(None, get_chat_client)
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
            )
        )
        message = resp.choices[0].message.content.strip()
        await self._broadcast("heartbeat_message", {
            "task_name": task["name"],
            "description": task.get("description", ""),
            "message": message,
            "fired_at": datetime.now().strftime("%H:%M"),
        })

    async def _action_summarize_sessions(self, task: dict):
        """汇总近期对话，写入 MEMORY.md [event] 条目"""
        db = SessionDB()
        messages = db.get_today_messages()
        if not messages:
            await self._broadcast("heartbeat_message", {
                "task_name": task["name"],
                "description": task.get("description", ""),
                "message": "📋 今日暂无对话记录，跳过汇总。",
                "fired_at": datetime.now().strftime("%H:%M"),
            })
            return

        conversation = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}：{m['content']}"
            for m in messages[-40:]  # 最多取40条
        )
        client, model = await asyncio.get_event_loop().run_in_executor(None, get_chat_client)
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content":
                    f"请用3~5句话总结以下对话的主要内容和要点：\n\n{conversation}"}],
                temperature=0.3,
            )
        )
        summary = resp.choices[0].message.content.strip()

        # 写入 MEMORY.md
        from week_13_2026_07_26.Examples.agent_memory_system.src.memory_flush import MemoryFlusher
        flusher = MemoryFlusher()
        flusher._append_to_memory_md([{
            "category": "event",
            "title": f"对话汇总 {datetime.now().strftime('%Y-%m-%d')}",
            "content": summary,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "id": f"summary_{datetime.now().strftime('%Y%m%d%H%M')}",
        }])
        await self._broadcast("heartbeat_message", {
            "task_name": task["name"],
            "description": task.get("description", ""),
            "message": f"📋 对话已汇总并写入记忆：\n{summary}",
            "fired_at": datetime.now().strftime("%H:%M"),
        })

    async def _action_compact_memory(self, task: dict):
        """触发 Memory Compaction"""
        from week_13_2026_07_26.Examples.agent_memory_system.src.memory_flush import MemoryFlusher
        loader = MemoryLoader()
        count = loader.get_memory_entry_count()
        if count < 5:
            await self._broadcast("heartbeat_message", {
                "task_name": task["name"],
                "description": task.get("description", ""),
                "message": f"🗜️ 当前记忆条目仅 {count} 条，无需压缩。",
                "fired_at": datetime.now().strftime("%H:%M"),
            })
            return
        flusher = MemoryFlusher()
        before, after = await asyncio.get_event_loop().run_in_executor(
            None, flusher._compact_memory
        )
        await self._broadcast("heartbeat_message", {
            "task_name": task["name"],
            "description": task.get("description", ""),
            "message": f"🗜️ Memory Compaction 完成：{before} → {after} 条",
            "fired_at": datetime.now().strftime("%H:%M"),
        })

    async def _action_user_profile_refresh(self, task: dict):
        """重新分析全部记忆，刷新 USER.md"""
        loader = MemoryLoader()
        memory_md = loader.get_memory_md_path().read_text(encoding="utf-8")
        user_md = loader.get_user_md_path().read_text(encoding="utf-8")

        client, model = await asyncio.get_event_loop().run_in_executor(None, get_chat_client)
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content":
                    f"根据以下长期记忆，更新用户画像文件 USER.md。\n"
                    f"只输出更新后的完整 USER.md 内容，不要其他说明。\n\n"
                    f"当前 USER.md：\n{user_md}\n\n长期记忆：\n{memory_md}"}],
                temperature=0.1,
            )
        )
        new_user_md = resp.choices[0].message.content.strip()
        new_user_md = re.sub(r"^```[a-zA-Z]*\n?", "", new_user_md)
        new_user_md = re.sub(r"\n?```$", "", new_user_md).strip()
        loader.get_user_md_path().write_text(new_user_md, encoding="utf-8")

        await self._broadcast("heartbeat_message", {
            "task_name": task["name"],
            "description": task.get("description", ""),
            "message": "👤 USER.md 已根据最新记忆刷新。",
            "fired_at": datetime.now().strftime("%H:%M"),
        })


import re  # noqa: E402（放在文件末尾供 user_profile_refresh 用）
