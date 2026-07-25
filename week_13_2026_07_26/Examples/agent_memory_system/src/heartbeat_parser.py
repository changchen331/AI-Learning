"""
HEARTBEAT.md 解析器 + 调度意图检测

教学重点：
  1. 正则 pattern 初筛：快速判断消息是否含调度意图，避免每条消息都调 LLM
  2. LLM 二次判断：pattern 命中后，让模型决定是否真的是定时任务请求
  3. 结构化写入：把 LLM 输出的任务定义追加到 HEARTBEAT.md 的 TASKS 块

使用方式：
  from src.heartbeat_parser import HeartbeatParser
  parser = HeartbeatParser()
  tasks = parser.load_tasks()
  if parser.may_contain_schedule_intent(msg):
      task = parser.analyze_and_write(msg)
"""

import json
import logging
import re
from datetime import date
from pathlib import Path

from week_13_2026_07_26.Examples.agent_memory_system.src.llm_config import get_chat_client

logger = logging.getLogger(__name__)

HEARTBEAT_PATH = Path(__file__).parent.parent / "memory" / "HEARTBEAT.md"

# ── 正则初筛：命中任意一条才走 LLM ───────────────────────────────────────────
SCHEDULE_PATTERNS = [
    re.compile(r'每天|每日'),
    re.compile(r'每周|每星期'),
    re.compile(r'每月'),
    re.compile(r'每\s*\d+\s*(分钟|小时|天|周)'),
    re.compile(r'定期|定时'),
    re.compile(r'提醒(我|一下)'),
    re.compile(r'帮我.*(记得|记住|定时|提醒)'),
    re.compile(r'自动.*(帮|给|整理|汇总|总结|提醒)'),
    re.compile(r'(早上|晚上|每天).*(提醒|告诉|汇报|说一下)'),
    re.compile(r'设置.*(提醒|任务|闹钟)'),
]

SUPPORTED_ACTIONS = ["send_message", "summarize_sessions", "compact_memory", "user_profile_refresh"]

# ── 取消意图正则（与新建意图互斥，优先检测取消）────────────────────────────────
CANCEL_PATTERNS = [
    re.compile(r'取消.*(提醒|任务|定时|通知)'),
    re.compile(r'停止.*(提醒|任务|定时|发消息)'),
    re.compile(r'关闭.*(提醒|任务|定时)'),
    re.compile(r'删除.*(任务|提醒|定时)'),
    re.compile(r'禁用.*(任务|提醒)'),
    re.compile(r'不(要|用).*(提醒|任务|通知|发消息|发送)'),
    re.compile(r'别.*(提醒|发|通知|打扰)'),
    re.compile(r'(停|关)掉.*(提醒|任务|定时)'),
]

_INTENT_PROMPT = """\
用户说："{message}"

判断这是否是一个定时任务请求（希望 Agent 定期自动执行某件事）。

如果是，返回如下 JSON（trigger 使用标准 cron 5字段表达式）：
{{
  "is_schedule": true,
  "name": "snake_case任务名",
  "trigger": "cron表达式",
  "action": "从以下选一个: send_message / summarize_sessions / compact_memory / user_profile_refresh",
  "description": "一句话描述任务",
  "prompt": "仅 send_message 时填写，LLM 生成消息时的指令，其他 action 留空字符串"
}}

如果不是定时任务请求，返回：{{"is_schedule": false}}

只返回 JSON，不要有其他文字。"""


class HeartbeatParser:
    def __init__(self, path: Path = HEARTBEAT_PATH):
        self.path = path

    # ── 解析任务列表 ──────────────────────────────────────────────────────────

    def load_tasks(self) -> list[dict]:
        """从 HEARTBEAT.md 解析所有启用的任务，返回 list[dict]"""
        content = self.path.read_text(encoding="utf-8")
        start = content.find("<!-- TASKS_START -->")
        end = content.find("<!-- TASKS_END -->")
        if start == -1 or end == -1:
            return []

        body = content[start + len("<!-- TASKS_START -->"):end].strip()
        if not body:
            return []

        tasks = []
        blocks = re.split(r"(?=### TASK:)", body)
        for block in blocks:
            block = block.strip()
            if not block.startswith("### TASK:"):
                continue
            task = self._parse_task_block(block)
            if task and task.get("enabled", True):
                tasks.append(task)
        return tasks

    def _parse_task_block(self, block: str) -> dict | None:
        lines = block.splitlines()
        name_line = lines[0]
        name = name_line.replace("### TASK:", "").strip()
        task = {"name": name}
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                task[key.strip()] = val.strip()
        if "trigger" not in task or "action" not in task:
            return None
        task["enabled"] = task.get("enabled", "true").lower() == "true"
        return task

    # ── 意图检测 ──────────────────────────────────────────────────────────────

    def load_all_tasks(self) -> list[dict]:
        """解析所有任务（含 enabled=false 的），用于取消时展示完整列表"""
        content = self.path.read_text(encoding="utf-8")
        start = content.find("<!-- TASKS_START -->")
        end = content.find("<!-- TASKS_END -->")
        if start == -1 or end == -1:
            return []
        body = content[start + len("<!-- TASKS_START -->"):end].strip()
        tasks = []
        for block in re.split(r"(?=### TASK:)", body):
            block = block.strip()
            if not block.startswith("### TASK:"):
                continue
            task = self._parse_task_block(block)
            if task:
                tasks.append(task)
        return tasks

    def may_contain_schedule_intent(self, message: str) -> bool:
        """正则初筛，True 表示可能含调度意图，需要进一步 LLM 判断"""
        return any(p.search(message) for p in SCHEDULE_PATTERNS)

    def may_contain_cancel_intent(self, message: str) -> bool:
        """正则初筛，True 表示可能含取消/停止任务意图"""
        return any(p.search(message) for p in CANCEL_PATTERNS)

    def analyze_and_cancel(self, message: str) -> str | None:
        """
        LLM 判断用户想取消哪个任务。
        是 → 在 HEARTBEAT.md 中将该任务设为 enabled: false，返回任务名；
        否或无匹配 → 返回 None。
        """
        tasks = self.load_all_tasks()
        enabled_tasks = [t for t in tasks if t.get("enabled", True)]
        if not enabled_tasks:
            return None

        task_list = "\n".join(
            f"- {t['name']}：{t.get('description', t.get('action', ''))}"
            for t in enabled_tasks
        )
        cancel_prompt = (
            f'用户说："{message}"\n\n'
            f"当前已启用的定时任务：\n{task_list}\n\n"
            "判断用户想停止哪个任务。如果能明确对应，返回 JSON：\n"
            '{"cancel": true, "name": "任务名"}\n'
            "如果无法判断或用户不是要取消任务，返回：\n"
            '{"cancel": false}\n\n'
            "只返回 JSON，不要其他文字。"
        )
        try:
            client, model = get_chat_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": cancel_prompt}],
                temperature=0,
            )
            data = self._parse_json_safe(resp.choices[0].message.content.strip())
            if not data or not data.get("cancel"):
                return None
            task_name = data.get("name", "").strip()
            if not task_name:
                return None
            self._disable_task(task_name)
            return task_name
        except Exception as e:
            logger.error(f"取消意图分析失败：{e}")
            return None

    def analyze_and_write(self, message: str) -> dict | None:
        """
        LLM 判断是否为调度任务请求。
        是 → 写入 HEARTBEAT.md，返回任务 dict；
        否 → 返回 None。
        """
        try:
            client, model = get_chat_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _INTENT_PROMPT.format(message=message)}],
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            data = self._parse_json_safe(raw)
            if not data or not data.get("is_schedule"):
                return None
            if data.get("action") not in SUPPORTED_ACTIONS:
                logger.warning(f"LLM 返回了不支持的 action：{data.get('action')}，跳过")
                return None
            self._append_task(data)
            return data
        except Exception as e:
            logger.error(f"调度意图分析失败：{e}")
            return None

    # ── 写入任务 ──────────────────────────────────────────────────────────────

    def _append_task(self, task: dict):
        """将新任务追加到 HEARTBEAT.md 的 TASKS 块末尾"""
        content = self.path.read_text(encoding="utf-8")
        end_marker = "<!-- TASKS_END -->"
        block_lines = [
            f"### TASK: {task['name']}",
            f"trigger: {task['trigger']}",
            "enabled: true",
            f"action: {task['action']}",
            f"description: {task.get('description', '')}",
        ]
        if task.get("prompt"):
            block_lines.append(f"prompt: {task['prompt']}")
        block_lines.append(f"added: {date.today().isoformat()}")
        block = "\n".join(block_lines) + "\n"
        updated = content.replace(end_marker, block + "\n" + end_marker)
        self.path.write_text(updated, encoding="utf-8")
        logger.info(f"新任务已写入 HEARTBEAT.md：{task['name']}")

    def _disable_task(self, task_name: str):
        """将指定任务的 enabled 改为 false"""
        content = self.path.read_text(encoding="utf-8")
        # 找到对应 TASK 块，把第一个 "enabled: true" 改为 "enabled: false"
        pattern = re.compile(
            r"(### TASK:\s*" + re.escape(task_name) + r".*?)(enabled:\s*true)",
            re.DOTALL,
        )
        new_content, count = pattern.subn(r"\1enabled: false", content, count=1)
        if count == 0:
            logger.warning(f"未找到任务 {task_name}，无法禁用")
            return
        self.path.write_text(new_content, encoding="utf-8")
        logger.info(f"任务已禁用：{task_name}")

    @staticmethod
    def _parse_json_safe(text: str) -> dict | None:
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
        text = re.sub(r"\n?```$", "", text.strip())
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return None
