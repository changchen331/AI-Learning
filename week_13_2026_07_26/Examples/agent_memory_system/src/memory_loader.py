"""
Layer 3 长期记忆：Markdown 文件加载与 System Prompt 组装

教学重点：
  1. Markdown 作为"记忆配置语言"：SOUL / USER / MEMORY / AGENTS.md 各司其职
  2. System Prompt 的分层拼接：人格 → 用户画像 → 操作规范 → 近期记忆
  3. Token 意识：每层注入多少 token，如何控制总量

使用方式：
  from src.memory_loader import MemoryLoader
  loader = MemoryLoader()
  result = loader.build_system_prompt(recent_memory_limit=10)
  print(result["system_prompt"])
  print(result["layers"])   # 每层来源与 token 数
"""

import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

MEMORY_DIR = Path(__file__).parent.parent / "memory"


@dataclass
class MemoryLayer:
    name: str
    source_file: str
    content: str
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.content)


@dataclass
class SystemPromptResult:
    system_prompt: str
    layers: list[MemoryLayer]
    total_chars: int = 0

    def __post_init__(self):
        self.total_chars = sum(l.char_count for l in self.layers)


class MemoryLoader:
    def __init__(self, memory_dir: Path = MEMORY_DIR):
        self.memory_dir = memory_dir

    def _read_md(self, filename: str) -> str:
        path = self.memory_dir / filename
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def _extract_memory_entries(self, memory_md: str, limit: int) -> str:
        """从 MEMORY.md 提取最近 N 条记忆条目"""
        start = memory_md.find("<!-- MEMORY_ENTRIES_START -->")
        end = memory_md.find("<!-- MEMORY_ENTRIES_END -->")
        if start == -1 or end == -1:
            return ""
        body = memory_md[start + len("<!-- MEMORY_ENTRIES_START -->"):end].strip()
        if not body:
            return ""
        # 按 ### 分割条目，取最近 limit 条
        entries = re.split(r"(?=### \[)", body)
        entries = [e.strip() for e in entries if e.strip()]
        recent = entries[-limit:] if len(entries) > limit else entries
        return "\n\n".join(recent)

    def _read_recent_day_logs(self, days: int = 2) -> tuple[str, list[str]]:
        """
        短期记忆（近端日志）：读取最近 N 天的 memory/YYYY-MM-DD.md。
        对应 OpenClaw 的每日日志——加载"今天 + 昨天"给 Agent 48h 连续感。
        返回 (拼接正文, 命中的文件名列表)。
        """
        today = date.today()
        parts, sources = [], []
        for i in range(days):
            d = today - timedelta(days=i)
            p = self.memory_dir / f"{d.isoformat()}.md"
            if p.exists():
                text = p.read_text(encoding="utf-8").strip()
                if text:
                    parts.append(f"### {d.isoformat()}\n{text}")
                    sources.append(p.name)
        return ("\n\n".join(parts)).strip(), sources

    def build_system_prompt(self, recent_memory_limit: int = 10) -> SystemPromptResult:
        """
        四层拼接：SOUL → USER → AGENTS → 近期 MEMORY 条目
        返回完整 system_prompt 文本和每层明细。
        """
        layers: list[MemoryLayer] = []
        parts: list[str] = []

        # Layer 3a: SOUL.md — 人格基调
        soul = self._read_md("SOUL.md")
        if soul:
            layers.append(MemoryLayer("soul", "SOUL.md", soul))
            parts.append(soul)

        # Layer 2 近端：每日日志（今天 + 昨天）—— OpenClaw 风格的短期记忆
        log_body, log_sources = self._read_recent_day_logs(days=2)
        if log_body:
            section = f"## 近期日志（今天 + 昨天）\n\n{log_body}"
            layers.append(MemoryLayer("daily_log", ", ".join(log_sources), section))
            parts.append(section)

        # Layer 3b: USER.md — 用户画像
        user = self._read_md("USER.md")
        if user:
            section = f"## 关于当前用户\n{user}"
            layers.append(MemoryLayer("user_profile", "USER.md", section))
            parts.append(section)

        # Layer 3c: AGENTS.md — 操作规范
        agents = self._read_md("AGENTS.md")
        if agents:
            layers.append(MemoryLayer("agents_manual", "AGENTS.md", agents))
            parts.append(agents)

        # Layer 3d: MEMORY.md 近期条目
        memory_md = self._read_md("MEMORY.md")
        memory_entries = self._extract_memory_entries(memory_md, recent_memory_limit)
        if memory_entries:
            section = f"## 长期记忆（最近 {recent_memory_limit} 条）\n\n{memory_entries}"
            layers.append(MemoryLayer("long_term_memory", "MEMORY.md", section))
            parts.append(section)

        system_prompt = "\n\n---\n\n".join(parts)
        return SystemPromptResult(system_prompt=system_prompt, layers=layers)

    def get_memory_entry_count(self) -> int:
        """返回当前 MEMORY.md 中的条目数量"""
        memory_md = self._read_md("MEMORY.md")
        return len(re.findall(r"^### \[", memory_md, re.MULTILINE))

    def get_user_md_path(self) -> Path:
        return self.memory_dir / "USER.md"

    def get_memory_md_path(self) -> Path:
        return self.memory_dir / "MEMORY.md"
