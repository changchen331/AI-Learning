"""
Memory Flush 与 Compaction — 教学核心

教学重点：
  1. Memory Flush：会话结束时，LLM 扮演"记忆提取器"角色
     - Pass 1：从对话中提取用户偏好/事实 → 更新 USER.md
     - Pass 2：提取值得长期记忆的事件/信息 → 追加到 MEMORY.md
     - Pass 3：新条目向量化 → 更新 FAISS 索引
  2. Compaction：记忆条目过多时，LLM 压缩旧条目（类似 Claude Code 的上下文压缩）
  3. 为什么用 LLM 而不是规则提取：自然语言理解、跨句推断、隐式偏好识别

使用方式：
  from src.memory_flush import MemoryFlusher
  flusher = MemoryFlusher()
  result = flusher.flush(messages, session_id=1)
  print(result.summary())

依赖：
  pip install openai faiss-cpu
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path

from src.fts_store import FTSStore
from src.llm_config import get_chat_client
from src.memory_loader import MemoryLoader
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

MEMORY_DIR = Path(__file__).parent.parent / "memory"
COMPACTION_THRESHOLD = 50  # 超过此条数触发压缩
COMPACTION_KEEP_RECENT = 20  # 压缩后保留最近 N 条不变


@dataclass
class FlushResult:
    session_id: int
    user_updates: list[str] = field(default_factory=list)
    new_memory_entries: list[dict] = field(default_factory=list)
    compacted: bool = False
    compaction_before: int = 0
    compaction_after: int = 0
    vectorized_count: int = 0
    error: str = ""

    def summary(self) -> str:
        lines = [f"=== Memory Flush 结果（会话 #{self.session_id}）==="]
        if self.error:
            lines.append(f"[错误] {self.error}")
            return "\n".join(lines)
        lines.append(f"USER.md 更新项：{len(self.user_updates)} 条")
        for u in self.user_updates:
            lines.append(f"  · {u}")
        lines.append(f"新增长期记忆：{len(self.new_memory_entries)} 条")
        for e in self.new_memory_entries:
            lines.append(f"  [{e.get('category', '?')}] {e.get('title', '')}")
        if self.compacted:
            lines.append(
                f"Compaction：{self.compaction_before} → {self.compaction_after} 条"
            )
        lines.append(f"向量化写入：{self.vectorized_count} 条")
        return "\n".join(lines)


def _chat(messages: list[dict], temperature: float = 0.1) -> str:
    client, model = get_chat_client()
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )
    return resp.choices[0].message.content.strip()


# ── Pass 1：提取用户偏好/事实，更新 USER.md ─────────────────────────────────

_USER_EXTRACT_PROMPT = """你是一个记忆提取助手。分析下面的对话，提取关于**用户**的新信息。

只提取对话中用户**明确表达**或**强烈暗示**的信息，不要推断。
格式输出为 JSON 数组，每个元素包含：
- "field"：信息所属字段（如"姓名"、"职业"、"所在地"、"偏好-XX"、"技术背景"等）
- "value"：具体内容（一句话）
- "confidence"："high" 或 "medium"（不确定的信息标 medium）

如果没有任何新信息，返回空数组 []。

对话内容：
{conversation}"""

_USER_MD_UPDATE_PROMPT = """你是一个文档更新助手。根据提取到的新信息，更新下面的 USER.md 文件。

规则：
1. 如果字段已存在且内容一致，保持不变
2. 如果字段已存在但内容更新了，替换旧内容
3. 如果是新字段，添加到对应章节下
4. "（尚未告知）"或"（暂无记录）"的占位符，如果有新信息则替换，没有则保留
5. 只输出更新后的完整 USER.md 内容，不要有任何其他说明文字

当前 USER.md：
{current_user_md}

新信息（JSON）：
{new_info}"""

# ── Pass 2：提取长期记忆条目，追加到 MEMORY.md ────────────────────────────

_MEMORY_EXTRACT_PROMPT = """你是一个记忆提取助手。分析下面的对话，提取值得**长期记忆**的信息。

长期记忆标准（满足其一即可）：
- 用户表达了明确的偏好（超出 USER.md 个人信息范畴的具体偏好）
- 发生了值得记录的事件（用户做了某件事、决定了某件事）
- 讨论了特定领域的知识（用户分享的见解、经验）
- 达成了某个约定或共识

每条记忆的格式（JSON 数组）：
- "category"：类别，取值之一：preference | fact | event | decision
- "title"：一句话标题（15字以内）
- "content"：详细内容（2~4句话）

如果没有值得记录的内容，返回空数组 []。

对话内容：
{conversation}"""

# ── Pass 3（Compaction）：压缩旧条目 ────────────────────────────────────────

_COMPACTION_PROMPT = """你是一个记忆整理助手。下面是一批旧的记忆条目，请将它们压缩合并为更简洁的摘要。

压缩规则：
1. 合并同类信息（多条偏好可合并为一条综合偏好）
2. 去掉重复或矛盾信息（保留最新的）
3. 保持信息密度，不要丢失重要内容
4. 输出格式与输入相同的 JSON 数组

待压缩的条目：
{entries}

请输出压缩后的 JSON 数组（条目数应明显少于输入）："""


class MemoryFlusher:
    def __init__(
            self,
            memory_dir: Path = MEMORY_DIR,
            compaction_threshold: int = COMPACTION_THRESHOLD,
    ):
        self.loader = MemoryLoader(memory_dir)
        self.vs = VectorStore()
        self.fts = FTSStore()
        self.compaction_threshold = compaction_threshold
        self.user_md_path = self.loader.get_user_md_path()
        self.memory_md_path = self.loader.get_memory_md_path()

    def flush(self, messages: list[dict], session_id: int) -> FlushResult:
        """
        完整 Memory Flush 流程：
          1. 对话 → 提取用户信息 → 更新 USER.md
          2. 对话 → 提取记忆条目 → 追加 MEMORY.md
          3. 新条目 → 向量化 → 更新 FAISS
          4. （可选）MEMORY.md 条目超限 → Compaction
        """
        result = FlushResult(session_id=session_id)
        conversation = self._format_conversation(messages)

        if not conversation.strip():
            result.error = "会话为空，跳过 Flush"
            return result

        try:
            # Pass 1: 更新 USER.md
            logger.info("Flush Pass 1: 提取用户信息...")
            user_updates = self._extract_and_update_user(conversation)
            result.user_updates = user_updates

            # Pass 2: 追加 MEMORY.md 条目 + 当日日志
            logger.info("Flush Pass 2: 提取长期记忆...")
            new_entries = self._extract_memory_entries(conversation)
            if new_entries:
                self._append_to_memory_md(new_entries)
                self._append_to_daily_log(new_entries, session_id)
                result.new_memory_entries = new_entries

            # Pass 3: 向量化新条目 + 同步 FTS5 全文索引
            if new_entries:
                logger.info(f"Flush Pass 3: 向量化 {len(new_entries)} 条新记忆...")
                count = self.vs.add_entries(new_entries)
                self.fts.add_entries(new_entries)
                result.vectorized_count = count

            # Compaction（如需要）
            entry_count = self.loader.get_memory_entry_count()
            if entry_count >= self.compaction_threshold:
                logger.info(f"触发 Compaction：当前 {entry_count} 条 >= 阈值 {self.compaction_threshold}")
                before, after = self._compact_memory()
                result.compacted = True
                result.compaction_before = before
                result.compaction_after = after

        except Exception as e:
            result.error = str(e)
            logger.error(f"Flush 失败: {e}", exc_info=True)

        return result

    # ── 内部实现 ──────────────────────────────────────────────────────────────

    def _format_conversation(self, messages: list[dict]) -> str:
        lines = []
        for m in messages:
            role = "用户" if m["role"] == "user" else "助手"
            lines.append(f"{role}：{m['content']}")
        return "\n".join(lines)

    def _extract_and_update_user(self, conversation: str) -> list[str]:
        """Pass 1：提取用户信息并更新 USER.md，返回更新项描述列表"""
        current_user_md = self.user_md_path.read_text(encoding="utf-8")

        # Step 1a: 提取新信息
        extract_resp = _chat([
            {"role": "user", "content": _USER_EXTRACT_PROMPT.format(conversation=conversation)}
        ])
        new_info = self._parse_json_safe(extract_resp, default=[])
        if not new_info:
            return []

        # Step 1b: 更新 USER.md
        update_resp = _chat([
            {
                "role": "user",
                "content": _USER_MD_UPDATE_PROMPT.format(
                    current_user_md=current_user_md,
                    new_info=json.dumps(new_info, ensure_ascii=False, indent=2),
                ),
            }
        ])
        # 写回文件（去掉 LLM 可能加的 markdown 代码块包裹）
        cleaned = self._strip_code_fence(update_resp)
        self.user_md_path.write_text(cleaned, encoding="utf-8")

        return [f"{item['field']}: {item['value']}" for item in new_info]

    def _extract_memory_entries(self, conversation: str) -> list[dict]:
        """Pass 2：从对话中提取长期记忆条目"""
        extract_resp = _chat([
            {"role": "user", "content": _MEMORY_EXTRACT_PROMPT.format(conversation=conversation)}
        ])
        entries = self._parse_json_safe(extract_resp, default=[])
        if not entries:
            return []

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        for entry in entries:
            entry.setdefault("date", now)
            entry.setdefault("id", f"entry_{now.replace(' ', '_').replace(':', '')}_{entries.index(entry)}")
        return entries

    def _append_to_memory_md(self, entries: list[dict]):
        """将新条目追加到 MEMORY.md 的 ENTRIES 区块内"""
        content = self.memory_md_path.read_text(encoding="utf-8")
        end_marker = "<!-- MEMORY_ENTRIES_END -->"

        new_blocks = []
        for entry in entries:
            block = (
                f"### [{entry.get('category', 'fact')}] {entry.get('title', '未命名')}\n"
                f"记录时间：{entry.get('date', '')}\n\n"
                f"{entry.get('content', '')}"
            )
            new_blocks.append(block)

        insertion = "\n\n".join(new_blocks) + "\n\n"
        updated = content.replace(end_marker, insertion + end_marker)
        self.memory_md_path.write_text(updated, encoding="utf-8")

    def _append_to_daily_log(self, entries: list[dict], session_id: int):
        """
        短期记忆（近端日志）：把本次新条目追加到 memory/YYYY-MM-DD.md。
        append-only，对应 OpenClaw 的 memory/YYYY-MM-DD.md 每日日志——
        会话启动时由 MemoryLoader 加载"今天 + 昨天"两天，给 Agent 48h 连续感。
        """
        log_path = self.loader.memory_dir / f"{date.today().isoformat()}.md"
        now = datetime.now().strftime("%H:%M")
        lines = [f"\n## {now} 会话 #{session_id}\n"]
        for e in entries:
            lines.append(
                f"- [{e.get('category', 'fact')}] {e.get('title', '')}：{e.get('content', '')}"
            )
        lines.append("")  # 结尾空行，便于下次追加
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _compact_memory(self) -> tuple[int, int]:
        """Compaction：压缩旧条目，重建 MEMORY.md 和 FAISS 索引"""
        content = self.memory_md_path.read_text(encoding="utf-8")
        start_marker = "<!-- MEMORY_ENTRIES_START -->"
        end_marker = "<!-- MEMORY_ENTRIES_END -->"
        start = content.find(start_marker)
        end = content.find(end_marker)
        body = content[start + len(start_marker):end].strip()

        entries_raw = re.split(r"(?=### \[)", body)
        entries_raw = [e.strip() for e in entries_raw if e.strip()]
        total_before = len(entries_raw)

        if total_before <= COMPACTION_KEEP_RECENT:
            return total_before, total_before

        # 保留最新 COMPACTION_KEEP_RECENT 条，压缩剩余
        old_entries = entries_raw[:-COMPACTION_KEEP_RECENT]
        recent_entries = entries_raw[-COMPACTION_KEEP_RECENT:]

        compact_resp = _chat([
            {
                "role": "user",
                "content": _COMPACTION_PROMPT.format(
                    entries=json.dumps(
                        [{"text": e} for e in old_entries],
                        ensure_ascii=False, indent=2
                    )
                ),
            }
        ])
        compacted_raw = self._parse_json_safe(compact_resp, default=[])

        # 将压缩结果转回 Markdown 格式
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        compacted_blocks = []
        for item in compacted_raw:
            if isinstance(item, dict) and "content" in item:
                block = (
                    f"### [{item.get('category', 'fact')}] {item.get('title', '压缩摘要')}\n"
                    f"记录时间：{now}（压缩整理）\n\n"
                    f"{item['content']}"
                )
            else:
                block = f"### [fact] 压缩摘要\n记录时间：{now}（压缩整理）\n\n{item}"
            compacted_blocks.append(block)

        all_blocks = compacted_blocks + recent_entries
        new_body = "\n\n".join(all_blocks)
        new_content = (
                content[:start + len(start_marker)]
                + "\n"
                + new_body
                + "\n\n"
                + content[end:]
        )
        self.memory_md_path.write_text(new_content, encoding="utf-8")

        # 重建 FAISS + FTS5：解析当前全部条目
        all_entries = self._parse_memory_entries_for_faiss()
        self.vs.rebuild_from_entries(all_entries)
        self.fts.rebuild_from_entries(all_entries)

        return total_before, len(all_blocks)

    def _parse_memory_entries_for_faiss(self) -> list[dict]:
        """从 MEMORY.md 解析所有条目用于重建向量索引"""
        content = self.memory_md_path.read_text(encoding="utf-8")
        start = content.find("<!-- MEMORY_ENTRIES_START -->")
        end = content.find("<!-- MEMORY_ENTRIES_END -->")
        if start == -1 or end == -1:
            return []
        body = content[start:end]
        blocks = re.split(r"(?=### \[)", body)
        entries = []
        for i, block in enumerate(blocks):
            if not block.strip() or not block.startswith("### ["):
                continue
            m = re.match(r"### \[(\w+)\]\s+(.+)\n记录时间：([^\n]+)\n\n([\s\S]+)", block.strip())
            if m:
                entries.append({
                    "id": f"compact_{i}",
                    "category": m.group(1),
                    "title": m.group(2),
                    "date": m.group(3),
                    "content": f"{m.group(2)}: {m.group(4).strip()}",
                })
        return entries

    @staticmethod
    def _parse_json_safe(text: str, default):
        """容错 JSON 解析：去掉 markdown 代码块，提取第一个 JSON 数组"""
        text = MemoryFlusher._strip_code_fence(text)
        # 尝试找 [...] 数组
        m = re.search(r"\[[\s\S]*\]", text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return default

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """去掉 LLM 输出中的 ```markdown 或 ```json 包裹"""
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
        text = re.sub(r"\n?```$", "", text.strip())
        return text.strip()
