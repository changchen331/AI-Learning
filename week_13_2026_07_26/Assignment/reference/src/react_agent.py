"""
原生 Skills ReAct Agent —— 渐进式加载（LLM 主动 read_skill）+ ReAct 循环

教学重点（核心三步）：
  Phase 1  预加载 meta：系统提示只塞各 skill 的 meta 列表（名字+描述，~几十 tokens）。
           完整 SKILL.md **不**预先注入 —— 留给 LLM 自己决定何时读。
  Phase 2  LLM 主动 read_skill：LLM 在 ReAct 里发出内置动作 read_skill(name=...)
           才把该 skill 的完整 SKILL.md 取回作为 Observation。这是「渐进」的真正落点
           —— 读不读、读哪个、何时读，都由 LLM 自己判断，而非 agent 一把梭塞进去。
  ReAct    Thought/Action/Observation 循环，stop=["Observation:"] 让模型在 Action 后停下，
           执行完把 Observation 回喂继续下一轮，直到 Final Answer。

模型：DeepSeek（deepseek-v4-flash，即 deepseek-chat），OpenAI 兼容接口。
"""
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

from skill_executor import SkillExecutor, parse_action
from skill_registry import SkillRegistry

# ── LLM 配置：DeepSeek，OpenAI 兼容 ──────────────────────────────────
MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=BASE_URL)

# 让 src/ 同目录可相互 import（python src/react_agent.py 直接跑）
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 内置动作名：读取 skill 完整说明（不是业务 skill，由 agent 自己处理，不进 executor）
READ_SKILL = "read_skill"

# ── 提示词模板（刻意简短，教学用）────────────────────────────────────
SYSTEM_PROMPT = """你是一个 Skills Agent，通过 ReAct（Thought/Action/Observation）循环完成任务。

可用 skill 的轻量列表（仅名字+描述，完整说明需用 read_skill 主动读取）：

{metas}

内置动作（用于渐进式加载）：
  read_skill(name="skill名")
    → 返回该 skill 的完整 SKILL.md。**在调用任何业务 skill 之前，必须先 read_skill
      读取它的完整说明**，然后严格按其流程与参数执行。一次只读一个。

业务 skill 调用格式（每轮只调一个 Action）：
  Thought: <你的分析>
  Action: skill-name(key=value, ...)
- skill 名与上方列表一致（含连字符，如 flash-card）。
- 参数用 Python 字面量：字符串加引号、字典/列表用字面量。
- 若 skill 要你「自己写出数据」（如 flash-card 的音标/释义/3 条例句/近义词），
  你必须把完整数据作为参数传入（如 data={{...}}），脚本只渲染、不补全内容。

read_skill 后严格按读到的说明分两类执行，不要对文本型 skill 强行发 Action：
  - 脚本型 skill（说明要求运行 scripts/ 下的脚本，如 flash-card）→ 按 Action 格式调用。
  - 文本型 skill（说明只要求你直接回应用户，如『告诉用户…』『还没做好』）
    → 直接输出 Final Answer 按说明回答用户，**不要**调用任何脚本 Action。

Action 之后换行即停，等系统返回 Observation: ... 再继续下一轮。
任务全部完成后输出：Final Answer: <给用户的最终回答（含产物路径）>"""


def _llm(messages, stop=None, temperature=0) -> str:
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=temperature, stop=stop,
    )
    return resp.choices[0].message.content or ""


def _print_box(title: str, body: str):
    print(f"\n{'=' * 60}\n{title}\n{'-' * 60}\n{body}\n{'=' * 60}")


class SkillsReActAgent:
    def __init__(self):
        self.registry = SkillRegistry()
        self.executor = SkillExecutor(self.registry)
        self._read: set[str] = set()  # 已 read_skill 过的 skill，用于「未读先调」拦截

    def run(self, query: str, max_steps: int = 8) -> str:
        print(f"\n{'#' * 60}\n# 用户请求：{query}\n{'#' * 60}")

        # Phase 1：系统提示只放 meta —— 完整 SKILL.md 一字未进上下文
        metas = self.registry.meta_summary()
        _print_box("Phase 1 · 预加载 meta（仅名字+描述，全文待 LLM 主动读取）", metas)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(metas=metas)},
            {"role": "user", "content": query},
        ]

        # ReAct 循环：Phase 2（read_skill）与业务调用都发生在这里
        for step in range(1, max_steps + 1):
            raw = _llm(messages, stop=["Observation:"])
            messages.append({"role": "assistant", "content": raw})
            _print_box(f"ReAct Step {step} · LLM 输出", raw)

            # 终止判定
            m = re.search(r"Final Answer:\s*(.+)", raw, re.DOTALL)
            if m:
                answer = m.group(1).strip()
                self._maybe_open_html(answer)
                _print_box("完成", answer)
                return answer

            # 解析 Action
            am = re.search(r"Action:\s*(.+)", raw, re.DOTALL)
            if not am:
                messages.append({"role": "user", "content":
                    "请输出 Action: read_skill(name=...) 或业务 skill 调用，或 Final Answer:"})
                continue
            action_text = am.group(1).strip()  # data={...} 可能跨多行，不能只取首行
            sname, params = parse_action(action_text)

            # ── 内置动作 read_skill：返回完整 SKILL.md（渐进加载第二阶段，LLM 主动）──
            if sname == READ_SKILL:
                # 兼容 read_skill("flash-card") 与 read_skill(name="flash-card") 两种写法
                target = (params.get("name")
                          or (params.get("_args") or [""])[0])
                if isinstance(target, str):
                    target = target.strip()
                full = self.registry.load_full(target)
                if not full:
                    obs_body = f"skill '{target}' 不存在。可用：{metas}"
                else:
                    self._read.add(target)
                    obs_body = f"已读取 [{target}] 的完整 SKILL.md（{len(full)} 字符）：\n\n{full}"
                _print_box(f"Phase 2 · read_skill · {target}（LLM 主动取回完整说明）",
                           full or obs_body)
                messages.append({"role": "user", "content": f"Observation: {obs_body}"})
                continue

            # ── 业务 skill：必须先 read_skill 过，否则拦截 ──────────────────
            if sname and sname not in self._read and self.registry.exists(sname):
                _print_box("拦截 · 未读先调", f"{sname} 尚未 read_skill，退回让 LLM 先读")
                messages.append({"role": "user", "content":
                    f"Observation: 你尚未读取 '{sname}' 的完整说明，"
                    f"请先 Action: read_skill(name=\"{sname}\") 再调用它。"})
                continue

            # ── 执行业务 skill ─────────────────────────────────────────────
            result = self.executor.execute(action_text)
            _print_box("Observation", json.dumps(result, ensure_ascii=False, indent=2))
            messages.append({"role": "user", "content":
                f"Observation: {json.dumps(result, ensure_ascii=False)}"})

        return "（达到最大步数仍未给出 Final Answer）"

    @staticmethod
    def _maybe_open_html(answer: str):
        # Windows 下用默认浏览器打开生成的闪卡，对应 SKILL.md 第 4 步「打开预览」
        m = re.search(r"outputs[/\\][\w-]+\.html", answer.replace("\\", "/"))
        if m:
            p = Path(m.group(0))
            if p.exists():
                try:
                    os.startfile(str(p))  # type: ignore[attr-defined]
                except Exception:
                    pass


if __name__ == "__main__":
    agent = SkillsReActAgent()
    q = sys.argv[1] if len(sys.argv) > 1 else "给我做一张 crazy 词的闪卡"
    agent.run(q)
