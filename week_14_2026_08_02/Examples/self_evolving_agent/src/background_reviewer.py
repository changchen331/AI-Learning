"""
后台回顾 Agent：分析近期对话，决定创建或修改哪些 Skill。

教学重点：
  1. 后台回顾 Agent 是独立的 LLM 调用，与主 Agent 解耦（Hermes 是异步 spawn）
  2. 回顾 Agent 拥有 policies.md（完整政策），主 Agent 没有——体现权限分层
  3. 回顾 Agent 输出结构化 JSON，由 SkillManager 执行 create/patch

使用方式：
  from background_reviewer import BackgroundReviewer
  reviewer = BackgroundReviewer("data/policies.md", sm)
  actions = reviewer.review(conversation_history)
  # actions 是 [{action, skill_name, ...}] 列表
"""

import json
import os
import re
from pathlib import Path

from openai import OpenAI

from skill_manager import SkillManager

REVIEWER_SYSTEM = """你是云购商城客服系统的"技能优化专家"。

以下给你的全部都是 Agent 最近一轮中**答错或推脱**的样本。你要做的是用最小改动
修复它们，让下次遇到同类问题能答对。

## 核心原则（严格遵守）

1. **仅修复观察到的失败**：只针对输入样本里出现的问题类型做改动，不要扩展到
   "政策里还有但样本里没出现"的场景
2. **最小改动优先**：
   - 能在已有 Skill 里追加或改一条分支解决的，不要新建 Skill
   - patch 的 old_text 只包含要改的那几行，不要把整段抄下来重写
3. **聚焦核心**：如果失败涉及多种类型，按失败条数从高到低，**只修复 1~2 类**
   —— 留出进化梯度，不要一次改完所有问题

你拥有完整政策文档，仅用于**核对 Agent 答错的具体数字/规则是否与政策一致**，
它是判定标准，不是 Agent 的知识补全大纲。

## 完整政策文档（判定标准）
{policies}

## 当前已有 Skill
{current_skills_summary}

## 输出格式
{{
  "analysis": "本轮失败 N 条，主要失败类型是 XXX",
  "actions": [
    {{"action": "create", "skill_name": "...", "reason": "修复哪条失败",
      "content": "完整SKILL.md（含frontmatter）"}},
    {{"action": "patch",  "skill_name": "...", "reason": "修复哪条失败",
      "old_text": "精确的原始文本", "new_text": "替换文本"}}
  ]
}}

只输出 JSON，不要有其他文字。如果发现失败数其实很少、没有清晰模式，可以只返回
1 条 action 甚至 0 条。"""

REVIEWER_USER = """## 本轮失败样本（共 {n} 条，都是 Agent 答错或推脱的）

{history_text}

## 当前 Skill 完整内容
{current_skills_full}

按核心原则给出最小必要的修复方案。"""


class BackgroundReviewer:
    def __init__(self, policies_path: str, skill_manager: SkillManager, model: str = "deepseek-chat"):
        self.policies = Path(policies_path).read_text(encoding="utf-8")
        self.skill_manager = skill_manager
        self.model = model
        self.last_analysis = ""  # UI 可读取最近一次回顾的分析文本
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )

    def review(self, failed_turns: list[dict]) -> list[dict]:
        """
        分析失败样本列表，返回最小必要的 Skill 操作。
        调用方应**仅传入本轮失败的条目**（每条形如 {question, answer, fail_reason}）；
        空列表直接返回 []，不做 LLM 调用。
        """
        if not failed_turns:
            return []

        current_skills = self.skill_manager.load_all()
        skills_summary = "\n".join(
            f"- {name}: {self._extract_description(content)}"
            for name, content in sorted(current_skills.items())
        ) or "（暂无已有Skill）"
        skills_full = "\n\n---\n\n".join(
            f"### {name}\n{content}" for name, content in sorted(current_skills.items())
        ) or "（暂无已有Skill）"

        system_msg = REVIEWER_SYSTEM.format(
            policies=self.policies,
            current_skills_summary=skills_summary,
        )
        user_msg = REVIEWER_USER.format(
            n=len(failed_turns),
            history_text=self._format_history(failed_turns),
            current_skills_full=skills_full,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=3000,
        )
        return self._parse_actions(response.choices[0].message.content.strip())

    def _format_history(self, turns: list[dict]) -> str:
        lines = []
        for i, t in enumerate(turns, 1):
            lines.append(f"[{i}] 用户: {t['question']}")
            lines.append(f"    客服: {t['answer'][:200]}{'...' if len(t['answer']) > 200 else ''}")
            # 评估结果（如果调用方附带了失败原因）
            if t.get("fail_reason"):
                lines.append(f"    ✗ 判定：{t['fail_reason']}")
        return "\n".join(lines)

    def _extract_description(self, content: str) -> str:
        m = re.search(r"description:\s*(.+)", content)
        return m.group(1).strip() if m else "(无描述)"

    def _parse_actions(self, raw: str) -> list[dict]:
        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                print(f"  [Reviewer] 无法提取 JSON，原始输出：{raw[:200]}")
                self.last_analysis = ""
                return []
            data = json.loads(json_match.group())
            self.last_analysis = data.get("analysis", "")
            print(f"  [Reviewer] 分析：{self.last_analysis[:100]}")
            return data.get("actions", [])
        except json.JSONDecodeError as e:
            print(f"  [Reviewer] JSON 解析失败: {e}\n原始: {raw[:300]}")
            self.last_analysis = ""
            return []
