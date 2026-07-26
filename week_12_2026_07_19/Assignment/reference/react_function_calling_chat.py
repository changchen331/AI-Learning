"""
Function Calling 版 ReAct Agent —— 多轮对话扩展

教学重点（在 react_function_calling.py 基础上的最小化改造）：
  1. 对话记忆：把 messages 列表从 run() 内部的局部变量，提升为
     ChatAgent 的实例属性，跨用户轮次持久化。这样模型能记住上一轮
     问过什么、查到过什么代码、算过什么数。
  2. 单轮 ReAct 循环不变：每一轮用户输入仍然走「调用工具 → 观察 →
     再调用」直到 finish_reason == "stop" 给出最终答案。
  3. 终止后把最终 assistant 回答写回 messages，作为下一轮的上下文。
  4. 命令行 REPL：while True 读输入，输入 exit/quit 退出。

使用方式：
  python react_function_calling_chat.py
  python react_function_calling_chat.py --max_steps 8

依赖：
  pip install openai faiss-cpu sentence-transformers akshare
  export DEEPSEEK_API_KEY="sk-xxx"   （或改回 DashScope，见下方注释）
"""

import argparse
import json
import logging
import os
import time
from typing import Generator, List, Dict

from openai import OpenAI

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# client = OpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
# MODEL = os.getenv("AGENT_MODEL", "qwen-max")
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")

FC_SYSTEM_PROMPT = """你是一个专业的A股金融分析助手，正在与用户进行多轮对话。
规则：
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- Final Answer 必须引用具体数据来源
- 如果没有合适工具能回答，直接说明原因
- 用户可能在后续轮次里基于上一轮的结论追问，请结合对话历史作答
"""


class ChatAgent:
    """
    支持多轮对话的 ReAct Agent。

    与 react_function_calling.run() 的唯一本质区别：
      messages 不再每次重建，而是作为实例属性持久保留，
      使每一轮的工具调用结果与最终回答都能成为下一轮的上下文。
    """

    def __init__(self, max_steps: int = 10, system_prompt: str = FC_SYSTEM_PROMPT):
        self.max_steps = max_steps
        # 对话记忆：system + 历次 user/assistant/tool 消息持久保留
        self.messages: List[Dict] = [
            {"role": "system", "content": system_prompt},
        ]

    def chat(self, user_input: str) -> Generator[dict, None, None]:
        """
        处理一轮用户输入，yield 每一步结构化结果。

        结束后，本轮的最终回答已写入 self.messages，下一轮直接追加即可。
        """
        from week_12_2026_07_19.Examples.react_financial_agent.src.tools import TOOLS_MAP, TOOLS_SCHEMA

        # 记录本轮用户输入到对话历史
        self.messages.append({"role": "user", "content": user_input})

        for step in range(1, self.max_steps + 1):
            response = client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0,
            )
            msg = response.choices[0].message
            reason = response.choices[0].finish_reason

            # 模型决定直接回答（无工具调用）—— 本轮结束
            if reason == "stop" or not msg.tool_calls:
                answer = msg.content or "（模型返回空内容）"
                # 把最终回答写回历史，作为下一轮上下文
                self.messages.append({"role": "assistant", "content": answer})
                yield {
                    "step": step,
                    "type": "final",
                    "thought": "",
                    "answer": answer,
                }
                return

            # 模型请求调用工具：先把 assistant 的 tool_calls 消息入历史
            self.messages.append(msg)

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                tool_fn = TOOLS_MAP.get(tool_name)
                if tool_fn is None:
                    observation = f"未知工具 '{tool_name}'"
                else:
                    try:
                        observation = tool_fn(**tool_args)
                    except TypeError as e:
                        observation = f"工具参数错误: {e}"

                yield {
                    "step": step,
                    "type": "action",
                    "thought": "",  # Function Calling 版 Thought 在模型内部，不可见
                    "action": tool_name,
                    "action_input": tool_args,
                    "observation": str(observation),
                }

                # 工具结果入历史，供模型下一跳消费
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(observation),
                })

        yield {
            "step": self.max_steps + 1,
            "type": "max_steps",
            "answer": f"已达最大步数 {self.max_steps}，未能得出最终答案",
        }

    def reset(self):
        """清空对话历史，保留 system prompt（可按需调用）"""
        self.messages = [{"role": "system", "content": self.messages[0]["content"]}]


# ── CLI 打印（复用 react_function_calling 的彩色输出） ────────────────────────

COLORS = {
    "thought": "\033[36m",
    "action": "\033[33m",
    "obs": "\033[32m",
    "final": "\033[1;93m",  # 加粗亮黄，黑色背景下清晰可读
    "error": "\033[31m",
    "reset": "\033[0m",
}


def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def chat_and_print(agent: ChatAgent, user_input: str):
    """跑一轮并把过程打印出来，返回耗时"""
    start = time.time()
    for step_data in agent.chat(user_input):
        stype = step_data["type"]

        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            print(_c("thought", "🧠 Thought: （模型内部推理，Function Calling 版不可见）"))
            print(_c("action", f"🔧 Action:  {step_data['action']}"))
            print(_c("action", f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
            print(_c("obs", f"👁  Obs:     {step_data['observation'][:300]}"))

        elif stype == "final":
            elapsed = time.time() - start
            print(f"\n{'─' * 60}")
            print(_c("final", f"\n✅ Final Answer:\n{step_data['answer']}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")

        elif stype in ("error", "max_steps"):
            print(_c("error", f"\n⚠️  {step_data.get('answer', '')}"))


def main():
    parser = argparse.ArgumentParser(description="多轮对话版 Function Calling ReAct Agent")
    parser.add_argument("--max_steps", type=int, default=10, help="每轮最大工具调用步数")
    args = parser.parse_args()

    agent = ChatAgent(max_steps=args.max_steps)

    print("=" * 60)
    print("A股金融分析助手 · 多轮对话模式")
    print(f"模型: {MODEL}  实现: Function Calling (Multi-turn)")
    print("输入 exit 或 quit 退出；输入 reset 清空对话历史")
    print("=" * 60)

    while True:
        try:
            question = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "退出"):
            print("再见！")
            break
        if question.lower() == "reset":
            agent.reset()
            print(_c("final", "🔁 对话历史已清空"))
            continue

        chat_and_print(agent, question)


if __name__ == "__main__":
    main()
