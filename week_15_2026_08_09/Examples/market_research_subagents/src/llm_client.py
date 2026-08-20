"""极简 LLM 客户端（市场调研 subagent 项目用）
DeepSeek deepseek-chat（即 deepseek-v4-flash），OpenAI 兼容接口。
依赖：pip install openai"""
import logging
import os
import time

from openai import OpenAI

logger = logging.getLogger(__name__)
DEEPSEEK_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
_client = None


def get_client():
    global _client
    if _client is None:
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key: raise EnvironmentError("请设置 DEEPSEEK_API_KEY")
        _client = OpenAI(api_key=key, base_url=DEEPSEEK_URL)
    return _client


def llm_chat(system, user, *, temperature=0.0, max_tokens=1024, stop=None, retries=3):
    """单轮 LLM 对话。stop 用于 ReAct 在 Observation 前截断。"""
    for attempt in range(retries):
        try:
            resp = get_client().chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature, max_tokens=max_tokens, stop=stop)
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == retries - 1: raise
            time.sleep(2 ** attempt);
            logger.warning(f"LLM 重试({attempt + 1}): {str(e)[:80]}")
