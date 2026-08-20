"""
统一 LLM / Embedding 客户端

教学重点：
  1. OpenAI 兼容接口：同一份代码结构，换 base_url + 模型名即可切提供商
  2. 提供商分工：LLM 用 DeepSeek（deepseek-chat，即 deepseek-v4-flash），
     Embedding 用 DashScope（text-embedding-v3）—— DeepSeek 不提供 Embedding
  3. DashScope embedding 单批上限 10 条（实测，非官方宣传的 25），批处理必须分块

使用方式：
  from llm_client import llm_chat, embed_texts, EMBED_DIM
  answer = llm_chat(system, user)
  vecs   = embed_texts(["文本1", "文本2", ...])   # 自动按 10 条分批

依赖：
  pip install openai numpy
  export DEEPSEEK_API_KEY="sk-xxx"
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import logging
import os
import time
from typing import Optional

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

# ── 提供商配置（环境变量切换，代码结构不变）────────────────────────────────────
# DeepSeek：LLM 对话。deepseek-chat 即官网当前的 deepseek-v4-flash，同一模型
DEEPSEEK_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# DashScope：Embedding。DeepSeek 不提供 Embedding API，固定走阿里云
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_MODEL = "text-embedding-v3"
EMBED_DIM = 1024
EMBED_BATCH = 10  # ⚠️ DashScope text-embedding-v3 单批上限 10 条（实测踩坑）

# ── 客户端单例（懒加载，避免 import 时就要求环境变量）──────────────────────────
_llm_client: Optional[OpenAI] = None
_embed_client: Optional[OpenAI] = None


def get_llm_client() -> OpenAI:
    """DeepSeek LLM 客户端。需环境变量 DEEPSEEK_API_KEY。"""
    global _llm_client
    if _llm_client is None:
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise EnvironmentError("请设置环境变量 DEEPSEEK_API_KEY")
        _llm_client = OpenAI(api_key=key, base_url=DEEPSEEK_URL)
    return _llm_client


def get_embed_client() -> OpenAI:
    """DashScope Embedding 客户端。需环境变量 DASHSCOPE_API_KEY。"""
    global _embed_client
    if _embed_client is None:
        key = os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY")
        _embed_client = OpenAI(api_key=key, base_url=DASHSCOPE_URL)
    return _embed_client


# ── LLM 对话 ──────────────────────────────────────────────────────────────────

def llm_chat(
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format_json: bool = False,
        retries: int = 3,
) -> str:
    """
    单轮 LLM 对话，返回文本。失败自动重试（指数退避）。

    参数：
      response_format_json: True 时要求模型返回 JSON（DeepSeek 支持 json_object）
        —— 抽三元组时用，让模型输出结构化结果，解析更稳
    """
    client = get_llm_client()
    kwargs = dict(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if response_format_json:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning(f"LLM 调用失败({type(e).__name__})，{wait}s 后重试: {str(e)[:80]}")
            time.sleep(wait)


# ── Embedding 批处理 ──────────────────────────────────────────────────────────

def embed_texts(texts: list[str], batch_log_every: int = 100) -> np.ndarray:
    """
    批量文本向量化，自动按 EMBED_BATCH=10 分批，返回 [N, EMBED_DIM] float32（已归一化）。

    DashScope text-embedding-v3 实测单批上限 10 条，超过报 batch size is invalid。
    每 batch_log_every 批打印一次进度，避免长时间无输出（CLAUDE.md 4.3）。
    """
    client = get_embed_client()
    all_vecs = []
    n_batches = (len(texts) + EMBED_BATCH - 1) // EMBED_BATCH
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        batch_idx = i // EMBED_BATCH + 1
        if batch_idx % batch_log_every == 0:
            logger.info(f"embedding 进度: {batch_idx}/{n_batches} 批")
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch, dimensions=EMBED_DIM)
        for d in resp.data:
            all_vecs.append(d.embedding)
    vecs = np.asarray(all_vecs, dtype="float32")
    # L2 归一化，便于后续直接用内积近似余弦相似度
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
    return vecs


def embed_query(text: str) -> np.ndarray:
    """单条查询向量化，返回 [1, EMBED_DIM]。"""
    return embed_texts([text])


if __name__ == "__main__":
    # 自测：调一次 LLM + 一次 embedding
    logging.basicConfig(level=logging.INFO)
    print("LLM 测试:", llm_chat("你是测试助手", "只回复 OK 两个字"))
    v = embed_texts(["贵州茅台2023年营业收入", "宁德时代研发投入"])
    print(f"Embedding 测试: shape={v.shape}, dim={v.shape[1]}, 已归一化")
