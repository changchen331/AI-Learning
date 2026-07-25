"""
CLI 版 Agent — 四层记忆联动演示

教学重点：
  1. 每次对话前，打印四层记忆的加载明细（哪层加了多少内容）
  2. 语义检索结果在回答前展示（学生看到 Layer 4 被调用）
  3. /flush 命令触发完整 Memory Flush 流程，逐步打印进度
  4. 新会话开始时，记忆已从上次 Flush 中恢复

使用方式：
  python src/agent.py

命令：
  /flush    手动触发 Memory Flush
  /memory   查看当前 MEMORY.md 和 USER.md
  /layers   重新打印四层记忆加载情况
  /new      开始新会话（不触发 flush）
  /exit     退出（自动触发 flush）

依赖：
  pip install openai faiss-cpu
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import logging
import os
import sys
from pathlib import Path

# Windows OpenMP 冲突修复
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 让 src/ 内的模块可以相互 import
sys.path.insert(0, str(Path(__file__).parent.parent))

from week_13_2026_07_26.Examples.agent_memory_system.src.session_db import SessionDB
from week_13_2026_07_26.Examples.agent_memory_system.src.memory_loader import MemoryLoader
from week_13_2026_07_26.Examples.agent_memory_system.src.vector_store import VectorStore
from week_13_2026_07_26.Examples.agent_memory_system.src.fts_store import FTSStore
from week_13_2026_07_26.Examples.agent_memory_system.src.retrieval import HybridRetriever
from week_13_2026_07_26.Examples.agent_memory_system.src.memory_flush import MemoryFlusher
from week_13_2026_07_26.Examples.agent_memory_system.src.llm_config import get_chat_client, current_model_info

logging.basicConfig(level=logging.WARNING)

AUTO_FLUSH_THRESHOLD = 20  # 消息数超过此值自动触发 flush

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
DIM = "\033[2m"


def print_layer_info(layers, semantic_results=None):
    print(f"\n{CYAN}{'─' * 60}{RESET}")
    print(f"{CYAN}  四层记忆加载情况{RESET}")
    print(f"{CYAN}{'─' * 60}{RESET}")
    layer_icons = {"soul": "🧠", "daily_log": "🫧", "user_profile": "👤", "agents_manual": "📋", "long_term_memory": "💾"}
    layer_names = {
        "soul": "Layer 3a  SOUL.md（人格定义）",
        "daily_log": "Layer 2   每日日志（今天 + 昨天）",
        "user_profile": "Layer 3b  USER.md（用户画像）",
        "agents_manual": "Layer 3c  AGENTS.md（操作规范）",
        "long_term_memory": "Layer 3d  MEMORY.md（长期记忆）",
    }
    for layer in layers:
        name = layer_names.get(layer.name, layer.name)
        chars = layer.char_count
        print(f"  {layer_icons.get(layer.name, '·')} {name}  {DIM}[{chars} 字符]{RESET}")

    if semantic_results:
        print(f"  🔍 Layer 4   混合检索（向量 0.7 + BM25 0.3）  {DIM}[{len(semantic_results)} 条命中]{RESET}")
        for r in semantic_results:
            score_pct = int(r["score"] * 100)
            cat = r.get("category", "?")
            title = r.get("title", r.get("content", "")[:30])
            src = r.get("source", "?")
            print(f"      {DIM}[{cat}] {title}  相似度 {score_pct}%  来源:{src}{RESET}")
    else:
        print(f"  🔍 Layer 4   混合检索（向量 0.7 + BM25 0.3）  {DIM}[暂无命中]{RESET}")
    print(f"{CYAN}{'─' * 60}{RESET}\n")


def do_flush(flusher: MemoryFlusher, db: SessionDB, session_id: int):
    messages = db.get_session_messages(session_id)
    user_messages = [m for m in messages if m["role"] in ("user", "assistant")]
    if not user_messages:
        print(f"{YELLOW}会话为空，跳过 Flush。{RESET}")
        return

    print(f"\n{MAGENTA}{'═' * 60}{RESET}")
    print(f"{MAGENTA}  Memory Flush 开始...{RESET}")
    print(f"{MAGENTA}{'═' * 60}{RESET}")
    print(f"  分析 {len(user_messages)} 条消息...")

    result = flusher.flush(user_messages, session_id)

    if result.error:
        print(f"{YELLOW}  [错误] {result.error}{RESET}")
        return

    print(f"\n  {GREEN}Pass 1 — 用户信息更新 ({len(result.user_updates)} 项){RESET}")
    for u in result.user_updates:
        print(f"    ✓ {u}")
    if not result.user_updates:
        print(f"    {DIM}（无新信息）{RESET}")

    print(f"\n  {GREEN}Pass 2 — 新增长期记忆 ({len(result.new_memory_entries)} 条){RESET}")
    for e in result.new_memory_entries:
        cat = e.get("category", "?")
        title = e.get("title", "")
        print(f"    [{cat}] {title}")
    if not result.new_memory_entries:
        print(f"    {DIM}（无新记忆）{RESET}")

    print(f"\n  {GREEN}Pass 3 — 向量化写入 FAISS：{result.vectorized_count} 条{RESET}")

    if result.compacted:
        print(f"\n  {YELLOW}Compaction：{result.compaction_before} → {result.compaction_after} 条{RESET}")

    db.mark_flushed(session_id)
    print(f"\n{MAGENTA}{'═' * 60}{RESET}")
    print(f"{MAGENTA}  Flush 完成！长期记忆已更新。{RESET}")
    print(f"{MAGENTA}{'═' * 60}{RESET}\n")


def show_memory(loader: MemoryLoader):
    user_md = loader.get_user_md_path().read_text(encoding="utf-8")
    memory_md = loader.get_memory_md_path().read_text(encoding="utf-8")
    entry_count = loader.get_memory_entry_count()
    print(f"\n{CYAN}=== USER.md ==={RESET}")
    print(user_md[:1500])
    print(f"\n{CYAN}=== MEMORY.md ({entry_count} 条记忆条目) ==={RESET}")
    print(memory_md[:2000])
    print()


def main():
    model_info = current_model_info()
    print(f"\n{BOLD}Agent 记忆系统 — CLI 演示{RESET}")
    print(f"当前模型：{CYAN}{model_info['display']}{RESET}  "
          f"{DIM}（切换：LLM_PROVIDER=deepseek 或 qwen）{RESET}")
    print("输入 /flush, /memory, /layers, /new, /exit 查看各功能\n")

    try:
        get_chat_client()  # 提前检查 API Key 是否设置
    except EnvironmentError as e:
        print(f"{YELLOW}{e}{RESET}")
        sys.exit(1)

    db = SessionDB()
    loader = MemoryLoader()
    vs = VectorStore()
    fts = FTSStore()
    retriever = HybridRetriever(vs, fts)
    flusher = MemoryFlusher()

    session_id = db.new_session()

    # 初始加载四层记忆
    prompt_result = loader.build_system_prompt(recent_memory_limit=10)
    print_layer_info(prompt_result.layers)

    messages: list[dict] = []

    while True:
        try:
            user_input = input(f"{BOLD}你：{RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            user_input = "/exit"

        if not user_input:
            continue

        # ── 命令处理 ──────────────────────────────────────────────────
        if user_input == "/exit":
            do_flush(flusher, db, session_id)
            db.close_session(session_id, title=messages[0]["content"][:30] if messages else "空会话")
            print("再见！")
            break

        if user_input == "/flush":
            do_flush(flusher, db, session_id)
            # Flush 后重新加载记忆
            prompt_result = loader.build_system_prompt(recent_memory_limit=10)
            continue

        if user_input == "/memory":
            show_memory(loader)
            continue

        if user_input == "/layers":
            # 用当前消息做一次语义搜索演示
            query = messages[-1]["content"] if messages else ""
            semantic = retriever.search(query, top_k=3) if query else []
            prompt_result = loader.build_system_prompt(recent_memory_limit=10)
            print_layer_info(prompt_result.layers, semantic)
            continue

        if user_input == "/new":
            db.close_session(session_id, title=messages[0]["content"][:30] if messages else "空会话")
            session_id = db.new_session()
            messages = []
            prompt_result = loader.build_system_prompt(recent_memory_limit=10)
            print(f"{GREEN}新会话已开始，记忆已重新加载。{RESET}")
            print_layer_info(prompt_result.layers)
            continue

        # ── Layer 4：混合检索（向量 0.7 + BM25 0.3）────────────────────
        semantic_results = retriever.search(user_input, top_k=3)
        if semantic_results:
            print(f"  {DIM}[混合检索] 找到 {len(semantic_results)} 条相关记忆{RESET}")

        # ── 组装 Context Window ────────────────────────────────────────
        semantic_context = ""
        if semantic_results:
            snippets = [f"- [{r['category']}] {r['content'][:100]}" for r in semantic_results]
            semantic_context = "相关历史记忆：\n" + "\n".join(snippets)

        system_prompt = prompt_result.system_prompt
        if semantic_context:
            system_prompt += f"\n\n## 语义检索到的相关记忆\n{semantic_context}"

        # Layer 2：当日会话历史（当前会话消息直接复用 messages 列表）
        api_messages = [{"role": "system", "content": system_prompt}] + messages
        api_messages.append({"role": "user", "content": user_input})

        # ── LLM 调用（流式输出）───────────────────────────────────────
        print(f"{GREEN}Muse：{RESET}", end="", flush=True)
        client, model = get_chat_client()
        stream = client.chat.completions.create(
            model=model, messages=api_messages, temperature=0.7, stream=True
        )
        response_text = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            response_text += delta
        print()

        # 记录到数据库 + 本地列表
        db.add_message(session_id, "user", user_input)
        db.add_message(session_id, "assistant", response_text)
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": response_text})

        # 超过阈值自动 flush
        if db.get_message_count(session_id) >= AUTO_FLUSH_THRESHOLD:
            print(f"\n{YELLOW}[自动触发 Flush：消息数达到 {AUTO_FLUSH_THRESHOLD}]{RESET}")
            do_flush(flusher, db, session_id)
            prompt_result = loader.build_system_prompt(recent_memory_limit=10)


if __name__ == "__main__":
    main()
