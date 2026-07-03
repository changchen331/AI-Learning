"""
RAG 问答链（LangChain LCEL 版）

核心对比点（与原生版 src/rag_pipeline.py）：
┌──────────────────┬──────────────────────┬──────────────────────┐
│ 环节             │ 原生版               │ LangChain 版         │
├──────────────────┼──────────────────────┼──────────────────────┤
│ 检索             │ FAISS + BM25 混合    │ FAISS 单路           │
│ 排序             │ RRF + CrossEncoder   │ 相似度得分直接排序   │
│ 链路组织         │ 手写流程控制         │ LCEL pipe (|) 操作符 │
│ 输入输出         │ dict                 │ Runnable 统一接口    │
│ 代码量           │ ~300 行              │ ~120 行              │
│ 可调试性         │ 高（每步都可打印）   │ 中（需用 callbacks） │
│ 扩展性           │ 需手动写             │ 框架提供 Agent/Memory│
└──────────────────┴──────────────────────┴──────────────────────┘

LCEL（LangChain Expression Language）核心概念：
  - 所有组件都是 Runnable（有 .invoke() / .stream() / .batch() 接口）
  - | 操作符串联 Runnable，上一个的输出是下一个的输入
  - RunnablePassthrough() 透传当前输入（用于多变量 context）
  - 整条链本身也是 Runnable，可以直接 chain.invoke(question)

使用方式：
  python rag_chain_lc.py                        # 交互式
  python rag_chain_lc.py --query "茅台2023营收"
  python rag_chain_lc.py --query "..." --with-sources  # 附带来源

依赖：
  pip install langchain langchain-openai langchain-community langchain-huggingface faiss-cpu
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import os
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_lc"
MODELS_DIR      = BASE_DIR / "models"
BGE_MODEL_PATH  = MODELS_DIR / "bge-small-zh-v1.5"

DASHSCOPE_URL   = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL       = "qwen-plus"

SYSTEM_PROMPT = """你是一个专业的财务分析助手，专门回答关于中国上市公司年度报告的问题。

回答规则：
1. 只根据【参考资料】中的内容回答，不得编造资料外的数据
2. 若参考资料不足以支撑回答，直接说"根据提供的资料无法回答此问题"
3. 引用具体数据时，在句末标注来源文件名，如：营业收入为1476亿元（来源：贵州茅台2023年报）
4. 数字要精确，回答简洁，重点突出"""


# ── 组件初始化 ────────────────────────────────────────────────────────────────

def get_llm():
    """
    LangChain ChatOpenAI 指向 DashScope。
    任何 OpenAI 接口兼容的服务只需改 base_url 和 model_name。
    """
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY")

    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=api_key,
        openai_api_base=DASHSCOPE_URL,
        temperature=0.1,
    )


def get_embeddings():
    """复用与 build_index_lc.py 相同的 embedding 模型（必须一致，否则向量空间不匹配）。"""
    from langchain_huggingface import HuggingFaceEmbeddings

    model_path = str(BGE_MODEL_PATH) if BGE_MODEL_PATH.exists() else "BAAI/bge-small-zh-v1.5"
    return HuggingFaceEmbeddings(
        model_name=model_path,
        cache_folder=str(MODELS_DIR),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(embeddings):
    """加载已构建的 FAISS 向量库。"""
    from langchain_community.vectorstores import FAISS

    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"向量库不存在: {VECTORSTORE_DIR}\n"
            "请先运行: python src_langchain/build_index_lc.py"
        )
    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,   # FAISS 本地加载需要此标志
    )


# ── LCEL 链构建 ───────────────────────────────────────────────────────────────

def build_chain(vectorstore):
    """
    构建标准 RAG 链。

    数据流图：
                    question
                      │
           ┌──────────┴──────────┐
           │                     │
      retriever              RunnablePassthrough
    (返回 docs)            (原样透传 question)
           │                     │
      format_docs            question
           │                     │
           └──────────┬──────────┘
                      │
              {"context": ..., "question": ...}
                      │
                   prompt
                      │
                     llm
                      │
                StrOutputParser
                      │
                  answer (str)
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    llm = get_llm()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    # 将 Document 列表格式化为带编号的上下文字符串
    def format_docs(docs) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            meta     = doc.metadata
            filename = meta.get("filename", "")
            page     = meta.get("page", "")
            label    = f"[{i}] {filename}"
            if page:
                label += f" 第{page+1}页"
            parts.append(f"{label}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "【参考资料】\n{context}\n\n【问题】\n{question}\n\n请根据参考资料回答，标注来源文件。"),
    ])

    # LCEL 核心：用 | 串联各组件
    chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def build_chain_with_sources(vectorstore):
    """
    返回答案 + 来源文档的版本。
    使用 RunnableParallel 同时执行检索和问题透传，然后分别处理。
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    from langchain_core.output_parsers import StrOutputParser

    llm       = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            meta  = doc.metadata
            label = f"[{i}] {meta.get('filename', '')} 第{meta.get('page', 0)+1}页"
            parts.append(f"{label}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "【参考资料】\n{context}\n\n【问题】\n{question}"),
    ])

    # 并行：一路检索原始 docs，另一路用格式化 docs 生成答案
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt | llm | StrOutputParser()
    )

    chain_with_sources = RunnableParallel(
        {
            "context":  retriever,
            "question": RunnablePassthrough(),
        }
    ).assign(answer=rag_chain_from_docs)

    return chain_with_sources


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="年报 RAG 问答（LangChain LCEL 版）")
    parser.add_argument("--query",        type=str, default=None)
    parser.add_argument("--with-sources", action="store_true", help="输出结果时附带来源文档片段")
    args = parser.parse_args()

    logger.info("加载 embedding 模型...")
    embeddings  = get_embeddings()
    logger.info("加载向量库...")
    vectorstore = get_vectorstore(embeddings)

    if args.with_sources:
        chain = build_chain_with_sources(vectorstore)
    else:
        chain, _ = build_chain(vectorstore)

    def run_query(question: str):
        print(f"\n{'='*60}")
        print(f"问题：{question}")
        print(f"{'='*60}")

        if args.with_sources:
            result  = chain.invoke(question)
            answer  = result["answer"]
            sources = result["context"]
            print(f"\n{answer}")
            print("\n── 来源文档片段 ──")
            for i, doc in enumerate(sources, 1):
                meta = doc.metadata
                print(f"[{i}] {meta.get('filename', '')} 第{meta.get('page', 0)+1}页")
                print(f"    {doc.page_content[:120]}...")
        else:
            answer = chain.invoke(question)
            print(f"\n{answer}")

    if args.query:
        run_query(args.query)
    else:
        print(f"年报 RAG 问答系统（LangChain LCEL 版）")
        print(f"模型：{LLM_MODEL}  |  向量库：{VECTORSTORE_DIR}")
        print("输入 'exit' 退出\n")
        while True:
            try:
                q = input("问题：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() == "exit":
                break
            run_query(q)


if __name__ == "__main__":
    main()
