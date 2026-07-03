"""
向量索引构建脚本（LangChain 版）

与原生版（src/build_index.py）的对比：
  原生版：自己实现 embedding 批处理、FAISS 操作、元数据管理
  本版本：LangChain 接管 loader / splitter / embedding / vectorstore 全链路
         代码量约为原生版的 1/3，但透明度更低

Embedding：本地 BAAI/bge-small-zh-v1.5
  需要先运行: python src_langchain/download_model.py

向量库：LangChain FAISS 封装
  保存路径：vectorstore/faiss_lc/

依赖：
  pip install langchain langchain-community langchain-huggingface faiss-cpu pymupdf sentence-transformers
"""

import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
RAW_DIR         = BASE_DIR / "data" / "raw_pdf"
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_lc"
MODELS_DIR      = BASE_DIR / "models"
BGE_MODEL_PATH  = MODELS_DIR / "bge-small-zh-v1.5"


# ── 1. 加载 PDF（LangChain Document Loader）───────────────────────────────────

def load_documents():
    """
    LangChain 的 PyMuPDFLoader 每页输出一个 Document。
    Document 结构：
      page_content: str   页面文字
      metadata: dict      包含 source（文件路径）、page（页码）
    """
    from langchain_community.document_loaders import PyMuPDFLoader

    pdf_files = list(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"data/raw_pdf/ 目录下没有 PDF 文件，请先运行 src/download_reports.py")

    all_docs = []
    for pdf_path in sorted(pdf_files):
        logger.info(f"加载: {pdf_path.name}")
        loader = PyMuPDFLoader(str(pdf_path))
        docs   = loader.load()

        # 从文件名解析股票代码和年份，写入每个 Document 的 metadata
        parts = pdf_path.stem.split("_")   # "600519_2023_贵州茅台_..."
        for doc in docs:
            doc.metadata["stock_code"]   = parts[0] if len(parts) > 0 else ""
            doc.metadata["year"]         = parts[1] if len(parts) > 1 else ""
            doc.metadata["company_name"] = parts[2] if len(parts) > 2 else ""
            doc.metadata["filename"]     = pdf_path.name

        all_docs.extend(docs)
        logger.info(f"  → {len(docs)} 页")

    logger.info(f"共加载 {len(all_docs)} 页（来自 {len(pdf_files)} 个文件）")
    return all_docs


# ── 2. 文本分块（LangChain Text Splitter）────────────────────────────────────

def split_documents(docs):
    """
    RecursiveCharacterTextSplitter：
      按优先级尝试不同的分隔符，从句子边界切，保留语义完整性。
      中文文档优先按句号、换行切，退而求其次按空格/字符切。

    对比原生版 semantic splitter：
      LangChain 版不感知文档结构（标题/表格），只看字符。
      原生版能识别标题层级，表格单独成块。
      LangChain 版代码更简单，但分块质量略低于原生语义分块。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["。\n", "！\n", "？\n", "\n\n", "。", "！", "？", "；", "\n", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(docs)
    logger.info(f"分块完成：{len(docs)} 页 → {len(chunks)} 个 chunk")
    logger.info(f"平均 chunk 长度：{sum(len(c.page_content) for c in chunks)//len(chunks)} 字符")
    return chunks


# ── 3. Embedding 模型（本地 BGE）──────────────────────────────────────────────

def get_embeddings():
    """
    HuggingFaceEmbeddings 封装 sentence-transformers，
    指定 cache_folder 将模型限定在项目目录内。

    BGE 模型说明：
      encode_kwargs={"normalize_embeddings": True}  保证向量 L2 归一化
      model_kwargs={"device": "cpu"}  明确指定 CPU（有 CUDA 时改为 "cuda"）

    如果模型未下载，会自动从 HuggingFace 下载（可能很慢）。
    建议先运行: python src_langchain/download_model.py
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    model_path = str(BGE_MODEL_PATH) if BGE_MODEL_PATH.exists() else "BAAI/bge-small-zh-v1.5"
    if not BGE_MODEL_PATH.exists():
        logger.warning(
            f"本地模型不存在: {BGE_MODEL_PATH}\n"
            "  将从 HuggingFace 下载（建议先运行 python src_langchain/download_model.py）"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        cache_folder=str(MODELS_DIR),       # 备用缓存也在项目目录
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info(f"Embedding 模型加载完成: {model_path}")
    return embeddings


# ── 4. 构建并保存 FAISS 向量库 ────────────────────────────────────────────────

def build_vectorstore(chunks, embeddings):
    """
    LangChain 的 FAISS.from_documents() 一行完成：
      - 提取每个 chunk 的文本
      - 批量计算 embedding
      - 构建 FAISS 索引
      - 绑定 metadata

    对比原生版：原生版手动写了 ~50 行代码完成相同的事情。
    """
    from langchain_community.vectorstores import FAISS

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"构建向量库（{len(chunks)} 个 chunk）...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 保存到项目目录
    vectorstore.save_local(str(VECTORSTORE_DIR))
    logger.info(f"向量库已保存 → {VECTORSTORE_DIR}")
    logger.info(f"  index.faiss: {(VECTORSTORE_DIR / 'index.faiss').stat().st_size // 1024} KB")
    return vectorstore


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    # 步骤 1: 加载
    docs   = load_documents()

    # 步骤 2: 分块
    chunks = split_documents(docs)

    # 步骤 3: Embedding
    embeddings = get_embeddings()

    # 步骤 4: 建库
    build_vectorstore(chunks, embeddings)

    print(f"\nLangChain 向量库构建完成！")
    print(f"  路径: {VECTORSTORE_DIR}")
    print(f"  下一步: python src_langchain/rag_chain_lc.py")


if __name__ == "__main__":
    main()
