"""
向量索引构建脚本（原生实现）

Embedding 方案：阿里云 DashScope text-embedding-v3
  - 无需下载本地模型，直接 API 调用
  - 维度：1024（可设为 768 / 512 节省存储）
  - 每批最多 25 条（DashScope 限制）
  - 费用极低：约 0.0007 元 / 千 token

向量库：FAISS（IndexFlatIP，内积 = 归一化后的余弦相似度）

依赖：
  pip install faiss-cpu openai numpy
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
CHUNKS_DIR      = BASE_DIR / "data" / "chunks"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY        = "semantic"          # 与 chunk_documents.py 保持一致
CHUNKS_FILE     = CHUNKS_DIR / f"all_{STRATEGY}.json"

EMBED_MODEL     = "text-embedding-v3"
EMBED_DIM       = 1024                # 可选 768 / 512 节省存储
BATCH_SIZE      = 10                  # DashScope text-embedding-v3 单次最多 10 条
DASHSCOPE_URL   = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ── DashScope 客户端 ──────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "请设置环境变量 DASHSCOPE_API_KEY\n"
            "  Windows: set DASHSCOPE_API_KEY=sk-xxx\n"
            "  Linux/Mac: export DASHSCOPE_API_KEY=sk-xxx"
        )
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_texts(client: OpenAI, texts: list[str], show_progress: bool = True) -> np.ndarray:
    """
    批量计算 embedding，每批最多 25 条。
    返回 shape=(N, EMBED_DIM) 的 float32 数组，已 L2 归一化。
    """
    all_embeddings = []
    total_batches  = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch     = texts[i : i + BATCH_SIZE]
        batch_idx = i // BATCH_SIZE + 1

        if show_progress and batch_idx % 100 == 0:
            logger.info(f"  Embedding 进度: {batch_idx}/{total_batches} 批")

        # 带重试
        for attempt in range(3):
            try:
                resp = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch,
                    dimensions=EMBED_DIM,
                )
                vecs = [e.embedding for e in resp.data]
                all_embeddings.extend(vecs)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning(f"  第{attempt+1}次失败，重试: {e}")
                time.sleep(2 ** attempt)

    embeddings = np.array(all_embeddings, dtype="float32")

    # L2 归一化（使内积等于余弦相似度）
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)     # 防止除零
    embeddings = embeddings / norms

    return embeddings


# ── FAISS 索引构建 ─────────────────────────────────────────────────────────────

def build_faiss_index(chunks: list[dict], client: OpenAI):
    """
    构建 FAISS 向量索引。

    FAISS 说明：
      IndexFlatIP = 暴力内积检索，精确但不近似。
      数据量 < 10 万时速度完全够用，是教学的首选。
      数据量更大时可换 IndexIVFFlat（需要 train）或 IndexHNSW。
    """
    import faiss

    logger.info(f"开始计算 {len(chunks)} 条 chunk 的 embedding...")
    texts      = [c["content"] for c in chunks]
    embeddings = embed_texts(client, texts)

    logger.info(f"构建 FAISS 索引，维度={EMBED_DIM}...")
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    logger.info(f"索引构建完成，共 {index.ntotal} 条向量")

    # 持久化：索引文件 + 元数据（分开存，避免把大向量序列化进 JSON）
    index_path = VECTORSTORE_DIR / "faiss_index.bin"
    meta_path  = VECTORSTORE_DIR / "faiss_meta.json"

    faiss.write_index(index, str(index_path))
    logger.info(f"FAISS 索引已保存 → {index_path}  ({index_path.stat().st_size//1024} KB)")

    meta_list = [
        {
            "chunk_id":   c["chunk_id"],
            "content":    c["content"],
            "stock_code": c["metadata"].get("stock_code", ""),
            "year":       c["metadata"].get("year", ""),
            "page_num":   c["metadata"].get("page_num", -1),
            "section":    c["metadata"].get("section", ""),
            "block_types":c["metadata"].get("block_types", []),
            "is_ocr":     c["metadata"].get("is_ocr", False),
            "strategy":   c["metadata"].get("strategy", ""),
            "source_file":c["metadata"].get("source_file", ""),
            # 层级分块时保留父块内容供 LLM 读取
            "parent_content": c["metadata"].get("parent_content", ""),
            "parent_id":      c["metadata"].get("parent_id", ""),
        }
        for c in chunks
    ]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)
    logger.info(f"元数据已保存 → {meta_path}")

    return index, meta_list


# ── ChromaDB 索引构建（可选对比） ──────────────────────────────────────────────

def build_chroma_index(chunks: list[dict], client: OpenAI):
    """
    ChromaDB 版本（可选）。
    优势：内置元数据过滤，可直接 where={"stock_code": "600519"} 过滤。
    劣势：写入较慢，不适合大批量。
    """
    try:
        import chromadb
    except ImportError:
        logger.error("请先安装 chromadb: pip install chromadb")
        return

    chroma_dir = VECTORSTORE_DIR / "chroma"
    client_db  = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client_db.get_or_create_collection(
        name="annual_reports",
        metadata={"hnsw:space": "cosine"},
    )

    logger.info(f"向 ChromaDB 写入 {len(chunks)} 条 chunk...")
    texts      = [c["content"] for c in chunks]
    embeddings = embed_texts(client, texts)

    for i in range(0, len(chunks), 100):
        batch   = chunks[i:i+100]
        ids     = [c["chunk_id"] for c in batch]
        docs    = [c["content"] for c in batch]
        embs    = embeddings[i:i+100].tolist()
        metas   = []
        for c in batch:
            m = dict(c["metadata"])
            # ChromaDB 只支持 str/int/float/bool 类型的 metadata value
            m["block_types"] = ",".join(m.get("block_types") or [])
            m.pop("parent_content", None)   # 太长
            metas.append(m)
        collection.add(documents=docs, embeddings=embs, ids=ids, metadatas=metas)
        logger.info(f"  已写入 {min(i+100, len(chunks))}/{len(chunks)}")

    logger.info(f"ChromaDB 写入完成，共 {collection.count()} 条")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    if not CHUNKS_FILE.exists():
        logger.error(f"找不到 {CHUNKS_FILE}，请先运行 chunk_documents.py")
        return

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"加载 {len(chunks)} 个 chunks（策略={STRATEGY}）")

    client = get_client()

    # 默认用 FAISS
    build_faiss_index(chunks, client)

    # 若要对比 ChromaDB，取消下行注释：
    # build_chroma_index(chunks, client)

    logger.info("\n索引构建完成！")
    logger.info(f"  FAISS 索引: {VECTORSTORE_DIR / 'faiss_index.bin'}")
    logger.info(f"  元数据:     {VECTORSTORE_DIR / 'faiss_meta.json'}")


if __name__ == "__main__":
    main()
