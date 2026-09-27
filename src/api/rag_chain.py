# src/api/rag_chain.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()

# 1. 初始化 Qwen 客户端
client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 2. 加载 BGE 中文 Embedding 模型
print("正在加载 BGE 中文 Embedding 模型...")
embed_model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

# 3. 连接 ChromaDB（不要让它自己下模型）
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection(
    name="day4_chunking_eval",
    embedding_function=None
)


def rag_query(question: str) -> dict:
    """纯向量检索 top-3 → LLM 生成。"""
    question_embedding = embed_model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )

    retrieved_chunks = results['documents'][0]
    metadatas = results['metadatas'][0]

    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt = f"""你是一个严谨的助手。请仅根据以下参考资料回答问题。
如果参考资料中没有答案，请直接说“根据提供的资料无法回答”。

参考资料：
{context}

问题：{question}
答案："""

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": metadatas,
        "contexts": retrieved_chunks
    }


def rag_query_with_rerank(question: str, recall_k: int = 10, top_k: int = 3) -> dict:
    """
    向量召回 recall_k 条 → Rerank → 取 top_k → 生成答案。
    rerank 模型延迟导入，避免每次 import 本文件都去加载 reranker。
    """
    from src.retrieval.rerank import rerank

    question_embedding = embed_model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    # 先召回过量候选
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=recall_k
    )
    candidates = results['documents'][0]
    metadatas = results['metadatas'][0]

    # Rerank
    ranked = rerank(question, candidates, top_k=top_k)
    reranked_chunks = [candidates[i] for i, _ in ranked]
    reranked_metas = [metadatas[i] for i, _ in ranked]

    context = "\n\n---\n\n".join(reranked_chunks)
    prompt = f"""你是一个严谨的助手。请仅根据以下参考资料回答问题。
如果参考资料中没有答案，请直接说“根据提供的资料无法回答”。

参考资料：
{context}

问题：{question}
答案："""

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": reranked_metas,
        "contexts": reranked_chunks
    }


if __name__ == "__main__":
    print("开始测试 RAG 链路...")
    test_res = rag_query("什么是马尔可夫决策过程？")
    print("\n答案：", test_res["answer"])
    print("\n来源：", test_res["sources"])