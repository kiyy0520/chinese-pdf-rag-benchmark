import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import chromadb
from sentence_transformers import SentenceTransformer

def ingest():
    # 1. 读取 Markdown 文件
    with open("data/kb/company_policy.md", "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # 2. 提取正文内容（去掉标题行）
    documents = []
    for line in lines:
        if line.startswith("#"):
            continue
        documents.append(line)
    
    print(f"共提取 {len(documents)} 条知识片段。")
    
    # 3. 加载嵌入模型
    model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
    embeddings = model.encode(documents, normalize_embeddings=True).tolist()
    
    # 4. 连接 ChromaDB 并创建新集合
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(name="company_policy_kb")
    
    # 5. 写入数据（注意：metadata 里必须带上 config_name，否则检索器的 where 过滤会失效）
    ids = [f"policy_{i}" for i in range(len(documents))]
    metadatas = [{"config_name": "company_policy"} for _ in documents]
    
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print("数据摄入完成！")

if __name__ == "__main__":
    ingest()