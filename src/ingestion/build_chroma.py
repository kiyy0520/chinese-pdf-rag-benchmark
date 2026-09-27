import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 确保国内网络可用
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # 屏蔽之前那个符号链接警告

import time
import chromadb
from sentence_transformers import SentenceTransformer
from src.ingestion.prepare_data import load_and_prepare

def build_database():
    # 1. 加载我们刚才准备好的数据
    print("正在读取数据...")
    all_ids, all_documents, all_metadatas = load_and_prepare()
    
    # 2. 加载 BGE 模型
    print("正在加载 Embedding 模型...")
    model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
    
    # 3. 计算嵌入向量并计时
    print(f"开始批量嵌入 {len(all_documents)} 个 chunk...")
    start_time = time.time()  # 开始计时
    
    embeddings = model.encode(
        all_documents, 
        batch_size=32,           # 每次喂给模型 32 条
        show_progress_bar=True,  # 显示进度条
        normalize_embeddings=True
    ).tolist()                   # 转成 list，Chroma 要求这种格式
    
    embed_time = time.time() - start_time
    print(f"✅ 嵌入完成，耗时 {embed_time:.2f} 秒")
    
    # 4. 初始化 Chroma 持久化客户端
    # 这里 path 用的是你项目里的 chroma_db 目录
    print("正在连接 Chroma 并建立持久化库...")
    client = chromadb.PersistentClient(path="chroma_db")
    
    # 5. 创建或获取一个集合（Collection）
    # 如果之前跑过，为了避免重复，我们先删掉旧的（或者你可以换个名字）
    try:
        client.delete_collection(name="day4_chunking_eval")
    except Exception:
        pass
    collection = client.create_collection(name="day4_chunking_eval")
    
    # 6. 将数据插入 Chroma
    collection.add(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_documents,
        metadatas=all_metadatas
    )
    
    print(f"✅ Chroma 建库完成！集合名：day4_chunking_eval")
    print(f"✅ 当前库中总数据量：{collection.count()} 条")
    print(f"✅ 数据库已保存至 chroma_db/ 目录，下次可直接读取。")

if __name__ == "__main__":
    build_database()