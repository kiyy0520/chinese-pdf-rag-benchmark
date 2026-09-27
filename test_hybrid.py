import chromadb
from src.retrieval.hybrid_retriever import HybridRetriever

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="day4_chunking_eval")

# 实例化混合检索器
hybrid_retriever = HybridRetriever(
    collection, 
    config_name="recursive_512_0",
    rrf_k=60,
    weights=(1.0, 1.0)  # 先使用默认权重
)

test_query = "马尔可夫决策过程的奖励函数是什么？"
results = hybrid_retriever.retrieve(test_query, top_k=5, candidate_k=20)

print(f"查询: {test_query}")
print(f"混合检索前 5 名：")
for rank, (doc_id, score) in enumerate(results):
    print(f"  第 {rank+1} 名 | RRF分数: {score:.6f} | ID: {doc_id}")