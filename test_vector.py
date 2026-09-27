import chromadb
from src.retrieval.vector_retriever import VectorRetriever

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="day4_chunking_eval")

# 实例化向量检索器
vector_retriever = VectorRetriever(collection, config_name="recursive_512_0")

# 用完全相同的测试问题
test_query = "马尔可夫决策过程的奖励函数是什么？"
results = vector_retriever.retrieve(test_query, top_k=5)

print(f"查询: {test_query}")
print(f"共检索到 {len(results)} 条结果：")
for rank, (doc_id, score) in enumerate(results):
    print(f"  第 {rank+1} 名 | 分数: {score:.4f} | ID: {doc_id[:30]}...")