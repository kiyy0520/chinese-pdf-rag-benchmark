import chromadb
from src.retrieval.bm25_retriever import BM25Retriever

# 1. 连接你的 Chroma
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="day4_chunking_eval")

# 2. 实例化 BM25（⚠️ 这里的 config_name 换成你 Day 4 评估出的最优配置）
# 假设你的最优配置是 "markdown_recursive_512_0"
best_config = "recursive_512_0" 
bm25_retriever = BM25Retriever(collection, config_name=best_config)

# 3. 用一条 QA 里的问题测试一下
test_query = "马尔可夫决策过程的奖励函数是什么？"  # 可以换成你 qa.json 里的真实问题
results = bm25_retriever.retrieve(test_query, top_k=5)

# 4. 打印结果
print(f"查询: {test_query}")
print(f"共检索到 {len(results)} 条结果：")
for rank, (doc_id, score) in enumerate(results):
    print(f"  第 {rank+1} 名 | 分数: {score:.4f} | ID: {doc_id[:30]}...")