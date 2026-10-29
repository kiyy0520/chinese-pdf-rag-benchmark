import chromadb
from src.retrieval.hybrid_retriever import HybridRetriever

def print_top5():
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection(name="day4_chunking_eval")
    retriever = HybridRetriever(collection, config_name="recursive_512_0", rrf_k=60, weights=(1.0, 1.0))

    cases = [
        ("那它的状态转移概率呢？", "马尔可夫决策过程 状态转移概率"),
        ("这个模型适合解决什么具体问题？", "马尔可夫决策过程 应用场景")
    ]

    for raw, rewrite in cases:
        print(f"\n==================================================")
        print(f"👤 原始 Query: {raw}")
        print(f"🔄 改写 Query: {rewrite}")
        
        for mode, query in [("无 Rewrite", raw), ("有 Rewrite", rewrite)]:
            results = retriever.retrieve(query, top_k=5)
            ids = [r[0] for r in results]
            docs = collection.get(ids=ids)["documents"]
            print(f"\n【{mode}】Top-5 文本片段：")
            for i, doc in enumerate(docs):
                clean_doc = doc.replace("\n", "").strip()[:80]
                print(f"  {i+1}. {clean_doc}...")

if __name__ == "__main__":
    print_top5()