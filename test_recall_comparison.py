import os
import json
import chromadb
from dotenv import load_dotenv
from src.retrieval.hybrid_retriever import HybridRetriever

load_dotenv()

def evaluate_recall():
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection(name="day4_chunking_eval")
    retriever = HybridRetriever(collection, config_name="recursive_512_0", rrf_k=60, weights=(1.0, 1.0))

    # 测试集：原始 Query（多轮场景中的省略式追问） vs 改写后的 Query
    test_cases = [
        {
            "raw": "那它的状态转移概率呢？",
            "rewrite": "马尔可夫决策过程 状态转移概率",
            "expected_keyword": "状态转移概率" # 用来简单判断是否召回了正确文档
        },
        {
            "raw": "这个模型适合解决什么具体问题？",
            "rewrite": "马尔可夫决策过程 应用场景",
            "expected_keyword": "应用场景"
        }
    ]

    results = []
    for case in test_cases:
        print(f"\n测试用例: {case['raw']}")
        
        # 1. 无 Rewrite 检索
        raw_res = retriever.retrieve(case['raw'], top_k=5, candidate_k=20)
        raw_ids = [doc_id for doc_id, score in raw_res]
        raw_docs = collection.get(ids=raw_ids)["documents"]
        raw_hit = any(case['expected_keyword'] in doc for doc in raw_docs)
        
        # 2. 有 Rewrite 检索
        rewrite_res = retriever.retrieve(case['rewrite'], top_k=5, candidate_k=20)
        rewrite_ids = [doc_id for doc_id, score in rewrite_res]
        rewrite_docs = collection.get(ids=rewrite_ids)["documents"]
        rewrite_hit = any(case['expected_keyword'] in doc for doc in rewrite_docs)
        
        results.append({
            "query": case['raw'],
            "raw_hit@5": "✅ 命中" if raw_hit else "❌ 未命中",
            "rewrite_hit@5": "✅ 命中" if rewrite_hit else "❌ 未命中"
        })
        
        print(f"  无 Rewrite (直接搜原话): {results[-1]['raw_hit@5']}")
        print(f"  有 Rewrite (搜索略词): {results[-1]['rewrite_hit@5']}")

    print("\n=== Recall@5 对比结果 ===")
    print(f"{'原始 Query':<20} | {'无 Rewrite':<10} | {'有 Rewrite':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['query']:<20} | {r['raw_hit@5']:<12} | {r['rewrite_hit@5']:<12}")

if __name__ == "__main__":
    evaluate_recall()