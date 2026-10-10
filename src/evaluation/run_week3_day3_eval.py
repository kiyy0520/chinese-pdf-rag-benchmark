# src/evaluation/run_day3_eval.py
import os
import csv
import json

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chromadb
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.conditional_reranker import (
    ConditionalReranker,
    recall_at_k_text,
    mrr_text,
    _extract_keywords,
)


def main():
    # 1. 连接 Chroma
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("day4_chunking_eval")

    # 2. 初始化检索器（使用 collection 里实际存在的 config_name）
    CONFIG = "markdown_recursive_512_0"
    vector_retriever = VectorRetriever(collection, config_name=CONFIG)
    bm25_retriever = BM25Retriever(collection, config_name=CONFIG)
    reranker = ConditionalReranker(collection, vector_retriever, bm25_retriever)

    # 3. 加载评估集
    with open("data/golden_qa/qa.json", "r", encoding="utf-8") as f:
        qa_list = json.load(f)

    rows = []
    trigger_count = 0
    total_queries = 0

    # 先用前 20 条跑通，稳定后再全量
        # 先用前 5 条跑通，稳定后再全量
    for idx, qa in enumerate(qa_list, start=1):
        query = qa["question"]
        source_context = qa["source_context"]

        print(f"[{idx}/5] 处理中: {query[:30]}...")

        total_queries += 1

        # 构建候选集
        candidates = reranker.build_candidates(query, candidate_k=20)

        # ===== 新增调试：只看第一个 query 的候选集 =====
        #if idx <= 2:
        #    keywords = _extract_keywords(source_context)
        #    print(f"\n=== 调试 query {idx} ===")
        #    print(f"source_context: {source_context}")
        #    print(f"提取的关键词: {keywords}")
        #    print(f"候选数: {len(candidates)}")
        #    for c in candidates[:5]:
        #        hit = sum(1 for kw in keywords if kw in c["text"])
        #        print(f"  {c['doc_id']} 命中 {hit}/{len(keywords)} 个关键词")
        #        print(f"    text: {c['text'][:100]}")
        # ==============================================

        # 纯向量
        docs, latency = reranker.pure_vector(query, candidates, top_k=5)
        rows.append({
            "query": query, "method": "pure_vector",
            "recall@5": round(recall_at_k_text(docs, source_context), 4),
            "mrr": round(mrr_text(docs, source_context), 4),
            "latency_ms": round(latency, 2),
            "triggered": False,
            "trigger_reason": "",
        })

        # 始终 Rerank
        docs, latency = reranker.always_rerank(query, candidates, top_k=5)
        rows.append({
            "query": query, "method": "always_rerank",
            "recall@5": round(recall_at_k_text(docs, source_context), 4),
            "mrr": round(mrr_text(docs, source_context), 4),
            "latency_ms": round(latency, 2),
            "triggered": True,
            "trigger_reason": "always",
        })

        # 条件 Rerank
        docs, triggered, latency, qinfo = reranker.conditional_rerank(query, candidates, top_k=5)
        if triggered:
            trigger_count += 1
        rows.append({
            "query": query, "method": "conditional_rerank",
            "recall@5": round(recall_at_k_text(docs, source_context), 4),
            "mrr": round(mrr_text(docs, source_context), 4),
            "latency_ms": round(latency, 2),
            "triggered": triggered,
            "trigger_reason": qinfo["trigger_reason"],
        })

    # 4. 写 CSV
    csv_path = "experiments/week3_day3_rerank_compare_markdown.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query", "method", "recall@5", "mrr", "latency_ms", "triggered", "trigger_reason"
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"实验完成，结果已写入 {csv_path}")
    print(f"条件触发比例: {trigger_count}/{total_queries} = {trigger_count/total_queries:.2%}")


if __name__ == "__main__":
    main()