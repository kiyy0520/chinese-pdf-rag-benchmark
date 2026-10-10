# src/evaluation/run_week3_day3_sweep.py
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
)


# 定义 3 组阈值
CONFIGS = {
    "loose":    {"variance_threshold": 0.005, "consistency_threshold": 0.33},
    "baseline": {"variance_threshold": 0.01,  "consistency_threshold": 0.66},
    "strict":   {"variance_threshold": 0.02,  "consistency_threshold": 1.0},
}


def main():
    # 1. 连接 Chroma
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("day4_chunking_eval")

    # 2. 初始化检索器（统一用 markdown_recursive_512_0）
    CONFIG_NAME = "markdown_recursive_512_0"
    vector_retriever = VectorRetriever(collection, config_name=CONFIG_NAME)
    bm25_retriever = BM25Retriever(collection, config_name=CONFIG_NAME)
    reranker = ConditionalReranker(collection, vector_retriever, bm25_retriever)

    # 3. 加载评估集
    with open("data/golden_qa/qa.json", "r", encoding="utf-8") as f:
        qa_list = json.load(f)

    # 4. 为每组阈值准备一个 rows 列表 + 触发计数器
    all_rows = {name: [] for name in CONFIGS}
    trigger_counts = {name: 0 for name in CONFIGS}
    total_queries = 0

    # 5. 遍历 query
    for idx, qa in enumerate(qa_list, start=1):
        query = qa["question"]
        source_context = qa["source_context"]

        total_queries += 1

        # 关键：候选集只构建一次，3 组共用
        candidates = reranker.build_candidates(query, candidate_k=20)

        # 对每组阈值，各跑一次条件 Rerank
        for name, cfg in CONFIGS.items():
            docs, triggered, latency, qinfo = reranker.conditional_rerank(
                query, candidates, top_k=5,
                variance_threshold=cfg["variance_threshold"],
                consistency_threshold=cfg["consistency_threshold"],
            )
            if triggered:
                trigger_counts[name] += 1

            all_rows[name].append({
                "query": query,
                "method": "conditional_rerank",
                "recall@5": round(recall_at_k_text(docs, source_context), 4),
                "mrr": round(mrr_text(docs, source_context), 4),
                "latency_ms": round(latency, 2),
                "triggered": triggered,
                "trigger_reason": qinfo["trigger_reason"],
            })

    # 6. 每组输出独立 CSV
    os.makedirs("experiments", exist_ok=True)
    for name, rows in all_rows.items():
        csv_path = f"experiments/week3_day3_rerank_{name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "query", "method", "recall@5", "mrr", "latency_ms", "triggered", "trigger_reason"
            ])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    # 7. 打印汇总
    print("\n=== 阈值扫描结果 ===")
    print(f"{'组名':<10} {'触发比例':<10} {'平均Recall@5':<15} {'平均MRR':<10} {'平均延迟(ms)':<12}")
    for name, rows in all_rows.items():
        n = len(rows)
        avg_recall = sum(r["recall@5"] for r in rows) / n
        avg_mrr = sum(r["mrr"] for r in rows) / n
        avg_latency = sum(r["latency_ms"] for r in rows) / n
        ratio = trigger_counts[name] / total_queries
        print(f"{name:<10} {ratio:<10.2%} {avg_recall:<15.4f} {avg_mrr:<10.4f} {avg_latency:<12.2f}")


if __name__ == "__main__":
    main()