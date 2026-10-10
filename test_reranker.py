import os
import csv
import time
import numpy as np

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base", device="cpu")


def compute_quality_score(candidates):
    """计算候选集质量评分：方差 + 一致性。"""
    vector_scores = [c["vector_score"] for c in candidates]
    variance = float(np.var(vector_scores))

    vec_top = [c["doc_id"] for c in sorted(candidates, key=lambda x: x["vector_score"], reverse=True)[:3]]
    bm25_top = [c["doc_id"] for c in sorted(candidates, key=lambda x: x["bm25_score"], reverse=True)[:3]]
    consistency = len(set(vec_top) & set(bm25_top)) / 3.0

    return variance, consistency


def recall_at_k(final_docs, relevant_ids, k=5):
    """计算 Recall@k。"""
    top_k_ids = [doc["doc_id"] for doc in final_docs[:k]]
    hit = sum(1 for doc_id in top_k_ids if doc_id in relevant_ids)
    return hit / len(relevant_ids) if relevant_ids else 0.0


def mrr(final_docs, relevant_ids):
    """计算单个 query 的 MRR。"""
    for rank, doc in enumerate(final_docs, start=1):
        if doc["doc_id"] in relevant_ids:
            return 1.0 / rank
    return 0.0


def pure_vector(query, candidates, top_k=5):
    """纯向量基线：只按 vector_score 排序。"""
    start = time.perf_counter()
    sorted_candidates = sorted(candidates, key=lambda x: x["vector_score"], reverse=True)
    final_docs = sorted_candidates[:top_k]
    latency_ms = (time.perf_counter() - start) * 1000
    return final_docs, latency_ms


def always_rerank(query, candidates, top_k=5):
    """始终 Rerank 基线。"""
    start = time.perf_counter()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    final_docs = [c for c, s in ranked[:top_k]]
    latency_ms = (time.perf_counter() - start) * 1000
    return final_docs, latency_ms


def conditional_rerank(query, candidates, top_k=5,
                       variance_threshold=0.01,
                       consistency_threshold=0.67):
    """条件 Rerank：只在候选集质量不达标时触发。"""
    start = time.perf_counter()

    if len(candidates) < 3:
        pairs = [(query, c["text"]) for c in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        final_docs = [c for c, s in ranked[:top_k]]
        latency_ms = (time.perf_counter() - start) * 1000
        quality_info = {
            "variance": None,
            "consistency": None,
            "trigger_reason": "候选数少于3，直接触发Rerank",
        }
        return final_docs, True, latency_ms, quality_info

    variance, consistency = compute_quality_score(candidates)

    triggered = False
    trigger_reason = ""

    if variance < variance_threshold:
        triggered = True
        trigger_reason = f"方差过低({variance:.4f} < {variance_threshold})，初筛区分度不足"
    elif consistency < consistency_threshold:
        triggered = True
        trigger_reason = f"一致性过低({consistency:.2f} < {consistency_threshold})，BM25与向量分歧大"
    else:
        trigger_reason = "候选集质量达标，无需Rerank"

    if triggered:
        pairs = [(query, c["text"]) for c in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        final_docs = [c for c, s in ranked[:top_k]]
    else:
        sorted_candidates = sorted(candidates, key=lambda x: x["vector_score"], reverse=True)
        final_docs = sorted_candidates[:top_k]

    latency_ms = (time.perf_counter() - start) * 1000

    quality_info = {
        "variance": variance,
        "consistency": consistency,
        "trigger_reason": trigger_reason,
    }

    return final_docs, triggered, latency_ms, quality_info


if __name__ == "__main__":
    # 预热模型，不计入指标
    reranker.predict([("warmup", "warmup doc")])

    # 用多个 mock query 做实验
    experiments = [
        {
            "query": "什么是条件 Rerank？",
            "candidates": [
                {"doc_id": "d1", "text": "条件 Rerank 只在候选集质量不够好时才触发。", "vector_score": 0.8, "bm25_score": 3.2},
                {"doc_id": "d2", "text": "今天天气很好。", "vector_score": 0.6, "bm25_score": 0.1},
                {"doc_id": "d3", "text": "Reranker 是一个 Cross-encoder 模型。", "vector_score": 0.7, "bm25_score": 2.5},
            ],
            "relevant_ids": {"d1", "d3"},
        },
        {
            "query": "Cross-encoder 是什么？",
            "candidates": [
                {"doc_id": "d1", "text": "Cross-encoder 把 query 和 doc 拼在一起打分。", "vector_score": 0.9, "bm25_score": 4.0},
                {"doc_id": "d2", "text": "今天天气很好。", "vector_score": 0.3, "bm25_score": 0.1},
                {"doc_id": "d3", "text": "条件 Rerank 只在候选集质量差时触发。", "vector_score": 0.5, "bm25_score": 1.0},
                {"doc_id": "d4", "text": "Cross-encoder 通常比 bi-encoder 更准，但更慢。", "vector_score": 0.85, "bm25_score": 3.5},
            ],
            "relevant_ids": {"d1", "d4"},
        },
    ]

    rows = []
    trigger_count = 0
    total_queries = 0

    for exp in experiments:
        query = exp["query"]
        candidates = exp["candidates"]
        relevant_ids = exp["relevant_ids"]

        total_queries += 1

        # 纯向量
        docs, latency = pure_vector(query, candidates, top_k=5)
        rows.append({
            "query": query, "method": "pure_vector",
            "recall@5": round(recall_at_k(docs, relevant_ids), 4),
            "mrr": round(mrr(docs, relevant_ids), 4),
            "latency_ms": round(latency, 2),
            "triggered": False,
            "trigger_reason": "",
        })

        # 始终 Rerank
        docs, latency = always_rerank(query, candidates, top_k=5)
        rows.append({
            "query": query, "method": "always_rerank",
            "recall@5": round(recall_at_k(docs, relevant_ids), 4),
            "mrr": round(mrr(docs, relevant_ids), 4),
            "latency_ms": round(latency, 2),
            "triggered": True,
            "trigger_reason": "always",
        })

        # 条件 Rerank
        docs, triggered, latency, qinfo = conditional_rerank(query, candidates, top_k=5)
        if triggered:
            trigger_count += 1
        rows.append({
            "query": query, "method": "conditional_rerank",
            "recall@5": round(recall_at_k(docs, relevant_ids), 4),
            "mrr": round(mrr(docs, relevant_ids), 4),
            "latency_ms": round(latency, 2),
            "triggered": triggered,
            "trigger_reason": qinfo["trigger_reason"],
        })

    # 写 CSV
    csv_path = "experiments/week3_day3_rerank_compare.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query", "method", "recall@5", "mrr", "latency_ms", "triggered", "trigger_reason"
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # 打印触发比例
    print(f"实验完成，结果已写入 {csv_path}")
    print(f"条件触发比例: {trigger_count}/{total_queries} = {trigger_count/total_queries:.2%}")