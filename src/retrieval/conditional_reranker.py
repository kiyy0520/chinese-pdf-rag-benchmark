# src/retrieval/conditional_reranker.py
import os
import time
import numpy as np
import jieba
import jieba.posseg as pseg

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import CrossEncoder


def _extract_keywords(text, top_n=3):
    """从文本中提取关键词，只保留名词/专有名词/英文词，按长度排序。"""
    allowed_flags = {"n", "nr", "ns", "nt", "nz", "eng", "vn"}
    words = []
    for word, flag in pseg.cut(text):
        if len(word) < 2:
            continue
        if flag in allowed_flags:
            words.append(word)
    words = sorted(set(words), key=len, reverse=True)
    return words[:top_n]


def recall_at_k_text(final_docs, source_context, k=5):
    """基于关键词命中的 Recall@k。"""
    keywords = _extract_keywords(source_context)
    if not keywords:
        return 0.0
    top_k_texts = [doc["text"] for doc in final_docs[:k]]
    for text in top_k_texts:
        hit = sum(1 for kw in keywords if kw in text)
        if hit >= 2:
            return 1.0
    return 0.0


def mrr_text(final_docs, source_context):
    """基于关键词命中的 MRR。"""
    keywords = _extract_keywords(source_context)
    if not keywords:
        return 0.0
    for rank, doc in enumerate(final_docs, start=1):
        hit = sum(1 for kw in keywords if kw in doc["text"])
        if hit >= 2:
            return 1.0 / rank
    return 0.0


class ConditionalReranker:
    def __init__(self, collection, vector_retriever, bm25_retriever):
        self.collection = collection
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = CrossEncoder("BAAI/bge-reranker-base", device="cuda")

        # 预热，避免首次调用延迟虚高
        self.reranker.predict([("warmup", "warmup doc")])

    def build_candidates(self, query, candidate_k=20):
        """合并向量和BM25的候选，并补充文本，生成统一格式的 candidates。"""
        vec_results = self.vector_retriever.retrieve(query, top_k=candidate_k)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=candidate_k)

        vec_dict = {doc_id: score for doc_id, score in vec_results}
        bm25_dict = {doc_id: score for doc_id, score in bm25_results}

        all_ids = list(set(vec_dict.keys()) | set(bm25_dict.keys()))

        # 批量从 Chroma 拉取文本
        res = self.collection.get(ids=all_ids, include=["documents"])
        id_to_text = dict(zip(res["ids"], res["documents"]))

        candidates = []
        for doc_id in all_ids:
            candidates.append({
                "doc_id": doc_id,
                "text": id_to_text.get(doc_id, ""),
                "vector_score": vec_dict.get(doc_id, 0.0),
                "bm25_score": bm25_dict.get(doc_id, 0.0),
            })
        return candidates

    def compute_quality_score(self, candidates):
        vector_scores = [c["vector_score"] for c in candidates]
        variance = float(np.var(vector_scores))

        vec_top = [c["doc_id"] for c in sorted(candidates, key=lambda x: x["vector_score"], reverse=True)[:3]]
        bm25_top = [c["doc_id"] for c in sorted(candidates, key=lambda x: x["bm25_score"], reverse=True)[:3]]
        consistency = len(set(vec_top) & set(bm25_top)) / 3.0

        return variance, consistency

    def pure_vector(self, query, candidates, top_k=5):
        start = time.perf_counter()
        sorted_candidates = sorted(candidates, key=lambda x: x["vector_score"], reverse=True)
        final_docs = sorted_candidates[:top_k]
        latency_ms = (time.perf_counter() - start) * 1000
        return final_docs, latency_ms

    def always_rerank(self, query, candidates, top_k=5):
        start = time.perf_counter()
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        final_docs = [c for c, s in ranked[:top_k]]
        latency_ms = (time.perf_counter() - start) * 1000
        return final_docs, latency_ms

    def conditional_rerank(self, query, candidates, top_k=5,
                           variance_threshold=0.01,
                           consistency_threshold=0.66):
        start = time.perf_counter()

        if len(candidates) < 3:
            pairs = [(query, c["text"]) for c in candidates]
            scores = self.reranker.predict(pairs)
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            final_docs = [c for c, s in ranked[:top_k]]
            latency_ms = (time.perf_counter() - start) * 1000
            quality_info = {
                "variance": None,
                "consistency": None,
                "trigger_reason": "候选数少于3，直接触发Rerank",
            }
            return final_docs, True, latency_ms, quality_info

        variance, consistency = self.compute_quality_score(candidates)

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
            scores = self.reranker.predict(pairs)
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