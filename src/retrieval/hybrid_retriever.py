# src/retrieval/hybrid_retriever.py
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever

class HybridRetriever:
    def __init__(self, collection, config_name="recursive_512_0", rrf_k=60, weights=(1.0, 1.0)):
        self.bm25 = BM25Retriever(collection, config_name=config_name)
        self.vector = VectorRetriever(collection, config_name=config_name)
        self.rrf_k = rrf_k         # RRF 的 k 值
        self.weights = weights     # (BM25 权重, 向量权重)

    def retrieve(self, query, top_k=5, candidate_k=20):
        # 1. 两路各取候选
        bm25_results = self.bm25.retrieve(query, top_k=candidate_k)
        vector_results = self.vector.retrieve(query, top_k=candidate_k)

        # 2. 计算 RRF 分数
        rrf_scores = {}
        
        # 处理 BM25 结果
        for rank, (doc_id, _) in enumerate(bm25_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + self.weights[0] * (1.0 / (self.rrf_k + rank))

        # 处理向量结果
        for rank, (doc_id, _) in enumerate(vector_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + self.weights[1] * (1.0 / (self.rrf_k + rank))

        # 3. 按 RRF 分数从高到低排序，取前 top_k
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return sorted_docs  # 返回 [(doc_id, rrf_score), ...]