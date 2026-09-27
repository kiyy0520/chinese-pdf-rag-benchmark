# src/retrieval/bm25_retriever.py
import jieba
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, collection, config_name="fixed_512_0"):
        # 1. 从 Chroma 拉取指定配置下的所有 chunks
        res = collection.get(
            where={"config_name": config_name},
            include=["documents", "metadatas"]
        )
        self.doc_ids = res["ids"]
        self.corpus = res["documents"]
        
        # 2. 分词（中文用 jieba，英文用空格）
        # 注意：为了让 BM25 更好的理解中文，可以针对文档建一个分词后的列表
        tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
        
        # 3. 建立 BM25 索引
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query, top_k=5):
        # 1. 对 query 分词
        tokenized_query = list(jieba.cut(query))
        # 2. 获取所有文档的 BM25 分数
        scores = self.bm25.get_scores(tokenized_query)
        # 3. 按分数从高到低排序，取前 top_k 个
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        # 4. 返回 [(doc_id, score), ...]
        return [(self.doc_ids[i], scores[i]) for i in top_n]