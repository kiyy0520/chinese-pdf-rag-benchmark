# src/retrieval/vector_retriever.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import chromadb
from sentence_transformers import SentenceTransformer

class VectorRetriever:
    # ⚠️ 新增了 config_name 参数
    def __init__(self, collection, config_name="recursive_512_0", model_name="BAAI/bge-large-zh-v1.5"):
        self.collection = collection
        self.config_name = config_name  # 保存配置名
        self.model = SentenceTransformer(model_name)

    def retrieve(self, query, top_k=5):
        query_emb = self.model.encode(query, normalize_embeddings=True).tolist()
        
        # ⚠️ 关键修正：加上 where 过滤条件，只从指定的切分配置里检索！
        query_res = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where={"config_name": self.config_name}  
        )
        
        doc_ids = query_res["ids"][0]
        distances = query_res["distances"][0]
        scores = [1.0 - d for d in distances]
        
        return list(zip(doc_ids, scores))