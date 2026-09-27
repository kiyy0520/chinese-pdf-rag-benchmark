# src/retrieval/rerank.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from FlagEmbedding import FlagReranker

_reranker = None

# ModelScope 下载的本地路径（注意多一层 models，且 BAAI--bge-reranker-large 是双横线）
LOCAL_MODEL_PATH = r"F:\Projects\chinese-pdf-rag-benchmark\models\models\BAAI--bge-reranker-large\snapshots\master"


def get_reranker():
    global _reranker
    if _reranker is None:
        print(f"正在加载 BGE Reranker 模型（本地）: {LOCAL_MODEL_PATH}")
        if not os.path.exists(LOCAL_MODEL_PATH):
            raise FileNotFoundError(f"本地模型路径不存在：{LOCAL_MODEL_PATH}")
        _reranker = FlagReranker(LOCAL_MODEL_PATH, use_fp16=True)
    return _reranker


def rerank(query: str, candidates: list[str], top_k: int = 3):
    reranker = get_reranker()
    pairs = [[query, c] for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True)

    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return indexed[:top_k]