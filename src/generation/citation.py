# src/generation/citation.py

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# ============ 相似度方法开关 ============
# 可选值："jaccard" / "embedding" / "tfidf"
SIMILARITY_METHOD = "tfidf"

# ============ embedding 模型（懒加载） ============
_MODEL_NAME = "shibing624/text2vec-base-chinese"
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ============ 方法 1：Jaccard ============
def similarity_jaccard(text1, text2):
    for ch in "，。！？、；：''（）【】":
        text1 = text1.replace(ch, "")
        text2 = text2.replace(ch, "")

    set1 = set(text1)
    set2 = set(text2)

    if not set1 or not set2:
        return 0.0

    return len(set1 & set2) / len(set1 | set2)


# ============ 方法 2：embedding + 余弦相似度 ============
def similarity_embedding(text1, text2):
    model = get_model()
    embeddings = model.encode([text1, text2], normalize_embeddings=True)
    sim = float(np.dot(embeddings[0], embeddings[1]))
    return max(0.0, min(1.0, sim))


# ============ 方法 3：TF-IDF + 余弦相似度 ============
def similarity_tfidf(text1, text2):
    """用 TF-IDF 按字切分，再算余弦相似度。"""
    # 空文本保护
    if not text1 or not text2:
        return 0.0

    # 把两句话当语料，用字符级 TF-IDF 向量化
    vectorizer = TfidfVectorizer(analyzer="char")
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # 用 sklearn 直接算余弦相似度
    sim = sklearn_cosine(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return max(0.0, min(1.0, float(sim)))


# ============ 统一入口 ============
def similarity(text1, text2):
    """根据开关选择相似度方法。"""
    if SIMILARITY_METHOD == "jaccard":
        return similarity_jaccard(text1, text2)
    elif SIMILARITY_METHOD == "embedding":
        return similarity_embedding(text1, text2)
    elif SIMILARITY_METHOD == "tfidf":
        return similarity_tfidf(text1, text2)
    else:
        raise ValueError(f"未知的相似度方法：{SIMILARITY_METHOD}")


def attach_citation(answer, sources):
    best_source = None
    best_score = -1

    for source in sources:
        score = similarity(answer, source["text"])
        if score > best_score:
            best_score = score
            best_source = source

    return {
        "answer": answer,
        "source": best_source,
        "score": best_score
    }


if __name__ == "__main__":
    # 基础测试
    answer = "公司年假是10天"
    sources = [
        {"id": "doc_01", "text": "正式员工每年享有10天年假"},
        {"id": "doc_02", "text": "员工每年有5天年假"},
        {"id": "doc_03", "text": "年假申请需提前三天"},
    ]
    result = attach_citation(answer, sources)
    print("测试 1 - 正常情况：")
    print(result)
    print()

    result_empty = attach_citation(answer, [])
    print("测试 2 - 空来源列表：")
    print(result_empty)
    print()

    answer2 = "员工每年有5天年假"
    result_exact = attach_citation(answer2, sources)
    print("测试 3 - 完全匹配 doc_02：")
    print(result_exact)
    print()

    # 三种方法对比
    pairs = [
        ("公司年假是10天", "正式员工每年享有10天年假"),
        ("员工每年有5天年假", "员工每年有5天年假"),
        ("年假申请需提前三天", "年假天数按工龄计算"),
    ]

    print("=== 三种方法对比 ===")
    for t1, t2 in pairs:
        j = similarity_jaccard(t1, t2)
        e = similarity_embedding(t1, t2)
        t = similarity_tfidf(t1, t2)
        print(f"{t1} vs {t2}")
        print(f"  Jaccard:   {j:.4f}")
        print(f"  embedding: {e:.4f}")
        print(f"  TF-IDF:    {t:.4f}")
        print()