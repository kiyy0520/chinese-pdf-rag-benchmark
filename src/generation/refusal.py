import numpy as np

def compute_retrieval_confidence(similarities, alpha=1.0):
    """
    计算检索置信度。
    公式：均值 - alpha * 方差
    参数:
        similarities: list[float] Top-K 相似度分数
        alpha: float 惩罚系数，默认1.0
    返回:
        (retrieval_confidence, mean_sim, var_sim)
    """
    # 边界保护：如果检索结果为空，直接返回0置信度，触发后续拒答
    if not similarities:
        return 0.0, 0.0, 0.0
        
    mean_sim = np.mean(similarities)
    var_sim = np.var(similarities)
    retrieval_confidence = mean_sim - alpha * var_sim
    
    return retrieval_confidence, mean_sim, var_sim

def compute_generation_confidence(answer, retrieved_texts, embed_model):
    """
    计算生成置信度：LLM回答和检索内容的语义一致性。
    用余弦相似度衡量。
    参数:
        answer: str LLM生成的回答
        retrieved_texts: list[str] 检索到的Top-K文本列表
        embed_model: SentenceTransformer模型实例
    返回:
        float 生成置信度（0~1之间）
    """
    # 边界保护：回答或检索内容为空时，直接返回0
    if not answer or not retrieved_texts:
        return 0.0
        
    # 将多条检索内容拼成一段完整上下文
    context = "\n".join(retrieved_texts)
    
    # 编码为向量，并做L2归一化
    answer_vec = embed_model.encode(answer, normalize_embeddings=True)
    context_vec = embed_model.encode(context, normalize_embeddings=True)
    
    # 两个归一化向量的点积 = 余弦相似度
    gen_conf = float(np.dot(answer_vec, context_vec))
    
    # 防止浮点误差导致略微超过1，做一下裁剪
    gen_conf = min(max(gen_conf, 0.0), 1.0)
    
    return gen_conf

# 固定拒答模板（全项目统一使用）
REFUSAL_TEMPLATE = "根据现有知识库，我无法回答这个问题。"

def should_refuse(retrieval_confidence, generation_confidence,
                  retrieval_threshold=0.5,
                  generation_threshold=0.6):
    """
    双重拒答判断。
    返回 True 表示拒答，返回 False 表示正常回答。
    """
    # 防御性编程：如果传入的是NaN（前面讲过的捣蛋鬼），直接拒答
    if np.isnan(retrieval_confidence) or np.isnan(generation_confidence):
        return True
        
    # 第一道防线：检索置信度低
    if retrieval_confidence < retrieval_threshold:
        return True
        
    # 第二道防线：生成置信度低
    if generation_confidence < generation_threshold:
        return True
        
    # 两道防线都通过，正常回答
    return False