import os
# 如果你在国内，网络下载模型容易失败，建议加上这行设置镜像（如果已经能科学上网则忽略）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer

def test_bge_model():
    print("正在加载 BGE-large-zh-v1.5 模型，首次运行会下载模型，请耐心等待...")
    
    # 1. 加载模型
    model_name = "BAAI/bge-large-zh-v1.5"
    model = SentenceTransformer(model_name)
    
    # 2. 准备一句测试文本
    test_text = "什么是文本切分？"
    
    # 3. 将文本转化为向量
    # normalize_embeddings=True 表示将向量归一化，这样计算余弦相似度时更准确，这是 BGE 模型的推荐用法
    embedding = model.encode(test_text, normalize_embeddings=True)
    
    # 4. 打印结果
    print(f"测试文本: {test_text}")
    print(f"生成的向量维度: {len(embedding)}")
    print(f"向量前5个数字: {embedding[:5]}")
    
    # 释放模型内存（测试用，实际建库时不需要）
    del model
    print("模型测试完毕！")

if __name__ == "__main__":
    test_bge_model()