import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import time
import re
import unicodedata
from difflib import SequenceMatcher
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# 1. 定义 9 种配置
configs = [
    "fixed_512_0", "fixed_512_50", "fixed_1024_0",
    "markdown_recursive_512_0", "markdown_recursive_512_100",
    "markdown_recursive_1024_0", "markdown_recursive_1024_50",
    "markdown_recursive_2048_0", "recursive_512_0"
]

# 2. 终极文本归一化函数
def normalize_text(text):
    if not text:
        return ""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[^\w\u4e00-\u9fa5]', '', text)
    return text.lower()

# 3. 模糊匹配逻辑
def is_hit(source_context, retrieved_chunk):
    norm_ctx = normalize_text(source_context)
    norm_ck = normalize_text(retrieved_chunk)
    
    if norm_ctx in norm_ck:
        return True
        
    similarity = SequenceMatcher(None, norm_ctx, norm_ck).ratio()
    return similarity >= 0.6 

def run_evaluation():
    print("加载 Golden QA 数据...")
    with open("data/golden_qa/qa.json", "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    print("连接 Chroma 持久化库...")
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection(name="day4_chunking_eval")

    print("加载 Embedding 模型（用于给提问做向量化）...")
    model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

    results = []
    print("开始 9 组配置 × 20 条 QA 的检索评估...")

    for config in configs:
        for qa in qa_data:
            question = qa["question"]
            source_context = qa["source_context"]

            query_emb = model.encode(question, normalize_embeddings=True).tolist()

            start_time = time.time()
            query_res = collection.query(
                query_embeddings=[query_emb],
                n_results=5,
                where={"config_name": config}
            )
            latency = (time.time() - start_time) * 1000

            retrieved_chunks = query_res["documents"][0]
            
            hit = False
            rank = 0
            for i, chunk in enumerate(retrieved_chunks):
                if is_hit(source_context, chunk):
                    hit = True
                    rank = i + 1
                    break

            results.append({
                "config_name": config,
                "question": question,
                "hit": hit,
                "rank": rank if hit else 0,
                "latency_ms": latency,
                "embedding_dim": len(query_emb)
            })
        print(f"  ✅ {config} 评估完成")

    # ================= 核心聚合逻辑（全在函数内部） =================
    df = pd.DataFrame(results)
    
    summary = []
    for config in configs:
        sub_df = df[df["config_name"] == config]
        recall_at_5 = sub_df["hit"].mean()
        mrr = sub_df.apply(lambda row: 1.0 / row["rank"] if row["hit"] else 0, axis=1).mean()
        avg_latency = sub_df["latency_ms"].mean()
        
        summary.append({
            "config_name": config,
            "recall@5": round(recall_at_5, 4),
            "mrr": round(mrr, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "total_chunks_evaluated": len(sub_df)
        })
        
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("experiments/day4_chunking_eval.csv", index=False, encoding="utf-8-sig")
    
    print("\n" + "="*50)
    print("📊 Day 4 切分策略评估结果（已保存至 experiments/day4_chunking_eval.csv）")
    print("="*50)
    print(summary_df.to_string(index=False))
    print("="*50)
    
    best_config = summary_df.sort_values(by=["recall@5", "mrr"], ascending=False).iloc[0]
    print(f"\n🏆 最优切分策略为：{best_config['config_name']}")
    print(f"   Recall@5: {best_config['recall@5']} | MRR: {best_config['mrr']} | 检索延迟: {best_config['avg_latency_ms']}ms")
    
    badcases = df[(df["hit"] == False) & (df["config_name"] == best_config['config_name'])]
    print(f"\n⚠️ 最优策略下的错误检索（badcase）数量：{len(badcases)}")
    if not badcases.empty:
        qa_dict = {qa["question"]: qa["source_context"] for qa in qa_data}
        print("以下是部分 badcase 示例：")
        for idx, row in badcases.head(2).iterrows():
            print(f"\n  ❌ 问题：{row['question']}")
            correct_context = qa_dict.get(row['question'], "未知")
            print(f"     (正确依据：{correct_context[:50]}...)")

if __name__ == "__main__":
    run_evaluation()