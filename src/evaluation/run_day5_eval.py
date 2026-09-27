# src/evaluation/run_day5_eval.py
import os
import sys
# 把项目根目录加入 Python 的搜索路径，保证无论从哪里运行都能找到 src 包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import jieba
import chromadb
from sentence_transformers import SentenceTransformer
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.hybrid_retriever import HybridRetriever

# ⚠️ 从你 Day 4 代码里直接复制这两个函数过来！
import re
import unicodedata
from difflib import SequenceMatcher

def normalize_text(text):
    if not text:
        return ""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[^\w\u4e00-\u9fa5]', '', text)
    return text.lower()

def is_hit_by_question(question, retrieved_chunk, threshold=0.5):
    """用问题中的关键词去匹配检索出的 chunk，避免公式和排版干扰"""
    # 提取问题中的词，只保留长度 >= 2 的词（过滤掉“的”、“什么”等无意义词）
    q_words = [w for w in jieba.lcut(question) if len(w) >= 2]
    if not q_words:
        return False
    
    # 统计这些关键词在 chunk 中出现的比例
    hit_count = sum(1 for w in q_words if w in retrieved_chunk)
    return (hit_count / len(q_words)) >= threshold 

def get_docs_by_ids(collection, doc_ids):
    """根据 ID 列表，从 Chroma 里捞出对应的文本内容"""
    if not doc_ids:
        return []
    res = collection.get(ids=doc_ids, include=["documents"])
    # 按照传入的 doc_ids 顺序返回文本
    id_to_doc = dict(zip(res["ids"], res["documents"]))
    return [id_to_doc.get(doc_id, "") for doc_id in doc_ids]

def evaluate_hit(docs, question):
    """判断这一路的 Top5 里，有没有命中正确文本，返回 (是否命中, 排名)"""
    for i, doc in enumerate(docs):
        if is_hit_by_question(question, doc):
            return True, i + 1
    return False, 0

def run_day5_evaluation():
    print("加载 Golden QA 数据...")
    with open("data/golden_qa/qa.json", "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    print("连接 Chroma 持久化库并初始化三种检索器...")
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection(name="day4_chunking_eval")
    
    # 统一配置，保证公平对比
    CONFIG_NAME = "recursive_512_0"
    
    # 实例化 3 种检索方案
    bm25_retriever = BM25Retriever(collection, config_name=CONFIG_NAME)
    vector_retriever = VectorRetriever(collection, config_name=CONFIG_NAME)
    hybrid_retriever = HybridRetriever(collection, config_name=CONFIG_NAME, rrf_k=10, weights=(1.0, 0.5))
    
    print(f"配置: {CONFIG_NAME} | RRF k值: 60 | 权重: (1.0, 1.0) | 候选数: 20")
    print("开始评估 20 条 QA...")
    
    # 用来存结果的字典
    results = {
        "vector": {"hit": [], "rank": []},
        "bm25": {"hit": [], "rank": []},
        "hybrid": {"hit": [], "rank": []}
    }
    
        # 诊断模式：打印每条 QA 的命中情况
    for idx, qa in enumerate(qa_data):
        question = qa["question"]
        source_context = qa["source_context"]
        
        v_results = vector_retriever.retrieve(question, top_k=5)
        v_docs = get_docs_by_ids(collection, [doc_id for doc_id, _ in v_results])
        v_hit, v_rank = evaluate_hit(v_docs, question)
        
        b_results = bm25_retriever.retrieve(question, top_k=5)
        b_docs = get_docs_by_ids(collection, [doc_id for doc_id, _ in b_results])
        b_hit, b_rank = evaluate_hit(b_docs, question)
        
        h_results = hybrid_retriever.retrieve(question, top_k=5, candidate_k=10)
        h_docs = get_docs_by_ids(collection, [doc_id for doc_id, _ in h_results])
        h_hit, h_rank = evaluate_hit(h_docs, question)
        
        # 打印每条 QA 的三种方案命中情况
        print(f"\n[{idx+1}] {question}")
        print(f"  source_context 前50字: {source_context[:50]}")
        print(f"  VECTOR: {'✅' if v_hit else '❌'} rank={v_rank}")
        print(f"  BM25:   {'✅' if b_hit else '❌'} rank={b_rank}")
        print(f"  HYBRID: {'✅' if h_hit else '❌'} rank={h_rank}")
        
                # 记录结果，用于后续汇总
        results["vector"]["hit"].append(v_hit)
        results["vector"]["rank"].append(v_rank)
        results["bm25"]["hit"].append(b_hit)
        results["bm25"]["rank"].append(b_rank)
        results["hybrid"]["hit"].append(h_hit)
        results["hybrid"]["rank"].append(h_rank)
        
        # 对某条 miss 的 QA，打印第一条检索结果，人工判断
        if not v_hit:
            print(f"  👉 VECTOR 第1名内容前80字: {v_docs[0][:80] if v_docs else '(空)'}")

    # ================= 循环结束，开始汇总 =================
    print("\n" + "="*50)
    print("📊 Day 5 检索方案对比结果")
    print("="*50)
    for method in ["vector", "bm25", "hybrid"]:
        recall_5 = sum(results[method]["hit"]) / len(results[method]["hit"])
        mrr = sum([1.0/r if r > 0 else 0 for r in results[method]["rank"]]) / len(results[method]["rank"])
        print(f"{method.upper():8} | Recall@5: {recall_5:.4f} | MRR: {mrr:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_day5_evaluation()