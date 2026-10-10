import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import csv
import chromadb
from sentence_transformers import SentenceTransformer

from src.retrieval.vector_retriever import VectorRetriever
from src.generation.refusal import (
    compute_retrieval_confidence,
    compute_generation_confidence,
    should_refuse,
    REFUSAL_TEMPLATE
)

def run_test():
    print("加载模型和检索器...")
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection(name="company_policy_kb")
    config_name = "company_policy"
    retriever = VectorRetriever(collection, config_name=config_name)
    
    # 用于算生成置信度的 embedding 模型，复用检索器的模型即可
    embed_model = retriever.model
    
    # 读取测试题
    test_cases = []
    with open("experiments/week3_day5_test_cases.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_cases.append(row)
            
    print(f"共加载 {len(test_cases)} 条测试题。\n")
    
    results = []
    
    # 遍历测试题
    for case in test_cases:
        case_id = case["case_id"]
        question = case["question"]
        expected = case["expected"]
        q_type = case["type"]
        
        # 第一步：检索
        retrieved = retriever.retrieve(question, top_k=5)
        scores = [score for _, score in retrieved]
        doc_ids = [doc_id for doc_id, _ in retrieved]
        
        # 获取检索到的文本
        retrieved_texts = []
        if doc_ids:
            res = collection.get(ids=doc_ids, include=["documents"])
            id_to_doc = dict(zip(res["ids"], res["documents"]))
            retrieved_texts = [id_to_doc.get(doc_id, "") for doc_id in doc_ids]
        
        # 第二步：算检索置信度
        retrieval_conf, mean_sim, var_sim = compute_retrieval_confidence(scores, alpha=1.0)
        
        # 第三步：模拟 LLM 生成回答
        # 为了先跑通逻辑，这里用最简单的“拼接检索内容前50字”来模拟回答。
        # 后续你可以替换成真正的 LLM 调用。
        if retrieved_texts:
            mock_answer = retrieved_texts[0]
        else:
            mock_answer = ""
            
        # 第四步：算生成置信度
        gen_conf = compute_generation_confidence(mock_answer, retrieved_texts, embed_model)
        
        # 第五步：双重拒答判断
        is_refused = should_refuse(retrieval_conf, gen_conf, 
                                   retrieval_threshold=0.45, 
                                   generation_threshold=0.6)
        
        # 第六步：根据判断给出最终输出
        if is_refused:
            final_output = REFUSAL_TEMPLATE
        else:
            final_output = mock_answer
            
        # 判断本次预测是否正确
        predicted = "refuse" if is_refused else "answer"
        is_correct = (predicted == expected)
        
        # 记录结果
        results.append({
            "case_id": case_id,
            "question": question,
            "type": q_type,
            "expected": expected,
            "predicted": predicted,
            "is_correct": is_correct,
            "retrieval_conf": round(retrieval_conf, 4),
            "mean_sim": round(mean_sim, 4),
            "var_sim": round(var_sim, 4),
            "gen_conf": round(gen_conf, 4),
            "final_output": final_output
        })
        
        print(f"[{case_id}] {question}")
        print(f"  期望: {expected} | 预测: {predicted} | 正确: {'✅' if is_correct else '❌'}")
        print(f"  检索置信度: {retrieval_conf:.4f} (均值{mean_sim:.4f} 方差{var_sim:.4f})")
        print(f"  生成置信度: {gen_conf:.4f}")
        print(f"  最终输出: {final_output[:30]}...\n")
        
    # 第七步：统计指标
    total = len(results)
    out_of_scope = [r for r in results if r["type"] == "out_of_scope"]
    normal = [r for r in results if r["type"] == "normal"]
    
    refusal_accuracy = sum(1 for r in out_of_scope if r["predicted"] == "refuse") / len(out_of_scope) if out_of_scope else 0
    false_refusal_rate = sum(1 for r in normal if r["predicted"] == "refuse") / len(normal) if normal else 0
    
    print("="*50)
    print("📊 拒答测试汇总")
    print("="*50)
    print(f"拒答准确率 (超范围题正确拒答): {refusal_accuracy:.2%} (目标 ≥ 90%)")
    print(f"误拒率 (正常题被错误拒答): {false_refusal_rate:.2%} (目标 < 10%)")
    print("="*50)
    
    # 第八步：写入 CSV
    os.makedirs("experiments", exist_ok=True)
    output_path = "experiments/week3_day5_refusal_eval.csv"
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"结果已保存到 {output_path}")

if __name__ == "__main__":
    run_test()