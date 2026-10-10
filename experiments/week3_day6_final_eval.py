# experiments/week3_day6_final_eval.py
import os
import sys
import json
import types
import pandas as pd
from functools import partial
from dotenv import load_dotenv
from openai import OpenAI

# ==========================================
# 1. 兼容性补丁与路径修复（必须放在所有导入之前）
# ==========================================
# 修复 RAGAS 0.4.3 与 langchain-community 0.4.x 的兼容性问题
_vertexai_module = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI:
    pass
_vertexai_module.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_module

# 把项目根目录加入 Python 搜索路径，保证 src 能够被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import chromadb
from src.api.rag_chain import rag_query
from src.routing.router import Router

# RAGAS 相关导入
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

load_dotenv()

# ==========================================
# 2. 初始化 LLM 与 向量库
# ==========================================
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection(name="day4_chunking_eval")

# ==========================================
# 3. 统一生成函数与三种方案
# ==========================================
def generate_with_llm(question, contexts=None):
    """统一的大模型生成函数，纯 LLM 和 自适应 RAG 生成时都用这个"""
    if contexts:
        context_str = "\n\n---\n\n".join(contexts)
        prompt = f"你是一个严谨的助手。请仅根据以下参考资料回答问题。\n如果参考资料中没有答案，请直接说'根据提供的资料无法回答'。\n\n参考资料：\n{context_str}\n\n问题：{question}\n答案："
    else:
        prompt = question  # 纯 LLM，不给任何资料
        
    res = client.chat.completions.create(
        model="deepseek-chat",  # ✅ 这里必须换成 deepseek-chat
        messages=[{"role": "user", "content": prompt}], 
        temperature=0.1
    )
    return res.choices[0].message.content

def get_docs_by_ids(collection, doc_ids):
    """辅助函数：根据 ID 从 Chroma 捞取文本（复用你 Day5 代码里的逻辑）"""
    if not doc_ids:
        return []
    res = collection.get(ids=doc_ids, include=["documents"])
    id_to_doc = dict(zip(res["ids"], res["documents"]))
    return [id_to_doc.get(doc_id, "") for doc_id in doc_ids]

def run_pure_llm(question):
    """方案 1：纯 LLM（不给上下文，直接问）"""
    answer = generate_with_llm(question)
    return answer, []

def run_basic_rag(question):
    """方案 2：基础 RAG（直接调用你的老 rag_query）"""
    res = rag_query(question)
    return res["answer"], res["contexts"]

def run_adaptive_rag(question, router):
    """方案 3：自适应 RAG（用你的老路由，接收外部传入的 router 实例）"""
    route_res = router.route(question)  # ✅ 直接使用传入的 router，不再重新加载模型
    if route_res["route"] == "skip":
        return route_res["answer"], []  # 走闲聊，直接返回，不检索
        
    doc_ids = [doc_id for doc_id, score in route_res["results"]]
    contexts = get_docs_by_ids(collection, doc_ids)
    
    answer = generate_with_llm(question, contexts)
    return answer, contexts

# ==========================================
# 4. RAGAS 评估流水线
# ==========================================
def evaluate_scheme(scheme_name, scheme_func, testset):
    """评估单种方案，返回 RAGAS 结果 DataFrame"""
    samples = []
    for i, item in enumerate(testset, 1):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"  [{scheme_name}] {i}/{len(testset)}: {q[:20]}...")
        
        answer, contexts = scheme_func(q)
        samples.append(SingleTurnSample(
            user_input=q,
            response=answer,
            retrieved_contexts=contexts,
            reference=gt
        ))
    
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0.0
    ))
    
    dataset = EvaluationDataset(samples=samples)
    metrics = [Faithfulness(), LLMContextPrecisionWithReference(), LLMContextRecall()]
    result = evaluate(dataset=dataset, metrics=metrics, llm=evaluator_llm)
    
    df = result.to_pandas()
    df["scheme"] = scheme_name
    return df

# ==========================================
# 5. 主流程
# ==========================================
def main():
    testset_path = "experiments/week2/day6_testset.json"
    if not os.path.exists(testset_path):
        print(f"❌ 找不到测试集: {testset_path}，请确认路径是否正确！")
        return

    with open(testset_path, "r", encoding="utf-8") as f:
        testset = json.load(f)
    
    # ✅ 关键：在这里全局初始化一次 Router，只加载一次 BGE 模型
    print("正在初始化自适应 RAG 路由器（加载BGE模型，请稍候）...")
    router_instance = Router(collection)
    print("路由器初始化完成！开始跑三种方案...")

    print("\n🚀 开始跑 纯 LLM...")
    df_llm = evaluate_scheme("纯 LLM", run_pure_llm, testset)
    
    print("\n🚀 开始跑 基础 RAG...")
    df_rag = evaluate_scheme("基础 RAG", run_basic_rag, testset)
    
    print("\n🚀 开始跑 自适应 RAG...")
    # ✅ 使用 partial 把 router_instance 绑定进去，保证接口依然是 scheme_func(q)
    df_adaptive = evaluate_scheme("自适应 RAG", partial(run_adaptive_rag, router=router_instance), testset)
    
    # 合并保存所有 RAGAS 结果
    all_df = pd.concat([df_llm, df_rag, df_adaptive], ignore_index=True)
    all_df.to_csv("experiments/week3_day6_ragas_results.csv", index=False, encoding="utf-8-sig")
    print("\n✅ RAGAS 结果已保存至 experiments/week3_day6_ragas_results.csv")
    
    # 生成对比表（取平均值）
    comparison = all_df.groupby("scheme")[["faithfulness", "llm_context_precision_with_reference", "context_recall"]].mean().reset_index()
    comparison.to_csv("experiments/week3_day6_comparison.csv", index=False, encoding="utf-8-sig")
    
    # 生成 Markdown 报告草稿
    with open("experiments/week3_day6_comparison.md", "w", encoding="utf-8") as f:
        f.write("# Day 6 三种方案 RAGAS 对比报告\n\n")
        f.write("## 对比表\n\n")
        f.write(comparison.to_markdown(index=False))
        f.write("\n\n> 注：自适应 RAG 需至少在 1 个指标上优于基础 RAG。\n")
    print("✅ 对比报告已保存至 experiments/week3_day6_comparison.md")
    
    # 生成分层归因表（基于基础 RAG 的 badcase 捞取）
    badcases = df_rag[(df_rag["faithfulness"] < 0.8) | (df_rag["context_recall"] < 1.0)].copy()
    if not badcases.empty:
        def attribute(row):
            if row["context_recall"] < 1.0: 
                return "检索"
            if row["llm_context_precision_with_reference"] < 0.5: 
                return "重排"
            return "生成"
        
        badcases["failure_layer"] = badcases.apply(attribute, axis=1)
        badcases["reason"] = "RAGAS 指标低于阈值自动判定"
        
        output_cols = ["user_input", "failure_layer", "reason", "faithfulness", "llm_context_precision_with_reference", "context_recall"]
        badcases[output_cols].rename(columns={"user_input": "question"}).to_csv(
            "experiments/week3_day6_layer_attribution.csv", index=False, encoding="utf-8-sig"
        )
        print(f"✅ 分层归因表已保存，共捞出 {len(badcases)} 条 badcase")
    else:
        print("⚠️ 基础 RAG 未发现 badcase，请检查阈值或指标计算。")

if __name__ == "__main__":
    main()