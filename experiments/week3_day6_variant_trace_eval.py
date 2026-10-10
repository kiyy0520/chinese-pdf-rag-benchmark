# experiments/week3_day6_variant_trace_eval.py
import os
import sys
import json
import types
import pandas as pd
from functools import partial
from dotenv import load_dotenv
from openai import OpenAI

# ==========================================
# 1. 兼容性补丁与路径修复
# ==========================================
_vertexai_module = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI:
    pass
_vertexai_module.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_module

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import chromadb
from src.api.rag_chain import rag_query
from src.routing.router import Router

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
    if contexts:
        context_str = "\n\n---\n\n".join(contexts)
        prompt = f"你是一个严谨的助手。请仅根据以下参考资料回答问题。\n如果参考资料中没有答案，请直接说'根据提供的资料无法回答'。\n\n参考资料：\n{context_str}\n\n问题：{question}\n答案："
    else:
        prompt = question
        
    res = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}], 
        temperature=0.1
    )
    return res.choices[0].message.content

def get_docs_by_ids(collection, doc_ids):
    if not doc_ids:
        return []
    res = collection.get(ids=doc_ids, include=["documents"])
    id_to_doc = dict(zip(res["ids"], res["documents"]))
    return [id_to_doc.get(doc_id, "") for doc_id in doc_ids]

def run_pure_llm(question):
    answer = generate_with_llm(question)
    return answer, []

def run_basic_rag(question):
    res = rag_query(question)
    return res["answer"], res["contexts"]

def run_adaptive_rag(question, router):
    """方案 3：自适应 RAG，新增 route_taken 返回"""
    route_res = router.route(question)
    route_taken = route_res["route"]  # ✅ 新增：记录路由轨迹
    
    if route_taken == "skip":
        return route_res["answer"], [], route_taken  # ✅ 返回三个值
        
    doc_ids = [doc_id for doc_id, score in route_res["results"]]
    contexts = get_docs_by_ids(collection, doc_ids)
    answer = generate_with_llm(question, contexts)
    return answer, contexts, route_taken  # ✅ 返回三个值

# ==========================================
# 4. RAGAS 评估流水线（增强版，支持轨迹记录）
# ==========================================
def evaluate_scheme(scheme_name, scheme_func, testset, track_route=False):
    """
    track_route: 是否记录路由轨迹（仅自适应 RAG 需要）
    """
    samples = []
    route_records = []  # ✅ 用于存储路由轨迹
    
    for i, item in enumerate(testset, 1):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"  [{scheme_name}] {i}/{len(testset)}: {q[:20]}...")
        
        if track_route:
            answer, contexts, route_taken = scheme_func(q)  # 接收三个值
            route_records.append({"user_input": q, "route_taken": route_taken})
        else:
            answer, contexts = scheme_func(q)
            route_records.append({"user_input": q, "route_taken": "N/A"})
            
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
    
    # ✅ 关键：把路由轨迹合并进 DataFrame
    route_df = pd.DataFrame(route_records)
    df = pd.merge(df, route_df, on="user_input", how="left")
    
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
    
    print("正在初始化自适应 RAG 路由器（加载BGE模型，请稍候）...")
    router_instance = Router(collection)
    print("路由器初始化完成！开始跑三种方案...\n")

    print("🚀 开始跑 纯 LLM...")
    df_llm = evaluate_scheme("纯 LLM", run_pure_llm, testset, track_route=False)
    
    print("\n🚀 开始跑 基础 RAG...")
    df_rag = evaluate_scheme("基础 RAG", run_basic_rag, testset, track_route=False)
    
    print("\n🚀 开始跑 自适应 RAG...")
    df_adaptive = evaluate_scheme("自适应 RAG", partial(run_adaptive_rag, router=router_instance), testset, track_route=True)
    
    # 合并保存所有 RAGAS 结果（包含 route_taken）
    all_df = pd.concat([df_llm, df_rag, df_adaptive], ignore_index=True)
    all_df.to_csv("experiments/week3_day6_ragas_results_with_trace.csv", index=False, encoding="utf-8-sig")
    print("\n✅ 增强版 RAGAS 结果已保存至 experiments/week3_day6_ragas_results_with_trace.csv")
    
    # ==========================================
    # ✅ 核心变式任务：路由轨迹与错误归因关联分析
    # ==========================================
    print("\n" + "="*50)
    print("📊 自适应 RAG 路由轨迹与错误归因关联分析")
    print("="*50)
    
    # 只拿自适应 RAG 的数据做分析
    adaptive_trace = df_adaptive.copy()
    
    # 按 route_taken 分组，计算各条路由的平均 RAGAS 分数
    trace_analysis = adaptive_trace.groupby("route_taken")[
        ["faithfulness", "llm_context_precision_with_reference", "context_recall"]
    ].mean().reset_index()
    
    # 统计各条路由的使用次数（可以看出哪条路由被触发最多）
    route_counts = adaptive_trace["route_taken"].value_counts().reset_index()
    route_counts.columns = ["route_taken", "count"]
    trace_analysis = pd.merge(trace_analysis, route_counts, on="route_taken", how="left")
    
    # 输出结果
    trace_analysis.to_csv("experiments/week3_day6_route_trace_analysis.csv", index=False, encoding="utf-8-sig")
    print(trace_analysis.to_string(index=False))
    
    # ==========================================
    # 生成分层归因表
    # ==========================================
    badcases = all_df[(all_df["faithfulness"] < 1.0) | (all_df["context_recall"] < 1.0)].copy()
    if not badcases.empty:
        def attribute(row):
            if row["scheme"] == "纯 LLM":
                return "Query理解"
            if row["context_recall"] < 0.6:
                return "检索"
            if row["llm_context_precision_with_reference"] < 0.6:
                return "重排"
            return "生成"
        
        badcases["failure_layer"] = badcases.apply(attribute, axis=1)
        badcases["reason"] = badcases.apply(
            lambda row: f"分数: Faith={row['faithfulness']:.2f}, Prec={row['llm_context_precision_with_reference']:.2f}, Recall={row['context_recall']:.2f}", axis=1
        )
        
        output_cols = ["scheme", "user_input", "route_taken", "failure_layer", "reason", "faithfulness", "llm_context_precision_with_reference", "context_recall"]
        badcases[output_cols].rename(columns={"user_input": "question"}).to_csv(
            "experiments/week3_day6_layer_attribution_with_trace.csv", index=False, encoding="utf-8-sig"
        )
        print(f"\n✅ 已生成带轨迹的归因明细表: experiments/week3_day6_layer_attribution_with_trace.csv")
    else:
        print("\n⚠️ 未发现 badcase。")

if __name__ == "__main__":
    main()