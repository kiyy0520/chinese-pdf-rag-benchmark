# 文件名: run_week4_day6_ragas_eval.py
# 运行位置: 项目根目录

import pandas as pd
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

# 注意：读取 V2 版本的对比结果
INPUT_CSV = "experiments/week4_day6_comparison_v2.csv"
OUTPUT_CSV = "experiments/week4_day6_ragas_results_v2.csv"

llm = LangchainLLMWrapper(ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.0
))

embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5"))

def run_ragas_for_plan(df, plan_name, ans_col):
    print(f"🚀 正在评测方案：{plan_name} ...")
    
    # 核心修改点：把 ||| 拆分成真正的 List[str] 给 RAGAS
    # 格式必须是：[[文档1, 文档2, 文档3], [文档4, 文档5]] 这样的二维列表
    formatted_contexts = []
    for ctx in df['retrieved_context'].tolist():
        if pd.notna(ctx):
            formatted_contexts.append(str(ctx).split("|||"))
        else:
            formatted_contexts.append(["无"])
            
    data = {
        "question": df['query'].tolist(),
        "answer": df[ans_col].tolist(),
        "contexts": formatted_contexts, # 使用拆分好的二维列表
        "ground_truth": df['response'].tolist()
    }
    
    dataset = Dataset.from_dict(data)
    
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings
        )
        scores_df = result.to_pandas()
        return {
            "方案": plan_name,
            "Faithfulness (忠实度)": round(scores_df['faithfulness'].mean(), 4),
            "Context Precision (上下文精确度)": round(scores_df['context_precision'].mean(), 4),
            "Context Recall (上下文召回率)": round(scores_df['context_recall'].mean(), 4)
        }
    except Exception as e:
        print(f"❌ {plan_name} 评测失败，错误信息: {e}")
        return {"方案": plan_name, "Faithfulness (忠实度)": 0.0, "Context Precision (上下文精确度)": 0.0, "Context Recall (上下文召回率)": 0.0}

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 找不到 {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    results = []
    
    results.append(run_ragas_for_plan(df, "纯LLM", "pure_llm_ans"))
    results.append(run_ragas_for_plan(df, "基础RAG", "basic_rag_ans"))
    results.append(run_ragas_for_plan(df, "多轮RAG(无状态)", "stateless_rag_ans"))
    results.append(run_ragas_for_plan(df, "多轮RAG(有状态)", "stateful_rag_ans"))
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    print("\n✅ 评测完成！最终 RAGAS 对比结果 (V2)：")
    print(result_df.to_string(index=False))
    print(f"📁 结果已保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()