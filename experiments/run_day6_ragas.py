import sys
import types

# 修复 RAGAS 0.4.3 与 langchain-community 0.4.x 的兼容性问题
# RAGAS 无条件导入已废弃的 langchain_community.chat_models.vertexai
# 这里预先注册一个占位模块，让导入通过
_vertexai_module = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI:
    pass  # 占位类，我们不用 VertexAI，所以留空即可
_vertexai_module.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_module

# experiments/run_day6_ragas.py
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall
from ragas.llms import LangchainLLMWrapper

from langchain_openai import ChatOpenAI

from src.api.rag_chain import rag_query

load_dotenv()

# 1. 加载测试集
with open("experiments/day6_testset.json", "r", encoding="utf-8") as f:
    testset = json.load(f)

# 2. 逐个跑 RAG，收集 answer + contexts
samples = []
raw_results = []
for i, item in enumerate(testset, 1):
    q = item["question"]
    gt = item["ground_truth"]
    print(f"[{i}/{len(testset)}] 跑 RAG: {q}")
    result = rag_query(q)
    answer = result["answer"]
    contexts = result.get("contexts", [])

    raw_results.append({
        "question": q,
        "answer": answer,
        "ground_truth": gt,
        "contexts": " || ".join(contexts),
    })

    samples.append(SingleTurnSample(
        user_input=q,
        response=answer,
        retrieved_contexts=contexts,
        reference=gt,
    ))

# 保存原始回答，供后面人工检查用
pd.DataFrame(raw_results).to_csv(
    "experiments/day6_rag_outputs.csv", index=False, encoding="utf-8-sig"
)
print("✅ 原始回答已保存到 experiments/day6_rag_outputs.csv")

# 3. 配置 RAGAS 裁判 LLM（用 Qwen，通过 LangChain 包装）
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.0,
))

# 4. 构建 RAGAS 数据集
dataset = EvaluationDataset(samples=samples)

metrics = [
    Faithfulness(),
    LLMContextPrecisionWithReference(),
    LLMContextRecall(),
]

# 5. 开始评估
print("🚀 开始 RAGAS 评估（10 个问题，预计几分钟）...")
result = evaluate(dataset=dataset, metrics=metrics, llm=evaluator_llm)

# 6. 保存结果
df = result.to_pandas()
df.to_csv("experiments/day6_ragas_results.csv", index=False, encoding="utf-8-sig")
print("✅ RAGAS 结果已保存到 experiments/day6_ragas_results.csv")

# 7. 打印三个指标平均值
print("\n=== 三个指标平均分 ===")
for col in ["faithfulness", "llm_context_precision_with_reference", "llm_context_recall"]:
    if col in df.columns:
        print(f"{col}: {df[col].mean():.4f}")