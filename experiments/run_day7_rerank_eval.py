# experiments/run_day7_rerank_eval.py
import sys
import types
import os

_vertexai_module = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI:
    pass
_vertexai_module.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_module

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 只在模型缓存已存在时启用离线模式
# 如果首次运行，注释掉下面两行
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import pandas as pd
from dotenv import load_dotenv

from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

from src.api.rag_chain import rag_query, rag_query_with_rerank

load_dotenv()

with open("experiments/day6_testset.json", "r", encoding="utf-8") as f:
    testset = json.load(f)

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.0,
))

def build_dataset(query_fn):
    samples = []
    for item in testset:
        q = item["question"]
        gt = item["ground_truth"]
        r = query_fn(q)
        samples.append(SingleTurnSample(
            user_input=q,
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=gt,
        ))
    return EvaluationDataset(samples=samples)

metrics = [Faithfulness(), LLMContextPrecisionWithReference(), LLMContextRecall()]

print("=== 方案 A：纯向量检索 top-3 ===")
ds_a = build_dataset(rag_query)
res_a = evaluate(dataset=ds_a, metrics=metrics, llm=evaluator_llm)
df_a = res_a.to_pandas()
df_a.to_csv("experiments/day7_rerank_vector_only.csv", index=False, encoding="utf-8-sig")
print(df_a[["faithfulness", "llm_context_precision_with_reference", "context_recall"]].mean())

print("\n=== 方案 B：向量召回 top-10 + Rerank top-3 ===")
ds_b = build_dataset(rag_query_with_rerank)
res_b = evaluate(dataset=ds_b, metrics=metrics, llm=evaluator_llm)
df_b = res_b.to_pandas()
df_b.to_csv("experiments/day7_rerank_vector_rerank.csv", index=False, encoding="utf-8-sig")
print(df_b[["faithfulness", "llm_context_precision_with_reference", "context_recall"]].mean())

print("\n=== 对比 ===")
compare = pd.DataFrame({
    "metric": ["Faithfulness", "ContextPrecision", "ContextRecall"],
    "vector_only": [
        df_a["faithfulness"].mean(),
        df_a["llm_context_precision_with_reference"].mean(),
        df_a["context_recall"].mean(),
    ],
    "vector_plus_rerank": [
        df_b["faithfulness"].mean(),
        df_b["llm_context_precision_with_reference"].mean(),
        df_b["context_recall"].mean(),
    ],
})
print(compare)
compare.to_csv("experiments/day7_rerank_comparison.csv", index=False, encoding="utf-8-sig")