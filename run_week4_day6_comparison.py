# 文件名: run_week4_day6_comparison.py
# 运行位置: 项目根目录

import pandas as pd
import os
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from src.conversation.multi_turn_rag import MultiTurnRAGPipeline

load_dotenv()

# 注意：读取 V2 版本的数据
INPUT_CSV_PATH = "experiments/week4_day6_badcases_v2.csv" 
OUTPUT_CSV = "experiments/week4_day6_comparison_v2.csv"

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com"
)

def mock_retrieve_and_map(query, mock_context_text):
    """核心修改点：把 ||| 分隔的多段文本，转成文档字典列表"""
    if not mock_context_text or pd.isna(mock_context_text):
        return [], 0.0
    
    # 按 ||| 切分，并去除首尾空格
    text_list = [t.strip() for t in str(mock_context_text).split("|||") if t.strip()]
    
    # 组装成系统需要的 docs 格式
    docs = [{"text": text, "id": f"mock_{i}", "score": 0.9 - (i * 0.1)} for i, text in enumerate(text_list)]
    return docs, 0.9

def run_pure_llm(history, query):
    prompt = f"历史对话：{history}\n用户当前问题：{query}\n请直接根据你的知识回答。"
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"调用失败: {e}"

def run_basic_rag(query, mock_ctx):
    try:
        pipeline = MultiTurnRAGPipeline()
        docs, _ = mock_retrieve_and_map(query, mock_ctx)
        llm_out = pipeline._generate_answer(query, docs)
        return llm_out.get("answer", "根据现有资料无法回答。")
    except Exception as e:
        return f"调用失败: {e}"

def run_multiturn_rag_stateless(query, mock_ctx):
    try:
        pipeline = MultiTurnRAGPipeline()
        docs, _ = mock_retrieve_and_map(query, mock_ctx)
        llm_out = pipeline._generate_answer(query, docs)
        return llm_out.get("answer", "根据现有资料无法回答。")
    except Exception as e:
        return f"调用失败: {e}"

def run_multiturn_rag_stateful(history, query, mock_ctx):
    try:
        pipeline = MultiTurnRAGPipeline()
        if history and history != '无':
            pipeline.history = [{"user": history, "assistant": "（历史系统回答）"}]
            
        state = pipeline._update_state(query)
        rewritten_query = pipeline._rewrite_query(state)
        
        docs, _ = mock_retrieve_and_map(rewritten_query, mock_ctx)
        llm_out = pipeline._generate_answer(rewritten_query, docs)
        return llm_out.get("answer", "根据现有资料无法回答。")
    except Exception as e:
        return f"调用失败: {e}"

def main():
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"❌ 找不到 {INPUT_CSV_PATH}")
        return
        
    df = pd.read_csv(INPUT_CSV_PATH)
    
    ans_pure, ans_basic, ans_stateless, ans_stateful = [], [], [], []
    
    print("🚀 开始对比实验（V2 多文档版本）...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="方案PK中"):
        query = row.get('query', '')
        history = str(row.get('history', ''))
        mock_ctx = row.get('retrieved_context', '')
        
        ans_pure.append(run_pure_llm(history, query))
        ans_basic.append(run_basic_rag(query, mock_ctx))
        ans_stateless.append(run_multiturn_rag_stateless(query, mock_ctx))
        ans_stateful.append(run_multiturn_rag_stateful(history, query, mock_ctx))
        
    df['pure_llm_ans'] = ans_pure
    df['basic_rag_ans'] = ans_basic
    df['stateless_rag_ans'] = ans_stateless
    df['stateful_rag_ans'] = ans_stateful
    
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✅ 修复后的对比结果已保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()