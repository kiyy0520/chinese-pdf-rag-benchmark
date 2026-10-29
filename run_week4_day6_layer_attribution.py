# 文件名: run_week4_day6_layer_attribution.py
# 运行位置: 项目根目录

import pandas as pd
import json
import os
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv # 新增：用于加载 .env 文件

# ================= 配置区 =================
# 加载 .env 文件中的环境变量
load_dotenv() 

INPUT_CSV_PATH = "experiments/week4_day6_badcases.csv" 
OUTPUT_DETAIL_CSV = "experiments/week4_day6_layer_attribution.csv"
OUTPUT_SUMMARY_CSV = "experiments/week4_day6_layer_attribution_summary.csv"

# 从环境变量中获取 API Key
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("❌ 未在 .env 文件中找到 DEEPSEEK_API_KEY，请检查配置！")

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=api_key, 
    base_url="https://api.deepseek.com/v1"
)

# ================= 提示词模板 =================
SYSTEM_PROMPT = """
你是一个极度严格的 RAG 系统质检员。你的任务是判定多轮对话中出现的 badcase 属于哪个层级。
请根据用户提供的【历史对话】、【当前问题】、【检索内容】和【最终回答】，判定错误层级，必须选择以下四者之一：
1. "query理解"：用户当前问题包含代词或省略，系统未结合历史对话正确改写问题，导致后续全错。
2. "检索"：Query 理解正确，但检索出的内容里完全不包含回答所需的知识。
3. "重排"：检索到了包含答案的文档，但由于排序太靠后，被模型忽略了。
4. "生成"：检索内容包含答案且位置合理，但模型自己胡编乱造或总结错误。

请严格以 JSON 格式输出，包含两个字段：
- "error_layer": 只能是 "query理解", "检索", "重排", "生成" 中的一个。
- "reason": 简明扼要说明为什么判在这一层。
"""

# ================= 核心逻辑 =================
def evaluate_single_case(row):
    """对单条数据进行归因打分"""
    user_content = f"""
    【历史对话】: {row.get('history', '无')}
    【当前问题】: {row.get('query', '')}
    【检索内容】: {row.get('retrieved_context', '无')}
    【最终回答】: {row.get('response', '')}
    """
    
    try:
        # 调用 DeepSeek 模型
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format={ "type": "json_object" },
            temperature=0.0
        )
        
        # 解析结果
        result = json.loads(response.choices[0].message.content)
        return result.get("error_layer", "未知"), result.get("reason", "")
        
    except Exception as e:
        print(f"\n解析出错: {e}")
        return "解析失败", str(e)

def main():
    print("🚀 开始进行多轮对话分层归因分析...")
    
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"❌ 找不到文件: {INPUT_CSV_PATH}。请确保你准备了 badcase 测试集，或修改代码中的路径。")
        return
        
    df = pd.read_csv(INPUT_CSV_PATH)
    
    layers = []
    reasons = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="质检进度"):
        layer, reason = evaluate_single_case(row)
        layers.append(layer)
        reasons.append(reason)
        
    df['error_layer'] = layers
    df['reason'] = reasons
    
    # 保存明细表
    df.to_csv(OUTPUT_DETAIL_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✅ 归因明细已保存至: {OUTPUT_DETAIL_CSV}")
    
    # 计算错误率
    valid_df = df[df['error_layer'].isin(["query理解", "检索", "重排", "生成"])]
    
    if valid_df.empty:
        print("⚠️ 没有任何成功归因的数据，无法生成汇总表。")
        return
        
    error_counts = valid_df['error_layer'].value_counts()
    error_rates = valid_df['error_layer'].value_counts(normalize=True) * 100
    
    summary_df = pd.DataFrame({
        '错误数量': error_counts,
        '错误率(%)': error_rates.round(2)
    })
    
    # 确保 4 层都有数据，没有错误的补 0
    for layer in ["query理解", "检索", "重排", "生成"]:
        if layer not in summary_df.index:
            summary_df.loc[layer] = [0, 0.0]
            
    summary_df = summary_df.reindex(["query理解", "检索", "重排", "生成"])
    summary_df.index.name = '错误层级'
    
    # 保存汇总表
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, encoding='utf-8-sig')
    print(f"✅ 分层归因汇总表已保存至: {OUTPUT_SUMMARY_CSV}")
    print("\n📊 最终错误率统计:")
    print(summary_df)

if __name__ == "__main__":
    main()