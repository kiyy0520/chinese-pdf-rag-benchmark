import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 1. 加载 .env
load_dotenv()

# 2. 初始化（使用最便宜的 deepseek-flash）
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com"
)

def llm_judge_fallback(current_query: str, rewritten_query: str) -> dict:
    system_prompt = """
    你是一个严谨的查询改写评估专家。你的任务是判断【改写后的问句】是否完美等价于【当前问句】在当前对话下的真实意图。
    
    判断标准：
    1. 如果改写后包含了当前问句的所有核心实体和限定词，且没有引入错误实体，则为 true。
    2. 如果改写后丢失了关键限定词（如“黑色”）、引入了无关实体，或者话题切错了，则为 false。
    
    你必须严格输出以下 JSON 格式，不要包含任何其他废话：
    {"is_equivalent": true/false, "reason": "用一句话说明理由"}
    """
    user_prompt = f"【当前问句】：{current_query}\n【改写后问句】：{rewritten_query}"

    try:
        response = client.chat.completions.create(
            model="deepseek-flash",  # 已修正为最便宜的 Flash 模型
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        result_text = response.choices[0].message.content
        return json.loads(result_text)
    except Exception as e:
        return {"is_equivalent": False, "reason": f"API报错，强制降级: {str(e)}"}

# ==========================================
# 3. 测试用例
# ==========================================
test_cases = [
    {"id": "case_01 (过度改写)", "current": "它多少钱？", "rewritten": "AirPro 降噪耳机和 Lite 降噪耳机多少钱？"},
    {"id": "case_02 (完美改写，之前被规则误杀)", "current": "这个有优惠吗？", "rewritten": "AirPro 有优惠吗？"},
    {"id": "case_03 (改写不足)", "current": "那黑色款多少钱？", "rewritten": "Lite 降噪耳机多少钱？"},
    {"id": "case_05 (话题误判)", "current": "你们支持七天无理由退货吗？", "rewritten": "AirPro 降噪耳机支持七天无理由退货吗？"}
]

if __name__ == "__main__":
    print("开始真实 LLM Judge 评估...\n")
    for tc in test_cases:
        print(f"--- 测试 {tc['id']} ---")
        print(f"当前问句: {tc['current']}")
        print(f"改写结果: {tc['rewritten']}")
        
        judge_result = llm_judge_fallback(tc['current'], tc['rewritten'])
        
        final_query = tc['rewritten'] if judge_result['is_equivalent'] else tc['current']
        fallback_triggered = not judge_result['is_equivalent']
        
        print(f"Judge 判定: {judge_result['is_equivalent']} (是否保留改写)")
        print(f"Judge 理由: {judge_result['reason']}")
        print(f"最终输出 final_query: {final_query}")
        print(f"是否触发降级: {fallback_triggered}\n")