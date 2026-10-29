# src/conversation/ambiguity_detector.py
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 初始化 LLM 客户端 (企业级工程化：从环境变量读取，不硬编码)
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "your-api-key-here"),
    base_url="https://api.deepseek.com"
)

PRONOUNS = ["它", "他", "她", "这个", "那个", "前者", "后者", "该", "此"]

def find_pronoun(user_text: str) -> str | None:
    """识别句子里的指代词"""
    for p in PRONOUNS:
        if p in user_text:
            return p
    return None

def llm_judge_ambiguity(user_text: str, candidate_entities: list[str], pronoun: str) -> dict:
    """
    LLM 兜底判断 (工程化版本)
    包含：Prompt 设计、JSON 解析、异常重试与降级
    """
    prompt = f"""你是一个专业的对话系统歧义检测助手。
用户当前说："{user_text}"
当前上下文中的候选实体有：{candidate_entities}
用户话中包含指代词："{pronoun}"

请判断：这句话是否存在歧义（即无法唯一确定指代哪个实体）？
只返回严格的 JSON 格式，不要包含任何其他解释或 markdown 标记。
格式如下：
{{"is_ambiguous": true或false, "reason": "简短原因"}}
"""
    try:
        # 调用真实的 DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个严格输出 JSON 的歧义检测 API。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}, # 强制结构化输出
            temperature=0.0 # 保持判断稳定
        )
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        return result
    except Exception as e:
        # 企业级降级策略：如果 API 挂了，保守起见，如果是多实体则默认有歧义
        print(f"[Warning] LLM 调用失败，触发降级策略: {e}")
        if len(candidate_entities) >= 2:
            return {"is_ambiguous": True, "reason": "LLM异常降级：多候选默认按歧义处理"}
        return {"is_ambiguous": False, "reason": "LLM异常降级：单候选默认无歧义"}

def detect_ambiguity(user_text: str, candidate_entities: list[str]) -> dict:
    """主入口：歧义检测流水线"""
    pronoun = find_pronoun(user_text)
    
    # 规则1：处理“省略主语”（如果没有明确指代词，但句子很短）
    if pronoun is None:
        if len(user_text) <= 5 and len(candidate_entities) >= 2 and not any(c in user_text for c in candidate_entities):
            pronoun = "省略的主语"
        else:
            return {
                "need_clarification": False,
                "pronoun": None,
                "candidates": candidate_entities,
                "reason": "没有指代词"
            }

    # 规则2：候选实体不足两个，不构成歧义
    if len(candidate_entities) < 2:
        return {
            "need_clarification": False,
            "pronoun": pronoun,
            "candidates": candidate_entities,
            "reason": "候选实体不足两个"
        }

    # 规则拿不准或者疑似歧义时，交给 LLM 兜底做语义判断
    llm_result = llm_judge_ambiguity(user_text, candidate_entities, pronoun)
    
    return {
        "need_clarification": llm_result.get("is_ambiguous", True),
        "pronoun": pronoun,
        "candidates": candidate_entities,
        "reason": llm_result.get("reason", "LLM 判断结果")
    }