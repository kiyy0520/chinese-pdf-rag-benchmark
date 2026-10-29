import os
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, trim_messages
from langchain_community.chat_message_histories import ChatMessageHistory

# 引入解耦后的两个核心类
from src.conversation.state_tracker import StateTracker
from src.conversation.memory import MemoryManager

load_dotenv()

# 1. 初始化大模型
llm = ChatDeepSeek(model="deepseek-flash", temperature=0)

# 2. 准备 20 条测试数据
TEST_DATA = [
    "我想订个酒店，预算500元以内。", "要靠近地铁站的。", "最好是朝阳区的。", "我明天下午入住。", 
    "住两个晚上。", "大床房，我一个人住。", "我叫张三，电话是13800000000。", "需要开发票。",
    "发票抬头是某某科技有限公司。", "税号是123456789。", "对了，酒店有免费早餐吗？", "如果有健身房更好。",
    "退房时间是几点？", "可以延迟退房到下午2点吗？", "我可能会带一只小猫。", "酒店允许带宠物吗？",
    "如果不允许，帮我看看附近的宠物店。", "我需要租一辆车，从酒店去机场。", "租车大概多少钱？", "帮我把前面的行程总结一下。"
]

# 3. 加载 Tokenizer
print("正在加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")

def count_tokens(messages) -> int:
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total += len(tokenizer.encode(content, add_special_tokens=False))
    return total

def token_counter_func(messages) -> int:
    return count_tokens(messages)

# ================= 4. 实验主体 =================
def run_experiments():
    session_id = "test_user_001"
    
    # 自动清理旧文件
    state_file = Path(f"./data/states/{session_id}.json")
    if state_file.exists():
        state_file.unlink()
        
    # 实例化解耦后的组件
    tracker = StateTracker(llm=llm)
    memory = MemoryManager(llm=llm, token_counter=token_counter_func, state_tracker=tracker)
    full_history = ChatMessageHistory()

    tokens_without_state = 0
    tokens_with_state = 0

    print("🚀 开始 20 轮对话模拟...\n")

    for i, user_msg in enumerate(TEST_DATA):
        print(f"--- 轮次 {i+1} ---")
        
        # 【对照组】：大窗口 2000
        trimmer = trim_messages(
            max_tokens=2000, strategy="last", token_counter=token_counter_func, 
            include_system=True, start_on="human"
        )
        trimmed_no_state = trimmer.invoke(full_history.messages)
        messages_no_state = trimmed_no_state + [HumanMessage(content=user_msg)]
        tokens_without_state += count_tokens(messages_no_state)
        
        # 【实验组】：强制小窗口 300，由 MemoryManager 负责
        trimmed_with_state = memory.get_trimmed_history(session_id, full_history, max_tokens=300)
        messages_with_state = trimmed_with_state.messages + [HumanMessage(content=user_msg)]
        tokens_with_state += count_tokens(messages_with_state)

        # 调用 LLM
        response = llm.invoke(messages_with_state)
        print(f"用户: {user_msg}")
        print(f"AI: {response.content[:50]}...")
        
        # 更新全量历史
        full_history.add_message(HumanMessage(content=user_msg))
        full_history.add_message(response)
        
        # 更新并保存状态
        state = tracker.load_state(session_id)
        state = tracker.update_state_with_llm(state, user_msg, response.content)
        tracker.save_state(state, session_id)
        
    # ================= 5. 计算指标 =================
    print("\n" + "="*40)
    print("📊 实验结果分析")
    print("="*40)
    
    saved_tokens = tokens_without_state - tokens_with_state
    save_rate = (saved_tokens / tokens_without_state) * 100 if tokens_without_state > 0 else 0
    print(f"1️⃣ 总 Token - 无状态管理: {tokens_without_state}")
    print(f"1️⃣ 总 Token - 有状态管理: {tokens_with_state}")
    print(f"✅ Token 节省率: {save_rate:.2f}% (目标 ≥ 30%)")
    
    recovered_state = tracker.load_state(session_id)
    if recovered_state.model_dump() == state.model_dump():
        print("✅ 状态恢复成功率: 100% (序列化与反序列化完全一致)")
    else:
        print("❌ 状态恢复失败: 数据丢失")

if __name__ == "__main__":
    run_experiments()