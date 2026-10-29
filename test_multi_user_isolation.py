import os
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory

# 引入解耦后的两个核心类
from src.conversation.state_tracker import StateTracker
from src.conversation.memory import MemoryManager

load_dotenv()
llm = ChatDeepSeek(model="deepseek-flash", temperature=0)

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

# ================= 变式练习：多用户隔离 =================
def run_multi_user_test():
    # 1. 升级数据结构：用字典管理不同用户的完整历史
    history_store = {}
    def get_full_history(sid):
        if sid not in history_store:
            history_store[sid] = ChatMessageHistory()
        return history_store[sid]

    # 实例化解耦后的组件
    tracker = StateTracker(llm=llm)
    memory = MemoryManager(llm=llm, token_counter=token_counter_func, state_tracker=tracker)
    
    # 清理旧文件
    for sid in ["user_A", "user_B"]:
        state_file = Path(f"./data/states/{sid}.json")
        if state_file.exists(): state_file.unlink()

    # 2. 设计交错对话集
    test_sequence = [
        ("user_A", "我想订北京的酒店。"),
        ("user_B", "帮我订去上海的机票。"),
        ("user_A", "预算500元以内。"),
        ("user_B", "要头等舱。"),
        ("user_A", "我刚才说要订哪里的酒店？") # 终极测试
    ]

    print("🚀 开始多用户并发隔离测试...\n")

    for i, (sid, user_msg) in enumerate(test_sequence):
        print(f"--- 轮次 {i+1} | 当前用户: {sid} ---")
        
        full_history = get_full_history(sid)
        
        # 获取注入状态后的历史（由 MemoryManager 负责）
        trimmed_with_state = memory.get_trimmed_history(sid, full_history, max_tokens=300)
        messages_with_state = trimmed_with_state.messages + [HumanMessage(content=user_msg)]

        # 调用 LLM
        response = llm.invoke(messages_with_state)
        print(f"{sid} 说: {user_msg}")
        print(f"AI 回复: {response.content[:60]}...\n")
        
        # 更新全量历史
        full_history.add_message(HumanMessage(content=user_msg))
        full_history.add_message(response)
        
        # 更新并保存状态
        state = tracker.load_state(sid)
        state = tracker.update_state_with_llm(state, user_msg, response.content)
        tracker.save_state(state, sid)
        
    # ================= 验证阶段 =================
    print("="*40)
    print("📊 状态文件独立验证")
    print("="*40)
    
    path_A = Path("./data/states/user_A.json")
    path_B = Path("./data/states/user_B.json")
    
    if path_A.exists() and path_B.exists():
        print("✅ 成功生成独立的 JSON 状态文件：")
        print(f"- A 的状态路径: {path_A}")
        print(f"- B 的状态路径: {path_B}")
        print("\n请打开这两个文件，确认 A 记的是北京，B 记的是上海。")
    else:
        print("❌ 状态文件生成失败！")

if __name__ == "__main__":
    run_multi_user_test()