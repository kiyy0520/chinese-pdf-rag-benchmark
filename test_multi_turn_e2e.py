import os
import sys
import json
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.conversation.multi_turn_rag import MultiTurnRAGPipeline

def run_e2e_test():
    print("初始化多轮 RAG 引擎...")
    try:
        pipeline = MultiTurnRAGPipeline()
    except Exception as e:
        print(f"❌ 引擎初始化失败: {e}")
        return
        
    queries = [
        "马尔可夫决策过程的奖励函数是什么？",
        "那它的状态转移概率呢？",
        "这个模型适合解决什么具体问题？"
    ]
    
    print("\n=== 开始 3 轮追问 E2E 测试 ===")
    for i, q in enumerate(queries):
        print(f"\n---------------- 第 {i+1} 轮 ----------------")
        print(f"👤 用户原始输入: {q}")
        
        result = pipeline.process_turn(q)
        
        print(f"🧠 状态对象: {json.dumps(result['current_state'], ensure_ascii=False)}")
        print(f"🔄 改写 Query: {result['rewritten_query']}")
        print(f"📊 检索得分: {result['retrieval_score']:.6f}")
        print(f"⏱️ 检索延迟: {result['retrieval_latency']} 秒")
        print(f"⏱️ 总延迟: {result['total_latency']} 秒")
        
        refuse_status = "🚫 拒答" if result['is_refused'] else "✅ 正常回答"
        print(f"🛡️ 拒答状态: {refuse_status} (原因: {result['refuse_reason']})")
        print(f"🤖 最终回答: {result['answer']}")
        
        if result['retrieval_latency'] < 1.0:
            print("⚡ 延迟达标 (<1s/轮)")
        else:
            print("⚠️ 警告：延迟超过 1 秒！")

if __name__ == "__main__":
    load_dotenv()
    run_e2e_test()