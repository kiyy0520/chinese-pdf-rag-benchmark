# test_week4_day4.py
from src.conversation.ambiguity_detector import detect_ambiguity
from src.conversation.clarification import build_clarification

TEST_CASES = [
    {"id": 1, "scene": "指代不明", "user_text": "把它关掉", "candidate_entities": ["手机", "电脑"], "expected_clarify": True},
    {"id": 2, "scene": "指代不明", "user_text": "这个多少钱", "candidate_entities": ["耳机", "键盘"], "expected_clarify": True},
    {"id": 3, "scene": "省略主语", "user_text": "已完成了", "candidate_entities": ["任务A", "任务B"], "expected_clarify": True},
    {"id": 4, "scene": "多实体场景", "user_text": "把前者发给我", "candidate_entities": ["报告A", "报告B", "报告C"], "expected_clarify": True},
    {"id": 5, "scene": "多实体场景（反例/无误报）", "user_text": "把手机号发我", "candidate_entities": ["手机", "电脑"], "expected_clarify": False},
    {"id": 6, "scene": "代词扩展", "user_text": "他刚才说了什么", "candidate_entities": ["张三", "李四"], "expected_clarify": True},
    {"id": 7, "scene": "单候选场景", "user_text": "把它关掉", "candidate_entities": ["手机"], "expected_clarify": False},
    {"id": 8, "scene": "显式指定场景", "user_text": "把电脑和手机都关掉", "candidate_entities": ["手机", "电脑"], "expected_clarify": False},
]

def run_experiment():
    tp = fp = fn = tn = 0
    print("🚀 开始运行真实 LLM 歧义检测实验...\n")

    for case in TEST_CASES:
        expected = case["expected_clarify"]
        result = detect_ambiguity(case["user_text"], case["candidate_entities"])
        actual = result["need_clarification"]
        
        clarification_text = ""
        # 如果检测到需要澄清，生成话术
        if actual:
            clarification_text = build_clarification(result["pronoun"], result["candidates"])
        
        # 统计指标
        if expected and actual:
            tp += 1
            status = "✅ 正确(TP)"
        elif not expected and actual:
            fp += 1
            status = "❌ 误报(FP)"
        elif expected and not actual:
            fn += 1
            status = "❌ 漏报(FN)"
        else:
            tn += 1
            status = "✅ 正确(TN)"
            
        print(f"ID: {case['id']} | 场景: {case['scene']}")
        print(f"输入: {case['user_text']} | 候选: {case['candidate_entities']}")
        print(f"期望澄清: {expected} | 实际澄清: {actual} -> {status}")
        print(f"判断理由: {result['reason']}")
        if clarification_text:
            print(f"💬 澄清话术: {clarification_text}")
        print("-" * 50)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\n📊 最终统计: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"🎯 召回率: {recall * 100:.2f}% (目标 >= 80%)")
    print(f"🎯 误报率: {fpr * 100:.2f}% (目标 < 15%)")

if __name__ == "__main__":
    run_experiment()