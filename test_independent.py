# test_independent.py
from src.conversation.state import ConversationState
from src.conversation.topic_switch import detect_topic_switch

def run_independent_test():
    # 1. 初始化状态对象
    state = ConversationState()
    print("=== 初始化状态 ===")
    print(f"当前实体: {state.current_entities}, 切换标记: {state.topic_switch_flag}\n")

    # 2. 准备 3 轮对话数据
    turns = [
        # 第 1 轮
        {"new_entities": ["机械键盘", "办公"], "new_confirmed": {"用途": "办公"}, "new_questions": []},
        # 第 2 轮
        {"new_entities": ["鼠标"], "new_confirmed": {}, "new_questions": []},
        # 第 3 轮
        {"new_entities": ["键盘", "静音"], "new_confirmed": {}, "new_questions": []}
    ]

    # 3. 循环处理每一轮
    for idx, turn in enumerate(turns, start=1):
        print(f"--- 第 {idx} 轮 ---")
        new_entities = turn["new_entities"]
        new_confirmed = turn["new_confirmed"]
        new_questions = turn["new_questions"]

        # 关键步骤 A：检测话题切换（注意传入参数的顺序）
        # 传入：新实体列表, 状态里记录的历史实体列表
        topic_switch = detect_topic_switch(new_entities, state.current_entities)
        
        # 关键步骤 B：将本轮信息应用到状态对象中
        state.apply_turn_update(
            new_entities=new_entities,
            new_confirmed=new_confirmed,
            new_questions=new_questions,
            topic_switch=topic_switch
        )

        # 打印状态
        print(f"本轮新实体: {new_entities}")
        print(f"话题切换标记: {state.topic_switch_flag}")
        print(f"状态中当前实体: {state.current_entities}")
        print(f"状态中已确认信息: {state.confirmed_info}\n")

if __name__ == "__main__":
    run_independent_test()