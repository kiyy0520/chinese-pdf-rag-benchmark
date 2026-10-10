# src/generation/pipeline_demo.py

from src.generation.citation import attach_citation
from src.generation.conflict_detector import detect_conflict


def run_pipeline(answer, candidate_sources, other_source):
    """
    串联流程：
    1. 用 attach_citation 给回答选出来源 A
    2. 如果来源 A 是 None，提前返回，不往下走
    3. 用 detect_conflict 比较来源 A 和来源 B（other_source）
    4. 返回完整结果
    """
    # 第 1 步：引用溯源，选出来源 A
    citation_result = attach_citation(answer, candidate_sources)
    source_a = citation_result["source"]
    score = citation_result["score"]

    # 第 2 步：没有来源，提前返回，避免对 None 取 ["text"] 报错
    if source_a is None:
        return {
            "answer": answer,
            "citation": citation_result,
            "conflict": None,
            "message": "没有候选来源，无法进行冲突检测"
        }

    # 第 3 步：冲突检测，比较来源 A 和来源 B
    conflict = detect_conflict(source_a, other_source)

    # 第 4 步：有矛盾就展示矛盾，没有就提示未检测到
    if conflict:
        message = conflict
    else:
        message = "未检测到冲突"

    return {
        "answer": answer,
        "citation": citation_result,
        "conflict": conflict,
        "message": message
    }


if __name__ == "__main__":
    # 场景 1：数字冲突
    print("=== 场景 1：数字冲突 ===")
    answer = "公司年假是10天"
    candidate_sources = [
        {"id": "doc_01", "text": "正式员工每年享有10天年假"},
        {"id": "doc_02", "text": "年假申请需提前三天"},
    ]
    other_source = {"id": "doc_B", "text": "员工每年有5天年假"}
    result = run_pipeline(answer, candidate_sources, other_source)
    print("回答：", result["answer"])
    print("选中来源：", result["citation"]["source"])
    print("相似度分数：", result["citation"]["score"])
    print("冲突结果：", result["message"])
    print()

    # 场景 2：无冲突
    print("=== 场景 2：无冲突 ===")
    answer2 = "年假申请需提前三天"
    candidate_sources2 = [
        {"id": "doc_03", "text": "年假申请需提前三天提交"},
        {"id": "doc_04", "text": "年假天数按工龄计算"},
    ]
    other_source2 = {"id": "doc_C", "text": "员工请假需走审批流程"}
    result2 = run_pipeline(answer2, candidate_sources2, other_source2)
    print("回答：", result2["answer"])
    print("选中来源：", result2["citation"]["source"])
    print("相似度分数：", result2["citation"]["score"])
    print("冲突结果：", result2["message"])
    print()

    # 场景 3：空候选来源
    print("=== 场景 3：空候选来源 ===")
    result3 = run_pipeline(answer, [], other_source)
    print("回答：", result3["answer"])
    print("选中来源：", result3["citation"]["source"])
    print("冲突结果：", result3["message"])