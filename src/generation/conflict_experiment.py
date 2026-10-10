# src/generation/conflict_experiment.py

from src.generation.conflict_detector import detect_conflict


def run_experiment():
    # 5 条矛盾用例，should_conflict=True
    conflict_cases = [
        {
            "id": "C1",
            "source_a": {"id": "A1", "text": "公司年假是10天"},
            "source_b": {"id": "B1", "text": "公司年假是5天"},
            "should_conflict": True,
        },
        {
            "id": "C2",
            "source_a": {"id": "A2", "text": "项目截止6月1日"},
            "source_b": {"id": "B2", "text": "项目截止7月1日"},
            "should_conflict": True,
        },
        {
            "id": "C3",
            "source_a": {"id": "A3", "text": "负责人是张三"},
            "source_b": {"id": "B3", "text": "负责人是李四"},
            "should_conflict": True,
        },
        {
            "id": "C4",
            "source_a": {"id": "A4", "text": "本产品支持安卓"},
            "source_b": {"id": "B4", "text": "本产品不支持安卓"},
            "should_conflict": True,
        },
        {
            "id": "C5",
            "source_a": {"id": "A5", "text": "报销上限1000元"},
            "source_b": {"id": "B5", "text": "报销上限2000元"},
            "should_conflict": True,
        },
    ]

    # 5 条正常用例，should_conflict=False
    normal_cases = [
        {
            "id": "N1",
            "source_a": {"id": "A6", "text": "年假申请需提前三天"},
            "source_b": {"id": "B6", "text": "年假天数按工龄计算"},
            "should_conflict": False,
        },
        {
            "id": "N2",
            "source_a": {"id": "A7", "text": "项目启动会已召开"},
            "source_b": {"id": "B7", "text": "项目进入开发阶段"},
            "should_conflict": False,
        },
        {
            "id": "N3",
            "source_a": {"id": "A8", "text": "负责人是张三"},
            "source_b": {"id": "B8", "text": "张三负责本项目"},
            "should_conflict": False,
        },
        {
            "id": "N4",
            "source_a": {"id": "A9", "text": "本产品支持安卓"},
            "source_b": {"id": "B9", "text": "本产品也支持苹果"},
            "should_conflict": False,
        },
        {
            "id": "N5",
            "source_a": {"id": "A10", "text": "报销需提交发票"},
            "source_b": {"id": "B10", "text": "报销流程共三步"},
            "should_conflict": False,
        },
    ]

    # 跑矛盾用例
    true_positive = 0
    false_negative = 0
    conflict_results = []

    for case in conflict_cases:
        result = detect_conflict(case["source_a"], case["source_b"])
        if result:
            true_positive += 1
            conflict_results.append((case["id"], "正确标记", result))
        else:
            false_negative += 1
            conflict_results.append((case["id"], "漏报", None))

    # 跑正常用例
    false_positive = 0
    true_negative = 0
    normal_results = []

    for case in normal_cases:
        result = detect_conflict(case["source_a"], case["source_b"])
        if result:
            false_positive += 1
            normal_results.append((case["id"], "误报", result))
        else:
            true_negative += 1
            normal_results.append((case["id"], "正确不标记", None))

    # 算指标
    total_conflict = len(conflict_cases)
    total_normal = len(normal_cases)

    recall = true_positive / total_conflict if total_conflict else 0
    false_positive_rate = false_positive / total_normal if total_normal else 0

    # 打印结果
    print("=== 矛盾用例结果 ===")
    for row in conflict_results:
        print(row)

    print()
    print("=== 正常用例结果 ===")
    for row in normal_results:
        print(row)

    print()
    print(f"召回率：{true_positive}/{total_conflict} = {recall:.2%}")
    print(f"误报率：{false_positive}/{total_normal} = {false_positive_rate:.2%}")

    return {
        "conflict_results": conflict_results,
        "normal_results": normal_results,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
    }


if __name__ == "__main__":
    run_experiment()