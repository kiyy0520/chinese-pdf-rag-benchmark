# src/generation/conflict_detector.py

import re


def check_number_conflict(source_a, source_b):
    """检查两个来源是否存在数字冲突。"""
    nums_a = re.findall(r"\d+", source_a["text"])
    nums_b = re.findall(r"\d+", source_b["text"])

    if nums_a and nums_b and nums_a != nums_b:
        return (
            f"来源 {source_a['id']} 与来源 {source_b['id']} 存在矛盾："
            f"数字不同（{nums_a} vs {nums_b}）"
        )
    return None


def check_date_conflict(source_a, source_b):
    """检查两个来源是否存在日期冲突。"""
    # 匹配 6月1日、2024-06-01、2024/06/01 等简单格式
    pattern = r"\d{1,4}[-/年]\d{1,2}[-/月]\d{1,2}日?|\d{1,2}月\d{1,2}日"
    dates_a = re.findall(pattern, source_a["text"])
    dates_b = re.findall(pattern, source_b["text"])

    if dates_a and dates_b and dates_a != dates_b:
        return (
            f"来源 {source_a['id']} 与来源 {source_b['id']} 存在矛盾："
            f"日期不同（{dates_a} vs {dates_b}）"
        )
    return None


def check_name_conflict(source_a, source_b):
    """检查两个来源是否存在人名冲突（简化版）。"""
    # 匹配“是XXX”或“由XXX负责”里的 2-4 个字
    pattern = r"(?:是|由)([\u4e00-\u9fa5]{2,4})(?:负责)?"

    names_a = re.findall(pattern, source_a["text"])
    names_b = re.findall(pattern, source_b["text"])

    if names_a and names_b and names_a != names_b:
        return (
            f"来源 {source_a['id']} 与来源 {source_b['id']} 存在矛盾："
            f"人名不同（{names_a} vs {names_b}）"
        )
    return None


def check_status_conflict(source_a, source_b):
    """检查两个来源是否存在状态冲突（支持/不支持）。"""
    text_a = source_a["text"]
    text_b = source_b["text"]

    # 支持 vs 不支持/只支持
    if ("支持" in text_a and "不支持" in text_b) or (
        "不支持" in text_a and "支持" in text_b
    ):
        return (
            f"来源 {source_a['id']} 与来源 {source_b['id']} 存在矛盾："
            f"支持状态冲突"
        )
    return None


def detect_conflict(source_a, source_b):
    """按顺序检查多条规则，返回第一条触发的矛盾描述；都不触发返回 None。"""
    # 日期优先于数字，因为日期里也包含数字
    conflict = check_date_conflict(source_a, source_b)
    if conflict:
        return conflict

    conflict = check_number_conflict(source_a, source_b)
    if conflict:
        return conflict

    conflict = check_name_conflict(source_a, source_b)
    if conflict:
        return conflict

    conflict = check_status_conflict(source_a, source_b)
    if conflict:
        return conflict

    return None


if __name__ == "__main__":
    # 测试 1：数字冲突
    a = {"id": "doc_A", "text": "公司年假是10天"}
    b = {"id": "doc_B", "text": "公司年假是5天"}
    print("测试 1 - 数字冲突：")
    print(detect_conflict(a, b))
    print()

    # 测试 2：日期冲突
    c = {"id": "doc_C", "text": "项目截止6月1日"}
    d = {"id": "doc_D", "text": "项目截止7月1日"}
    print("测试 2 - 日期冲突：")
    print(detect_conflict(c, d))
    print()

    # 测试 3：人名冲突
    e = {"id": "doc_E", "text": "负责人是张三"}
    f = {"id": "doc_F", "text": "负责人是李四"}
    print("测试 3 - 人名冲突：")
    print(detect_conflict(e, f))
    print()

    # 测试 4：状态冲突
    g = {"id": "doc_G", "text": "本产品支持安卓"}
    h = {"id": "doc_H", "text": "本产品不支持安卓"}
    print("测试 4 - 状态冲突：")
    print(detect_conflict(g, h))
    print()

    # 测试 5：不冲突
    i = {"id": "doc_I", "text": "年假申请需提前三天"}
    print("测试 5 - 不冲突：")
    print(detect_conflict(a, i))