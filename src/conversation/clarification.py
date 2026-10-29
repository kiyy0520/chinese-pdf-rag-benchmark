# src/conversation/clarification.py

def build_clarification(pronoun: str, candidates: list[str]) -> str:
    """根据指代词和候选实体，生成澄清话术"""
    quoted = [f"“{c}”" for c in candidates]

    if len(quoted) == 1:
        return f"您说的“{pronoun}”是指{quoted[0]}吗？"

    if len(quoted) == 2:
        return f"您说的“{pronoun}”是指{quoted[0]}还是{quoted[1]}？"

    # 三个及以上
    head = "、".join(quoted[:-1])
    tail = quoted[-1]
    return f"您说的“{pronoun}”是指{head}还是{tail}？"