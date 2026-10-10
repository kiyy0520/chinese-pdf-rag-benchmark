# src/routing/query_classifier.py
"""
规则层 Query 分类器

四层路由：
- skip: 跳过检索（闲聊、问候、超范围）
- exact: 精确查询（术语/符号/公式，走 BM25）
- semantic: 语义查询（概念理解，走向量）
- hybrid: 混合查询（对比/步骤/综合，走 BM25+向量）
"""
import re
from dataclasses import dataclass


@dataclass
class ClassificationResult:
    query: str
    query_type: str          # skip / exact / semantic / hybrid / out-of-scope
    confidence: float        # 0.0 - 1.0
    reason: str              # 命中的规则说明


# ============ 规则配置 ============

SKIP_PATTERNS = [
    r"^你好",
    r"^您好",
    r"^hi\b",
    r"^hello\b",
    r"^谢谢",
    r"^感谢",
    r"天气",
    r"你是谁",
    r"你能做什么",
    r"介绍一下你自己",
]

OUT_OF_SCOPE_PATTERNS = [
    r"transformer",
    r"自注意力",
    r"attention",
    r"q-?learning",
    r"sarsa",
    r"dqn",
    r"ppo\b",
    r"alphago",
    r"bert\b",
    r"gpt\b",
    r"神经网络",
    r"深度学习框架",
    r"pytorch",
    r"tensorflow",
]

HYBRID_PATTERNS = [
    r"区别", r"比较", r"对比", r"vs\.?", r"异同",
    r"步骤", r"流程", r"哪些过程", r"哪两种", r"哪几种",
    r"各自", r"分别", r"关系", r"又.*又", r"和.*的区别",
    r"描述",      # 保留（"描述MDP的动态过程"靠它命中）
    # r"过程",    # ❌ 删除，太宽泛
]

SEMANTIC_PATTERNS = [
    r"什么是",     # 🆕 "什么是X"是概念性问题
    r"为什么",
    r"如何理解",
    r"目的是什么",
    r"作用是什么",
    r"意义",
    r"理念",
    r"思想",
]

EXACT_SIGNALS = [
    r"[A-Z]\^?\*?",
    r"γ|β|α|π|ρ|θ|λ",
    r"P_?\{?[a-z]+\}?",
    r"五元组",
    # r"是什么",   # ❌ 删除，交给 semantic 的"什么是"处理
    r"等式",
    r"定义",
    r"代表什么",
    r"含义",
    r"公式",
    r"表示为",
    r"如何表示",
    r"如何估计",
]


# ============ 核心分类函数 ============

def classify_query(query: str) -> ClassificationResult:
    q = query.strip()
    q_lower = q.lower()

    # 1. skip
    for pat in SKIP_PATTERNS:
        if re.search(pat, q_lower):
            return ClassificationResult(q, "skip", 0.95, f"命中 skip 模式: {pat}")

    # 2. out-of-scope
    for pat in OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, q_lower):
            return ClassificationResult(q, "out-of-scope", 0.90, f"命中超范围模式: {pat}")

    # 3. hybrid
    for pat in HYBRID_PATTERNS:
        if re.search(pat, q):
            return ClassificationResult(q, "hybrid", 0.85, f"命中 hybrid 模式: {pat}")

    # 4. semantic（🆕 提前到 exact 之前）
    for pat in SEMANTIC_PATTERNS:
        if re.search(pat, q):
            return ClassificationResult(q, "semantic", 0.75, f"命中 semantic 模式: {pat}")

    # 5. exact
    for pat in EXACT_SIGNALS:
        if re.search(pat, q):
            return ClassificationResult(q, "exact", 0.80, f"命中 exact 信号: {pat}")

    # 6. 默认 semantic
    return ClassificationResult(q, "semantic", 0.50, "无规则命中，默认 semantic")


def classify_batch(queries: list[str]) -> list[ClassificationResult]:
    return [classify_query(q) for q in queries]


if __name__ == "__main__":
    test_queries = [
        "你好，请问你能做什么？",
        "在MDP的五元组中，γ代表什么？",
        "为什么要有折扣因子？",
        "价值迭代和策略迭代有哪些区别？",
        "请解释Transformer中的自注意力机制。",
        "什么是马尔可夫过程？",
        "描述MDP的动态过程（从状态s0开始）。",
        "折扣因子γ 的目的是什么？",
    ]
    for q in test_queries:
        r = classify_query(q)
        print(f"[{r.query_type:>12}] conf={r.confidence:.2f} | {r.query[:30]} | {r.reason}")