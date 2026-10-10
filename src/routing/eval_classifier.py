# src/routing/eval_classifier.py
"""
在 qa_extended.json 上评估规则分类器的准确率。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.routing.query_classifier import classify_query


def load_qa(path: str = "data/golden_qa/qa_extended.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    qa = load_qa()
    total = len(qa)
    correct = 0
    wrong_cases = []

    for item in qa:
        q = item["question"]
        gold_type = item.get("query_type", "unknown")
        pred = classify_query(q)
        # 只有有 query_type 标注的才评估
        if gold_type == "unknown":
            continue
        if pred.query_type == gold_type:
            correct += 1
        else:
            wrong_cases.append((q, gold_type, pred.query_type, pred.reason))

    print(f"总计 {total} 条")
    print(f"正确 {correct} 条，准确率 {correct/total:.2%}")
    print()
    if wrong_cases:
        print("=== 错误案例 ===")
        for q, g, p, r in wrong_cases:
            print(f"  Q: {q}")
            print(f"     期望={g}  预测={p}  原因={r}")
    else:
        print("全部正确！")


if __name__ == "__main__":
    main()