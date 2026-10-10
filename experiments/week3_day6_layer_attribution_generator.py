# experiments/week3_day6_layer_attribution_generator.py
import os
import pandas as pd

def run_layer_attribution():
    ragas_csv_path = "experiments/week3_day6_ragas_results.csv"
    if not os.path.exists(ragas_csv_path):
        print(f"❌ 找不到 {ragas_csv_path}，请先跑完 RAGAS 评估生成该文件！")
        return

    df = pd.read_csv(ragas_csv_path)
    total_samples = len(df)  # 核心：总样本数是 30，不是 10！
    print(f"📊 总评估样本数: {total_samples}")

    # 1. 捞取 badcase：放宽阈值，只要有一项不是满分就进入候选池
    badcases = df[(df["faithfulness"] < 1.0) | (df["context_recall"] < 1.0)].copy()
    print(f"🚨 捞取到的 badcase 数量: {len(badcases)}")

    # 2. 核心：设计 4 层归因规则 (使用 if/elif/else)
    def attribute_layer(row):
        scheme = row["scheme"]
        recall = row["context_recall"]
        precision = row["llm_context_precision_with_reference"]
        faithfulness = row["faithfulness"]

        # 【纯 LLM 方案】：没有检索行为，Answer全凭自身记忆，归为 Query 理解（无检索辅助）
        if scheme == "纯 LLM":
            return "Query理解"
        
        # 【基础 RAG / 自适应 RAG 方案】：有检索行为，按指标判定
        if recall < 0.6:
            return "检索"       # 完全没找对资料
        elif precision < 0.6:
            return "重排"       # 资料找对了，但排得太靠后
        elif faithfulness < 0.6:
            return "生成"       # 资料给对了，但大模型乱答
        else:
            return "生成"       # 兜底：分数都及格但有微瑕，归为生成层微调

    badcases["failure_layer"] = badcases.apply(attribute_layer, axis=1)

    # 3. 生成归因理由
    def generate_reason(row):
        return f"分数: Faith={row['faithfulness']:.2f}, Prec={row['llm_context_precision_with_reference']:.2f}, Recall={row['context_recall']:.2f}"
    badcases["reason"] = badcases.apply(generate_reason, axis=1)

    # 4. 输出明细表
    output_cols = ["scheme", "user_input", "failure_layer", "reason", "faithfulness", "llm_context_precision_with_reference", "context_recall"]
    badcases[output_cols].rename(columns={"user_input": "question"}).to_csv(
        "experiments/week3_day6_layer_attribution.csv", index=False, encoding="utf-8-sig"
    )
    print("✅ 已生成归因明细表: experiments/week3_day6_layer_attribution.csv")

    # 5. 计算每层错误数
    layer_counts = badcases["failure_layer"].value_counts().reset_index()
    layer_counts.columns = ["failure_layer", "error_count"]

    # 6. 强制补齐 4 层（这是解决“重排和生成层消失”的关键！）
    all_layers = pd.DataFrame({"failure_layer": ["Query理解", "检索", "重排", "生成"]})
    layer_summary = pd.merge(all_layers, layer_counts, on="failure_layer", how="left")
    
    # 7. 把 NaN 填充为 0，并计算错误率（分母是 total_samples）
    layer_summary["error_count"] = layer_summary["error_count"].fillna(0)
    layer_summary["error_rate"] = (layer_summary["error_count"] / total_samples).round(4)

    # 8. 保存归因汇总表
    layer_summary.to_csv("experiments/week3_day6_layer_attribution_summary.csv", index=False, encoding="utf-8-sig")
    print("✅ 已生成归因汇总表（带错误率）: experiments/week3_day6_layer_attribution_summary.csv")
    print("\n📊 分层错误率概览：")
    print(layer_summary.to_string(index=False))

if __name__ == "__main__":
    run_layer_attribution()