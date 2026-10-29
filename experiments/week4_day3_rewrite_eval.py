import csv
import json
import re

# ==========================================
# 1. 20 条测试数据（覆盖指代、切换话题、过度改写等）
# ==========================================
dataset = [
    # --- 简单指代 ---
    {"id": "case_01", "history": [{"role": "user", "content": "我想看看你们家的降噪耳机"}, {"role": "assistant", "content": "好的，我们有 AirPro 降噪耳机和 Lite 降噪耳机。"}], "current_query": "它多少钱？", "gold_rewrite": "降噪耳机多少钱？", "must_include_entities": ["降噪耳机"], "must_not_include_entities": ["手机套餐"]},
    {"id": "case_02", "history": [{"role": "user", "content": "AirPro 支持降噪吗？"}, {"role": "assistant", "content": "支持的，AirPro 有主动降噪功能。"}], "current_query": "这个有优惠吗？", "gold_rewrite": "AirPro 有优惠吗？", "must_include_entities": ["AirPro"], "must_not_include_entities": ["Lite", "降噪"]},
    
    # --- 多轮指代 / 限定词 ---
    {"id": "case_03", "history": [{"role": "user", "content": "Lite 降噪耳机有黑色吗？"}, {"role": "assistant", "content": "有的，Lite 降噪耳机有黑色和白色。"}], "current_query": "那黑色款多少钱？", "gold_rewrite": "Lite 降噪耳机的黑色款多少钱？", "must_include_entities": ["Lite", "降噪耳机", "黑色"], "must_not_include_entities": ["白色"]},
    {"id": "case_04", "history": [{"role": "user", "content": "那第二款呢？"}, {"role": "assistant", "content": "第二款是 Lite 降噪耳机，售价 399 元。"}], "current_query": "它有现货吗？", "gold_rewrite": "Lite 降噪耳机有现货吗？", "must_include_entities": ["Lite", "降噪耳机"], "must_not_include_entities": ["AirPro", "399"]},
    
    # --- 话题切换 ---
    {"id": "case_05", "history": [{"role": "user", "content": "AirPro 降噪耳机多少钱？"}, {"role": "assistant", "content": "AirPro 是 899 元。"}], "current_query": "你们支持七天无理由退货吗？", "gold_rewrite": "你们支持七天无理由退货吗？", "must_include_entities": ["七天无理由退货"], "must_not_include_entities": ["AirPro", "降噪耳机", "899 元"]},
    {"id": "case_06", "history": [{"role": "user", "content": "发货地是哪里？"}, {"role": "assistant", "content": "我们是深圳发货。"}], "current_query": "那大概几天能到北京？", "gold_rewrite": "从深圳发货大概几天能到北京？", "must_include_entities": ["北京", "深圳"], "must_not_include_entities": ["广州", "上海"]},
    
    # --- 容易过度改写 ---
    {"id": "case_07", "history": [{"role": "user", "content": "我之前买过你们家的手机，现在想买耳机。"}, {"role": "assistant", "content": "好的，请问想看哪款耳机？"}], "current_query": "有便宜的吗？", "gold_rewrite": "有便宜的耳机吗？", "must_include_entities": ["耳机"], "must_not_include_entities": ["手机"]},
    {"id": "case_08", "history": [{"role": "user", "content": "降噪耳机和普通耳机有什么区别？"}, {"role": "assistant", "content": "降噪耳机能隔绝环境音，普通耳机不行。"}], "current_query": "那价格呢？", "gold_rewrite": "降噪耳机和普通耳机的价格有什么区别？", "must_include_entities": ["降噪耳机", "普通耳机", "价格"], "must_not_include_entities": ["手机", "赠品"]},
    
    # --- 容易改写不足 ---
    {"id": "case_09", "history": [{"role": "user", "content": "这个耳机保修多久？"}, {"role": "assistant", "content": "保修一年。"}], "current_query": "那如果是人为损坏呢？", "gold_rewrite": "耳机人为损坏保修多久？", "must_include_entities": ["耳机", "人为损坏"], "must_not_include_entities": ["手机", "自然损坏"]},
    {"id": "case_10", "history": [{"role": "user", "content": "刚才那个 AirPro 耳机能折叠吗？"}, {"role": "assistant", "content": "不能折叠。"}], "current_query": "那 Lite 呢？", "gold_rewrite": "Lite 降噪耳机能折叠吗？", "must_include_entities": ["Lite", "降噪耳机", "折叠"], "must_not_include_entities": ["AirPro"]},

    # --- 更多复杂多轮 ---
    {"id": "case_11", "history": [{"role": "user", "content": "帮我推荐一款运动耳机。"}, {"role": "assistant", "content": "推荐 Sport 款，防水防汗。"}], "current_query": "它续航多久？", "gold_rewrite": "Sport 运动耳机续航多久？", "must_include_entities": ["Sport", "运动耳机", "续航"], "must_not_include_entities": ["AirPro"]},
    {"id": "case_12", "history": [{"role": "user", "content": "Sport 款有蓝色吗？"}, {"role": "assistant", "content": "只有黑色和红色。"}], "current_query": "那红色什么时候补货？", "gold_rewrite": "Sport 红色款什么时候补货？", "must_include_entities": ["Sport", "红色", "补货"], "must_not_include_entities": ["蓝色", "黑色"]},
    {"id": "case_13", "history": [{"role": "user", "content": "我想买头戴式耳机。"}, {"role": "assistant", "content": "头戴式有 H1 和 H2 两款。"}], "current_query": "哪个更轻？", "gold_rewrite": "H1 和 H2 头戴式耳机哪个更轻？", "must_include_entities": ["H1", "H2", "头戴式耳机", "轻"], "must_not_include_entities": ["入耳式"]},
    {"id": "case_14", "history": [{"role": "user", "content": "H1 耳机多少钱？"}, {"role": "assistant", "content": "H1 售价 599 元。"}], "current_query": "有分期免息吗？", "gold_rewrite": "H1 耳机有分期免息吗？", "must_include_entities": ["H1", "耳机", "分期免息"], "must_not_include_entities": ["H2", "599"]},
    {"id": "case_15", "history": [{"role": "user", "content": "你们发什么快递？"}, {"role": "assistant", "content": "默认发顺丰。"}], "current_query": "可以指定发中通吗？", "gold_rewrite": "可以指定发中通快递吗？", "must_include_entities": ["中通", "快递"], "must_not_include_entities": ["顺丰"]},
    {"id": "case_16", "history": [{"role": "user", "content": "这个耳机支持苹果手机吗？"}, {"role": "assistant", "content": "支持的，蓝牙 5.0 通用。"}], "current_query": "那安卓呢？", "gold_rewrite": "耳机支持安卓手机吗？", "must_include_entities": ["耳机", "安卓"], "must_not_include_entities": ["苹果", "蓝牙"]},
    {"id": "case_17", "history": [{"role": "user", "content": "刚买的耳机坏了。"}, {"role": "assistant", "content": "非常抱歉，请问是哪里坏了？"}], "current_query": "右耳没声音了。", "gold_rewrite": "新买的耳机右耳没声音了。", "must_include_entities": ["耳机", "右耳", "没声音"], "must_not_include_entities": ["左耳"]},
    {"id": "case_18", "history": [{"role": "user", "content": "可以开发票吗？"}, {"role": "assistant", "content": "可以开电子普票。"}], "current_query": "那专票呢？", "gold_rewrite": "可以开增值税专用发票吗？", "must_include_entities": ["专票", "增值税专用发票"], "must_not_include_entities": ["电子普票"]},
    {"id": "case_19", "history": [{"role": "user", "content": "耳机外壳什么材质？"}, {"role": "assistant", "content": "是塑料材质。"}], "current_query": "容易刮花吗？", "gold_rewrite": "塑料材质的耳机外壳容易刮花吗？", "must_include_entities": ["耳机", "外壳", "刮花", "塑料"], "must_not_include_entities": ["金属"]},
    {"id": "case_20", "history": [{"role": "user", "content": "我要退货。"}, {"role": "assistant", "content": "好的，请提供订单号。"}], "current_query": "退到哪里？", "gold_rewrite": "退货寄回哪里？", "must_include_entities": ["退货", "地址"], "must_not_include_entities": ["订单号"]}
]

# ==========================================
# 2. 核心函数
# ==========================================
def mock_llm_rewrite(current_query, history):
    """模拟 LLM 改写。这里我故意写死了结果来模拟各种失效。"""
    if current_query == "它多少钱？": return "AirPro 降噪耳机和 Lite 降噪耳机多少钱？" # 过度改写
    if current_query == "这个有优惠吗？": return "AirPro 有优惠吗？" # 成功
    if current_query == "那黑色款多少钱？": return "Lite 降噪耳机多少钱？" # 改写不足(漏黑色)
    if current_query == "它有现货吗？": return "Lite 降噪耳机有现货吗？" # 成功
    if current_query == "你们支持七天无理由退货吗？": return "AirPro 降噪耳机支持七天无理由退货吗？" # 话题误判
    if current_query == "那大概几天能到北京？": return "快递几天能到北京？" # 改写不足(漏深圳)
    if current_query == "有便宜的吗？": return "有便宜的耳机吗？" # 成功
    if current_query == "那价格呢？": return "降噪耳机和普通耳机价格多少钱？" # 成功
    if current_query == "那如果是人为损坏呢？": return "那人为损坏保修多久？" # 改写不足(漏耳机)
    if current_query == "那 Lite 呢？": return "Lite 降噪耳机能折叠吗？" # 成功
    if current_query == "它续航多久？": return "Sport 运动耳机续航多久？" # 成功
    if current_query == "那红色什么时候补货？": return "红色什么时候补货？" # 改写不足(漏Sport)
    if current_query == "哪个更轻？": return "H1 和 H2 头戴式耳机哪个更轻？" # 成功
    if current_query == "有分期免息吗？": return "H1 耳机有分期免息吗？" # 成功
    if current_query == "可以指定发中通吗？": return "可以指定发中通吗？" # 成功
    if current_query == "那安卓呢？": return "耳机支持安卓手机吗？" # 成功
    if current_query == "右耳没声音了。": return "刚买的耳机右耳没声音了。" # 成功
    if current_query == "那专票呢？": return "可以开增值税专用发票吗？" # 成功
    if current_query == "容易刮花吗？": return "耳机外壳容易刮花吗？" # 改写不足(漏塑料)
    if current_query == "退到哪里？": return "退货寄回哪里？" # 成功
    return current_query

def extract_entities_simple(text):
    words = re.split(r'[，。？\s]', text)
    return [w for w in words if w]

def match_entities(rewritten, must_include, must_not_include):
    included_hit = [e for e in must_include if e in rewritten]
    forbidden_hit = [e for e in must_not_include if e in rewritten]
    return included_hit, forbidden_hit

def compute_metrics(must_include, included_hit, forbidden_hit):
    recall = 1.0 if len(must_include) == 0 else len(included_hit) / len(must_include)
    denom = len(included_hit) + len(forbidden_hit)
    precision = 1.0 if denom == 0 else len(included_hit) / denom
    return precision, recall

def should_fallback(current_query_entities, rewritten_entities, threshold=0.5):
    if len(current_query_entities) == 0: return False
    common = set(current_query_entities) & set(rewritten_entities)
    overlap = len(common) / len(current_query_entities)
    return overlap < threshold

# ==========================================
# 3. 批量执行主循环
# ==========================================
rows = []
for sample in dataset:
    rewritten = mock_llm_rewrite(sample["current_query"], sample["history"])
    included_hit, forbidden_hit = match_entities(rewritten, sample["must_include_entities"], sample["must_not_include_entities"])
    missing = [e for e in sample["must_include_entities"] if e not in included_hit]
    precision, recall = compute_metrics(sample["must_include_entities"], included_hit, forbidden_hit)
    
    cur_ents = extract_entities_simple(sample["current_query"])
    rew_ents = extract_entities_simple(rewritten)
    fallback = should_fallback(cur_ents, rew_ents)
    
    final_query = sample["current_query"] if fallback else rewritten
    
    rows.append({
        "id": sample["id"], "current_query": sample["current_query"],
        "rewritten_query": rewritten, "final_query": final_query,
        "must_include_entities": json.dumps(sample["must_include_entities"], ensure_ascii=False),
        "must_not_include_entities": json.dumps(sample["must_not_include_entities"], ensure_ascii=False),
        "included_hit": json.dumps(included_hit, ensure_ascii=False),
        "missing": json.dumps(missing, ensure_ascii=False),
        "forbidden_hit": json.dumps(forbidden_hit, ensure_ascii=False),
        "precision": round(precision, 3), "recall": round(recall, 3),
        "fallback_triggered": fallback
    })

with open("experiments/week4_day3_rewrite_eval.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("评估完成！")