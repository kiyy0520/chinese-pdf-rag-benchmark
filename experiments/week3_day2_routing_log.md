# Week 3 Day 2：路由链路实现 + 对接

## 一、任务回顾

把 Day 1 的"分诊台图纸"变成真正能派活的实体：
- 接通 3 个检索器（BM25 / 向量 / 混合）
- 写 `router.py` 实现分类 → 分发 → 调用
- 测量路由延迟，确保 < 50ms

## 二、测试环境

- 日期：2026-09-30
- 脚本：`test_routing.py`
- 集合名：`day4_chunking_eval`
- 检索器：`BM25Retriever` / `VectorRetriever` (BGE) / `HybridRetriever` (RRF)
- 路由实现：`src/routing/router.py`

## 三、路由链路结构
query
↓
classify_query() → ClassificationResult(query_type, confidence, reason)
↓
ROUTING_TABLE.get(query_type, SKIP) → RouteTarget
↓
route_skip / route_exact / route_semantic / route_hybrid
↓
返回统一字典：{route, query, results, category, routing_latency_ms, total_latency_ms}


## 四、Day 2 路由决策表（更新版）

| 层级 | 名称 | 判定条件 | 路由目标 | 实际对接 | 路由延迟 | 失效条件 |
|---|---|---|---|---|---|---|
| L0 | 超范围拒答 | Transformer/Q-learning/DQN 等 | 直接拒答（归入 skip） | `route_skip()` | ~0.03ms | 主题词未覆盖时漏判 |
| L1 | 跳过检索 | 闲聊/问候/天气 | 直接回复，不检索 | `route_skip()` | 0.00~0.15ms | 用户伪装闲聊问知识 → 漏答 |
| L2 | 精确查询 | γ、五元组、定义、公式 | BM25 top-5 | `BM25Retriever.retrieve()` | 0.11~0.44ms | 同义表述无关键词 → BM25 漏召 |
| L3 | 语义查询 | 什么是/为什么/如何理解 | 向量 top-5 | `VectorRetriever.retrieve()` | 0.01~0.05ms | 术语密集但无信号 → 向量偏 |
| L4 | 混合查询 | 区别/比较/步骤/关系 | RRF 融合 top-5 | `HybridRetriever.retrieve()` | 0.04~0.05ms | 单路强命中被另一路噪音挤占 |

**Day 2 新增结论**：
- 平均路由延迟 **约 0.11ms**，远低于 50ms 目标 ✅
- 路由决策与检索执行**完全解耦**：路由 0.1ms 级，检索 60~400ms 级
- `total_latency_ms` 首次调用偏高（模型预热），不纳入路由考核

## 五、测试结果汇总

| 路由 | 测试 query 数 | 预期命中 | 实际命中 | 路由延迟范围 |
|------|--------------|---------|---------|-------------|
| skip | 3 | 3 | 3 | 0.00 ~ 0.15 ms |
| exact | 3 | 3 | 3 | 0.11 ~ 0.44 ms |
| semantic | 3 | 3 | 3 | 0.01 ~ 0.05 ms |
| hybrid | 3 | 3 | 3 | 0.04 ~ 0.05 ms |

**平均路由延迟：约 0.11 ms**（远低于 50 ms 目标 ✅）

## 六、详细日志

| query | 分类结果 | 路由目标 | 路由耗时(ms) | 总耗时(ms) | 结果数 |
|-------|---------|---------|-------------|-----------|--------|
| 你好，你是谁？ | skip | skip | 0.14 | 0.15 | 0 |
| 今天天气怎么样？ | skip | skip | 0.15 | 0.15 | 0 |
| 谢谢你的帮助 | skip | skip | 0.00 | 0.00 | 0 |
| MDP 的五元组是什么？ | exact | exact | 0.44 | 0.64 | 5 |
| 折扣因子 γ 代表什么？ | exact | exact | 0.18 | 0.29 | 5 |
| 价值函数的公式如何表示？ | exact | exact | 0.11 | 0.20 | 5 |
| 什么是马尔可夫过程？ | semantic | semantic | 0.01 | 155.19 | 5 |
| 为什么要引入折扣因子？ | semantic | semantic | 0.05 | 59.94 | 5 |
| 如何理解贝尔曼方程的意义？ | semantic | semantic | 0.05 | 63.86 | 5 |
| 价值迭代和策略迭代有哪些区别？ | hybrid | hybrid | 0.04 | 133.44 | 5 |
| 描述 MDP 的动态过程。 | hybrid | hybrid | 0.05 | 61.17 | 5 |
| 价值迭代和策略迭代的关系是什么？ | hybrid | hybrid | 0.04 | 63.05 | 5 |

## 七、错误分类 case 记录

### Case 1：Q-learning / SARSA 比较问题

- **query**：比较 Q-learning 和 SARSA 的异同。
- **预期**：hybrid（含"比较""异同"）
- **实际**：out-of-scope → skip
- **原因**：分类器规则优先级为 `skip → out-of-scope → hybrid → semantic → exact`。`out-of-scope` 高于 `hybrid`，`q-learning`、`sarsa` 命中超范围词，被一票否决。
- **分析**：
  - 从**语料范围**看：Q-learning/SARSA 不在 PDF 里，判 skip 是合理且正确的。
  - 从**测试意图**看：这条 query 原本是为测 hybrid，导致 hybrid 测试覆盖出现缺口。
  - 因此这不是"系统错了"，而是"预期与实际的差异"，属于**测试覆盖缺口**。
- **行动**：不改分类器优先级（改了会误伤其他对比问题，导致去检索不存在的文档）。另补一条 hybrid 测试 query"价值迭代和策略迭代的关系是什么？"替换原 query，确保 hybrid 覆盖 3 条。

## 八、可量化验证对照

| 验证项 | 目标 | 实际 | 状态 |
|---|---|---|---|
| 4 条路由全部跑通 | 是 | 是 | ✅ |
| 每条路由至少 3 个测试 query | 是 | 是（共 12 条） | ✅ |
| 平均路由延迟 < 50ms | < 50ms | 约 0.11ms | ✅ |
| 错误分类 case 记录并分析 | 是 | 1 条（Q-learning/SARSA） | ✅ |
| 日志含 query/分类/路由/耗时 | 是 | 是 | ✅ |

## 九、当日产出

- [x] `src/routing/router.py`
- [x] `experiments/week3_day2_routing_log.md`
- [x] 更新后的路由决策表

## 十、下一步

- Day 3 预计：把 router 接入 `rag_query` 主流程，实现"分类 → 检索 → 生成"完整链路
- 待补：5 条超范围题的拒答实验（当前只分类 + 路由，未做真正的拒答话术）
- 未来优化：规则分类器 + LLM 分类器双轨，处理"V* 和 π* 的关系"这类语义歧义