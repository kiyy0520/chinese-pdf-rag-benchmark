# Week 3 Day 3：条件触发 Rerank 决策记录

## 1. 任务概述

在向量检索 + BM25 初筛的基础上，实现条件触发 Rerank：
- 先计算候选集质量（Top-K 相似度方差、BM25 与向量得分一致性）。
- 只在质量不达标时触发 BGE-Reranker 精排。
- 对比纯向量、始终 Rerank、条件 Rerank 的 Recall@5、MRR、延迟。

## 2. 阈值选择依据

| 参数 | 取值 | 依据 |
|---|---|---|
| variance_threshold | 0.01 | 观察候选集向量分数分布，低于 0.01 说明分数挤在一起，初筛区分度不足 |
| consistency_threshold | 0.66 | 一致性只有 0.33、0.66、1.0 三种取值；0.66 可避免浮点精度误判（2/3=0.6666...） |
| candidate_k | 20 | 两路各取 20 个候选，去重后约 20~25 个，兼顾召回和延迟 |

## 3. 触发比例

- 全量 20 条 query，触发 Rerank 10 条。
- **触发比例：50%**，落在合理区间（20%~60%）。

## 4. 效果对比

| 方法 | Recall@5 | MRR | 平均延迟 |
|---|---|---|---|
| pure_vector | 0.60 | 0.60 | ~0.005 ms |
| always_rerank | 0.70 | 0.70 | ~248 ms |
| conditional_rerank | 0.70 | 0.70 | ~135 ms |

**结论**：
- 条件 Rerank 效果与始终 Rerank 完全一致。
- 相比纯向量，Recall@5 和 MRR 各提升 10 个百分点。
- 延迟增加约 135ms，< 200ms 目标达标。
- 平均节省约 113ms/query（相比始终 Rerank）。

## 5. Reranker 反效果 case 记录

全量 20 条 query 中，未发现 Reranker 降低 Recall@5 或 MRR 的 case。

早期小样本曾出现 1 条 MRR 下降（基于模型的强化学习），后因评估方式优化而消失。

**失效条件**：
- 数据集换了：不同领域、不同语言的分数分布完全不同，阈值需要重调。
- 检索器变了：换 embedding 模型或 BM25 实现，分数分布会变。
- 候选集数量变了：候选数从 20 变成 100，方差自然变小，阈值不准。
- query 类型偏了：全是短 query 或长 query，方差分布会偏移。
- 候选数只有 1 个：方差恒为 0、一致性恒为 1，判断失去意义（已加保护：候选数 < 3 直接触发）。

## 6. 交付物清单

- `src/retrieval/conditional_reranker.py`：条件 Rerank 实现。
- `experiments/week3_day3_rerank_compare.csv`：全量 20 条对比结果。
- `experiments/week3_day3_rerank_decision.md`：本决策记录。

## 7. 阈值参数扫描（markdown_recursive_512_0）

在 markdown_recursive_512_0 配置下，对三组阈值做扫描：

| 组名 | variance_threshold | consistency_threshold | 触发比例 | Recall@5 | MRR | 平均延迟 |
|---|---|---|---|---|---|---|
| loose | 0.005 | 0.33 | 10% | 0.700 | 0.675 | 27 ms |
| baseline | 0.01 | 0.66 | 45% | 0.750 | 0.725 | 120 ms |
| strict | 0.02 | 1.0 | 90% | 0.750 | 0.725 | 226 ms |

**结论**：
- loose 组效果下降明显（Recall@5 从 0.75 → 0.70），不可接受。
- strict 组效果与 baseline 完全一致，但触发比例翻倍、延迟翻倍，性价比低。
- baseline 组在效果与延迟间取得最佳平衡。

**最终选择**：baseline（variance_threshold=0.01, consistency_threshold=0.66）。

**失效条件**：
- 若评估集换成另一领域，方差分布不同，需重扫阈值。
- 若候选集大小显著变化（如 candidate_k 从 20 改为 100），方差会自然变小，阈值需重调。
- 若 Reranker 换成其他模型（如 large 版），打分分布变化，阈值需重新验证。