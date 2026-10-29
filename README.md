# chinese-pdf-rag-benchmark

> 一个面向中文强化学习教材的 PDF 知识库问答系统。支持上传 PDF、自适应路由检索、条件 Rerank、引用溯源、冲突检测、拒答、**多轮对话状态追踪与 Query 改写**、LLM 生成、FastAPI 服务化、RAGAS 自动评估及 Docker 一键交付。

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)

---

## 项目简介

本项目以中文强化学习课件（马尔可夫决策过程、强化学习基础）为语料，构建了一套完整的**自适应 RAG 流水线**，覆盖：

**PDF 解析 → 文本切分 → 向量化 → 混合检索 → 条件 Rerank → 引用溯源 → 冲突检测 → 拒答 → LLM 生成 → 分层归因 → RAGAS 评估 → API 服务化 → Docker 交付**

同时集成 **LangChain Agent**、**规则路由分类器**，并在 Week 4 深度攻克多轮对话“越聊越蠢”的痛点：引入结构化对话状态对象、Query 改写、歧义检测与主动澄清、话题切换降级机制，让系统根据 Query 类型与对话历史自主选择检索路径，实现“按需检索”与“智能多轮”，并通过分层归因表量化系统瓶颈。

### 三周迭代路线

| 阶段 | 主题 | 核心交付 |
|---|---|---|
| Week 2 | 基础 RAG 全链路 | PDF 解析、切分、Embedding、Chroma、BM25/RRF、生成、FastAPI、RAGAS、Docker |
| Week 3 | 自适应检索与质量保障 | Query 分类器、4 条路由、条件 Rerank、引用溯源、冲突检测、自适应拒答、分层归因 |
| Week 4 | 多轮对话与失效归因 | 结构化状态对象、Query Rewriting、歧义澄清、话题切换降级、多轮 RAG、四方案对比 |

## 核心特性

### 基础 RAG 能力（Week 2）
- **端到端 RAG 链路**：PDF → 切分 → 向量化 → 检索 → 生成，一步到位。
- **FastAPI 服务化**：`/upload` 上传 PDF，`/ask` 提问，自动生成 Swagger 文档。
- **RAGAS 自动评估**：Faithfulness / Context Precision / Context Recall 三指标。
- **Agent 工具化**：`search_pdf` 工具可被 LLM 自主调用。
- **Docker 一键部署**：`docker build` + `docker run` 即可复现。
- **可复现实验记录**：9 组切分对比、3 种检索方案对比、10 条端到端测试记录、badcase 归因。

### 自适应质量保障（Week 3）
- **自适应路由**：4 条检索路径（skip / exact / semantic / hybrid），基于规则分类器自动分发。
- **条件 Rerank**：仅在检索一致性低时触发 Rerank，平衡性能与延迟。
- **引用溯源**：每条回答附带来源（source + chunk_id + 相似度），可回溯原文。
- **冲突检测**：多文档出现矛盾时主动提示用户。
- **自适应拒答**：超出知识库范围或检索置信度低时拒绝作答。
- **分层归因**：Query 理解 / 检索 / 重排 / 生成 四层错误率量化。

### 多轮对话与状态管理（Week 4）
- **结构化对话状态对象**：维护 ≥5 个字段（意图、硬约束、检索 Query、话题切换标记、澄清标记等），解决多轮指代消解。
- **Query 改写与质量评估**：指代不明时自动重写 Query，并用精确率/召回率量化改写质量。
- **主动澄清**：极度模糊时触发澄清追问，降低瞎答风险。
- **话题切换降级**：改写后 Query 与当前 Query 实体重叠度过低时，降级为只用当前 Query 检索。
- **多轮 RAG 链路**：状态对象驱动检索，检索 Query 只继承“当前意图 + 硬约束”，不携带全部历史。
- **多轮失效分层归因**：按 Query 理解 / 检索 / 重排 / 生成四层标注多轮 badcase。

## 技术栈

| 类别 | 选型 |
|---|---|
| LLM | DeepSeek Chat（OpenAI 兼容模式，也支持 Qwen-Plus） |
| Embedding | BAAI/bge-large-zh-v1.5（sentence-transformers） |
| 向量库 | ChromaDB 0.5.23（持久化） |
| 检索 | BM25 / 向量 / Hybrid（RRF 融合） |
| 重排 | BGE-Reranker（条件触发） |
| PDF 解析 | pypdf / PyMuPDF |
| 后端 | FastAPI + Uvicorn |
| Agent | LangChain Tool + AgentExecutor |
| 多轮记忆 | LangChain RunnableWithMessageHistory + 结构化状态对象 |
| 评估 | RAGAS 0.4.3 |
| 依赖管理 | uv（容器内基于 Python 3.10 自动解析） |
| 容器化 | Docker（python:3.10-slim） |

## 目录结构

```text
chinese-pdf-rag-benchmark/
├── src/
│   ├── api/                  # FastAPI 服务与 RAG 链路
│   │   ├── main.py           # 上传、提问接口
│   │   └── rag_chain.py      # 检索 → Prompt → LLM 生成
│   ├── conversation/         # Week 4：多轮对话与状态管理
│   │   ├── state.py          # 结构化状态对象（Pydantic）
│   │   ├── state_tracker.py  # 状态追踪与更新
│   │   ├── memory.py         # 滑动窗口 / Token 预算 / 摘要压缩
│   │   ├── query_rewriter.py # Query Rewriting
│   │   ├── ambiguity_detector.py # 歧义检测
│   │   ├── clarification.py  # 主动澄清机制
│   │   ├── topic_switch.py   # 话题切换检测与降级
│   │   └── multi_turn_rag.py # 多轮 RAG 主流程
│   ├── agent/                # Agent 工具化
│   │   ├── tools.py          # search_pdf 工具
│   │   └── agent.py          # AgentExecutor 入口
│   ├── chunking/             # 切分策略（fixed / recursive / regex / markdown）
│   ├── ingestion/            # PDF 解析、文本清洗、建库
│   ├── retrieval/            # 向量检索、BM25、混合检索、条件 Rerank
│   │   └── conditional_reranker.py
│   ├── routing/              # 规则路由分类器与路由器
│   │   ├── query_classifier.py  # 四类路由规则
│   │   └── router.py            # 路由分发与链路执行
│   ├── generation/           # 生成层增强
│   │   ├── citation.py          # 引用溯源
│   │   ├── conflict_detector.py # 冲突检测
│   │   └── refusal.py           # 自适应拒答
│   └── evaluation/           # 切分评估、检索评估、RAGAS 评估
├── experiments/              # 每日实验记录与评估结果
│   ├── week2/                # 第 2 周（基础 RAG 全链路）
│   │   ├── day1_log.md
│   │   ├── day2_loader_compare.md
│   │   ├── day3_chunking_matrix.md
│   │   ├── day4_chunking_eval.csv
│   │   ├── day5_retrieval_compare.md
│   │   ├── day6_testset.json
│   │   ├── day6_rag_outputs.csv
│   │   ├── day6_ragas_results.csv
│   │   ├── day6_test_record.md
│   │   ├── run_day6_ragas.py
│   │   └── final_report.md
│   ├── week3/                # 第 3 周（自适应 RAG + 评估 + 交付）
│   │   ├── week3_day1_log.md               # 路由分类器
│   │   ├── week3_day2_routing_log.md       # 路由链路测试
│   │   ├── week3_day3_rerank_*.csv/md      # 条件 Rerank 对比
│   │   ├── week3_day4_citation_conflict.md # 引用与冲突检测
│   │   ├── week3_day5_refusal_*.csv/md     # 自适应拒答评估
│   │   ├── week3_day6_layer_attribution*.csv # 分层归因表
│   │   ├── week3_day6_ragas_results.csv    # 三方案 RAGAS 结果
│   │   ├── week3_day6_comparison.md        # 三种方案对比报告
│   │   └── week3_final_report.md           # 最终实验报告（≥2500字）
│   └── week4/                # 第 4 周（多轮对话与失效归因）
│       ├── week4_day1_log.md                    # 状态对象与多轮测试集
│       ├── week4_day2_state_tracking.md         # 状态追踪与 Token 预算
│       ├── week4_day3_rewrite_eval.csv          # Query 改写质量评估
│       ├── week4_day4_ambiguity.md              # 歧义检测与主动澄清
│       ├── week4_day5_e2e_log.md                # 多轮 RAG 端到端测试
│       ├── week4_day6_layer_attribution.csv     # 多轮分层归因明细表
│       ├── week4_day6_comparison.md             # 四方案对比报告
│       ├── week4_day6_ragas_results_v2.csv      # 多轮 RAGAS 评测结果
│       └── week4_final_report.md                # Week 4 最终实验报告（≥2500字）
├── data/
│   ├── raw/                  # 原始 PDF
│   ├── day2_cleaned/         # 清洗后文本
│   ├── golden_qa/            # 黄金测试集
│   │   ├── qa.json           # 20 条基础 Golden QA
│   │   ├── qa_extended.json  # 25 条扩展 QA（含 5 条超范围题）
│   │   └── multi_turn_test.json # 20 条多轮对话测试集
│   ├── kb/                   # 知识库文档
│   └── uploaded_pdfs/        # API 上传的 PDF
├── chroma_db/                # ChromaDB 持久化目录（.gitignore）
├── Dockerfile
├── .dockerignore
├── requirements.in           # 顶层依赖（供 uv 解析）
├── requirements.txt          # 锁定版本（由 uv 生成）
└── README.md
```

## 数据与测试集说明

| 数据/测试集 | 数量与内容 | 路径 | 用途 |
|---|---|---|---|
| 基础 Golden QA | 20 条：easy 8 / medium 8 / hard 4，每条含 question / ground_truth / source_context / difficulty | `data/golden_qa/qa.json` | 切分、检索、端到端评估 |
| 扩展 QA | 25 条：20 条正常 + 5 条超范围题（ground_truth 标注“应拒答”） | `data/golden_qa/qa_extended.json` | 拒答评估、路由分类评估 |
| 多轮测试集 | 20 条，覆盖指代消解、话题切换、追问、省略补全、话题回退等 ≥5 种场景，每条 ≥3 轮 | `data/golden_qa/multi_turn_test.json` | Week 4 状态追踪、Rewrite、歧义、多轮 RAG 评估 |
| 矛盾测试用例 | 5 条多来源矛盾用例 | `experiments/week3/week3_day4_citation_conflict.md` | 冲突检测召回率评估 |
| 歧义测试用例 | 5 条指代不明 / 省略主语 / 多实体场景 | `experiments/week4/week4_day4_ambiguity.md` | 歧义检测与主动澄清评估 |
| 端到端测试集 | 10 个端到端问题 | `experiments/week2/day6_testset.json` | RAGAS 三指标与人工错误分析 |

## 快速开始

### 方式一：本地运行

```bash
# 1. 克隆项目
git clone https://github.com/kiyy0520/chinese-pdf-rag-benchmark.git
cd chinese-pdf-rag-benchmark

# 2. 创建虚拟环境
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 API Key
# 在项目根目录新建 .env 文件，内容：
# DEEPSEEK_API_KEY=sk-你的真实key
# 或者使用 Qwen：
# QWEN_API_KEY=sk-你的真实key

# 5. 启动服务
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

访问 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 查看 Swagger 文档。

### 方式二：Docker 运行

```bash
# 1. 构建镜像（容器内自动用 uv 解析 Python 3.10 兼容依赖）
docker build -t rag-benchmark:v4 .

# 2. 运行容器（挂载模型缓存和向量库，避免重复下载）
docker run -p 8000:8000 --env-file .env \
  -v C:\Users\<你的用户名>\.cache\huggingface:/root/.cache/huggingface \
  -v F:\Projects\chinese-pdf-rag-benchmark\chroma_db:/app/chroma_db \
  rag-benchmark:v4
```

访问 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)。

> **说明**：`requirements.in` 只写顶层依赖，Docker 构建时用 `uv pip compile` 在容器内基于 Python 3.10 解析并锁定版本，避免本地 Python 版本不一致导致的依赖冲突。Windows 路径请按本机实际路径替换；macOS / Linux 使用 `/home/...` 或 `~/...` 挂载。

## API 接口

### GET /

健康检查。

响应：

```json
{"status": "ok", "message": "RAG API is running"}
```

### POST /upload

上传 PDF，后端自动完成切分、向量化、入库。

请求：`multipart/form-data`，字段 `file`。

响应：

```json
{
  "message": "上传并入库成功",
  "filename": "sjtju_rl_lecture.pdf",
  "chunks_added": 128
}
```

### POST /ask

提问，返回答案、引用来源、路由轨迹及改写后的查询，支持传入对话历史实现多轮对话。

请求：

```json
{
  "question": "什么是马尔可夫决策过程？",
  "history": []
}
```

多轮示例：

```json
{
  "question": "那它的五元组是什么？",
  "history": [
    {"role": "user", "content": "什么是马尔可夫决策过程？"},
    {"role": "assistant", "content": "马尔可夫决策过程是..."}
  ]
}
```

响应：

```json
{
  "answer": "马尔可夫决策过程是...",
  "sources": [{"source": "xxx.pdf", "chunk_id": 3}],
  "contexts": ["检索到的原文片段..."],
  "route_taken": "semantic",
  "rewritten_query": "马尔可夫决策过程"
}
```

> `history` 元素以 `role/content` 表示对话历史；具体请求模型以 `src/api/main.py` 中的 Pydantic 定义为准。

## 路由决策表

系统根据 Query 特征自动选择检索路径：

| 路由类型 | 触发条件 | 检索方式 | 典型 Query |
|---|---|---|---|
| **skip** | 命中闲聊/超范围模式 | 不检索，直接回复 | "你好"、"今天天气怎么样" |
| **exact** | 命中术语/公式/符号 | BM25（关键词精确匹配） | "MDP 五元组是什么"、"γ 代表什么" |
| **semantic** | 命中"什么是"、"为什么"等 | 向量检索（语义相似） | "什么是马尔可夫过程"、"为什么要有折扣因子" |
| **hybrid** | 命中"区别"、"步骤"、"对比" | BM25 + 向量 RRF 融合 | "价值迭代和策略迭代的区别" |

分类器基于正则规则实现，无需训练即可解释。Week 3 验收目标：20 条 Golden QA 分类准确率 ≥ 85%，4 条路由全部跑通，平均路由延迟 < 50ms；实测记录见 `experiments/week3/week3_day2_routing_log.md`。

## Week 3 专项设计：自适应检索与质量保障

### 1. Query 分类器
- 实现方式：规则（正则）分类器，零训练、可解释、迭代快。
- 输出：`skip / exact / semantic / hybrid` 四类路由。
- 验收目标：分类准确率 ≥ 85%；错误分类 case 全部记录并分析原因。
- 失效边界：规则泛化能力弱于模型，复杂自然语言问法可能误分类；通过路由日志持续补规则。

### 2. 条件 Rerank
- 触发条件：当向量检索结果的 Top-K 相似度方差或 BM25/向量得分一致性低于阈值时触发 Reranker。
- 对比实验：始终 Rerank vs 条件 Rerank，比较 Recall@5、MRR、延迟。
- 验收目标：条件触发比例可统计；延迟增加 < 200ms；记录“Reranker 反而降低效果”的 case ≥ 2 条。
- 失效边界：原始检索已正确排序时，Reranker 可能引入新偏差；短 Query 或专有名词密集时效果下降。

### 3. 引用溯源
- 每条回答附带来源：`source` + `chunk_id` + 相似度分数。
- 支持回溯原文片段，降低幻觉排查成本。

### 4. 冲突检测
- 规则：多来源 chunk 语义相似度低于阈值或关键数值/结论不一致时，标记“来源 A 与来源 B 存在矛盾”。
- 测试：5 条矛盾用例。
- 验收目标：冲突检测召回率 ≥ 80%，并记录误报率。

### 5. 自适应拒答
- 检索置信度：Top-K 相似度均值 + 方差。
- 生成置信度：LLM 回答与检索内容的语义一致性。
- 双重拒答：检索置信度低 **或** 生成置信度低 → 拒答。
- 拒答模板固定：“根据现有知识库，我无法回答这个问题。”
- 验收目标：拒答准确率 ≥ 90%，误拒率 < 10%；阈值调优记录 3 组对比数据，见 `experiments/week3/week3_day5_refusal_eval.csv`。

### 6. 分层归因
- 四层：Query 理解 / 检索 / 重排 / 生成。
- 每条 badcase 标注错误层级，计算每层错误率。
- 根因定位：系统核心瓶颈是**检索召回率**而非生成质量；一旦检索到正确文档，系统表现极度稳定。

## Week 4 专项设计：多轮对话与状态管理

### 1. 结构化对话状态对象
系统维护结构化状态对象，核心字段如下：

| 字段 | 含义 | 更新条件 |
|---|---|---|
| `intent` | 当前用户意图 | 每轮根据 Query 分类与改写结果更新 |
| `hard_constraints` | 硬约束（如“不辣”、指定公式符号） | 用户明确提出约束时更新 |
| `search_query` | 剥离代词后的精简检索词 | 每轮由 Query Rewriter 生成 |
| `topic_switch_flag` | 话题切换标记 | 改写后 Query 与当前 Query 实体重叠度低于阈值时置位 |
| `clarification_needed` | 歧义澄清标记 | 歧义检测命中时置位 |
| `current_entities` | 当前讨论实体（可选扩展） | 实体抽取后更新 |
| `confirmed_info` | 已确认信息（可选扩展） | 用户确认或系统回答被采纳后更新 |
| `pending_questions` | 待澄清问题（可选扩展） | 触发澄清时写入，澄清完成后清空 |
| `history_summary` | 历史轮次摘要（可选扩展） | 超出 Token 预算时压缩生成 |

> 最小可用状态对象含 ≥5 个字段；扩展字段用于状态追踪、澄清与摘要压缩。

### 2. 状态转换规则
- **更新实体**：每轮从 Query 与回答中抽取实体，合并到 `current_entities`。
- **触发话题切换**：改写后 Query 与当前 Query 的实体重叠度低于阈值时，置位 `topic_switch_flag`，并降级为只用当前 Query 检索。
- **重置状态**：检测到全新话题且无指代依赖时，清空 `pending_questions` 与旧实体，保留必要硬约束。
- **触发澄清**：指代词存在多个候选实体或省略主语导致无法确定检索对象时，置位 `clarification_needed`。

### 3. 话题切换与降级策略
- 计算改写后 Query 与当前 Query 的实体重叠度。
- 重叠度低于阈值时，**宁可少补，不可乱补**：降级为只用当前 Query 检索，避免历史污染当前检索。
- 阈值通过多轮实验跑出，记录于 `experiments/week4/week4_day1_log.md`。

### 4. Query Rewriting 与质量评估
- 用 LLM 将依赖上下文的问句重写为独立问句。
- 对 20 条多轮测试集标注“重写后应包含的实体”和“不应包含的实体”。
- 计算精确率与召回率：当前记录 Precision 82%、Recall 88%。
- 失效 case ≥ 5 条，按类型分析：

| 失效类型 | 表现 | 根因 | 降级策略 |
|---|---|---|---|
| 过度改写 | 引入上下文中不相关实体 | LLM 过度推断 | 实体重叠度检查 |
| 改写不足 | 遗漏关键指代实体 | 指代链过长 | 回溯多轮历史 |
| 话题误判 | 用户已切换话题但改写仍用旧上下文 | 缺少话题切换检测 | 话题切换阈值触发 |

### 5. 歧义检测与主动澄清
- 规则检测：指代词是否存在多个候选实体。
- LLM 判断：规则不确定时，用 LLM 判断指代是否歧义。
- 澄清话术：“您说的‘它’是指 A 还是 B？”
- 验收目标：歧义检测召回率 ≥ 80%，误报率 < 15%；5 条歧义测试用例全部有明确澄清回复，澄清后追问轮次 ≤ 1 轮。

### 6. 多轮 RAG 链路
- 改写后的 Query 走 Week 2/3 的混合检索 + 条件 Reranker。
- 状态对象驱动检索：检索 Query 只继承“当前意图 + 硬约束”，不携带全部历史。
- 多轮场景下的自适应拒答：结合检索置信度 + 生成置信度 + 状态对象。
- 验收目标：3 轮追问全部正确回答；检索延迟 < 1s/轮；多轮场景拒答准确率 ≥ 85%；记录有/无 Query Rewriting 的 Recall@5 差异。

### 7. 多轮失效分层归因
- 对多轮 badcase 按 Query 理解 / 检索 / 重排 / 生成四层标注。
- 输出每层错误率，定位“是改写丢了实体，还是检索没召回，还是生成总结偏差”。

## 评估指标定义

| 指标 | 定义 |
|---|---|
| Recall@5 | Top-5 检索结果中是否包含正确来源 chunk |
| MRR | 正确来源 chunk 在检索结果中的平均倒数排名 |
| Faithfulness | 答案对检索内容的忠实度，越高越少幻觉 |
| Context Precision | 检索片段精确率，越高噪声越少 |
| Context Recall | 检索召回率，越高漏召越少 |
| 分层错误率 | 某层错误数 / 总 badcase 数，按 Query 理解 / 检索 / 重排 / 生成统计 |
| Rewrite Precision | 改写后 Query 中“应包含实体”的命中比例，衡量是否引入错误实体 |
| Rewrite Recall | 应包含实体被改写 Query 覆盖的比例，衡量是否遗漏关键实体 |

## 评估结果汇总

### 1. 基础三方案对比（10 个端到端问题，RAGAS 评估）

| 方案 | Faithfulness | Context Precision | Context Recall |
|---|---|---|---|
| 纯 LLM（无检索） | 0.0000 | 0.0000 | 0.0000 |
| 基础 RAG（向量检索） | 0.9000 | 0.9000 | 0.9000 |
| **自适应 RAG（路由 + 条件 Rerank）** | **1.0000** | 0.8656 | **1.0000** |

**结论**：
1. 纯 LLM 无检索辅助，在专业领域全面溃败。
2. 自适应 RAG 在 **Faithfulness** 和 **Context Recall** 两项优于基础 RAG，满足“至少在 1 个指标上优于基础 RAG”的验证条件。
3. 自适应 RAG 的 Context Precision 略低，说明混合检索引入了少量噪音，后续可通过调整 RRF 权重或加 Reranker 优化。

### 2. 多轮四方案对比（RAGAS 指标）

| 方案 | Faithfulness | Context Precision | Context Recall |
|---|---|---|---|
| 纯 LLM | 0.0152 | 0.8333 | 1.0 |
| 基础 RAG | 1.0 | 0.8333 | 1.0 |
| 多轮 RAG（无状态） | 1.0 | 0.8333 | 1.0 |
| 多轮 RAG（有状态） | 1.0 | 0.8333 | 1.0 |

> 注：控制变量 Mock 下三种 RAG 打平，证明有状态架构不降智；真实指代场景下无状态 Recall 将断崖式下跌。有/无状态管理的真实对比记录见 `experiments/week4/week4_day6_comparison.md`。

### 3. 路由轨迹分析（自适应 RAG）

| 路由 | Faithfulness | Context Precision | Context Recall | 触发次数 |
|---|---|---|---|---|
| exact | 1.0000 | 0.8222 | 1.0000 | 4 |
| hybrid | 0.8889 | 0.7611 | 1.0000 | 3 |
| semantic | 1.0000 | 1.0000 | 1.0000 | 3 |

**结论**：`semantic` 路由精度最高（1.0），`hybrid` 路由精度最低（0.76），后续应优先优化 `hybrid` 的 RRF 权重。

### 4. 基础分层归因表

| 错误层级 | 错误数量 | 错误率 | 归因说明 |
|---|---|---|---|
| Query 理解 | 10 | 33.3% | 纯 LLM 无检索能力，全部答错归为此层 |
| 检索 | 1 | 3.3% | 基础 RAG 有 1 道题完全未召回正确文档 |
| 重排 | 0 | 0.0% | 检索到的资料均被正确排序 |
| 生成 | 0 | 0.0% | 资料给对后，大模型均能忠实作答 |

**根因定位**：系统核心瓶颈是**检索召回率**而非生成质量。一旦检索到正确文档，系统表现极度稳定。

### 5. 多轮失效分层归因表

| 错误层级 | 错误数量 | 错误率 | 归因说明 |
|---|---|---|---|
| Query 理解 | 0 | 0.00% | 改写逻辑正常 |
| 检索 | 1 | 33.33% | 知识库缺失资料导致无法召回 |
| 重排 | 0 | 0.00% | 资料均被正确排序 |
| 生成 | 2 | 66.67% | 资料给对后，模型总结产生偏差 |

### 6. 其他专项评估

| 评估项 | 结果 / 验收目标 | 记录文件 |
|---|---|---|
| 切分策略对比 | 9 组配置，最优 recall@5 = 0.3 | `experiments/week2/day4_chunking_eval.csv` |
| 检索方案对比 | 向量 / BM25 / 混合三路对比，MRR/Recall@5 与 badcase ≥5 条 | `experiments/week2/day5_retrieval_compare.md` |
| 条件 Rerank | 与 always Rerank 对比，延迟增加 < 200ms；失效 case ≥2 | `experiments/week3/week3_day3_rerank_compare.csv` |
| 自适应拒答 | 拒答准确率 ≥ 90%，误拒率 < 10%；阈值调优 3 组 | `experiments/week3/week3_day5_refusal_eval.csv` |
| 冲突检测 | 5 条矛盾用例，召回率 ≥ 80%，记录误报率 | `experiments/week3/week3_day4_citation_conflict.md` |
| Query 分类器 | 分类准确率 ≥ 85%，4 条路由跑通，路由延迟 < 50ms | `experiments/week3/week3_day2_routing_log.md` |
| Query Rewriting | Precision 82%，Recall 88%，失效 case ≥5 条 | `experiments/week4/week4_day3_rewrite_eval.csv` |
| 歧义检测与澄清 | 召回率 ≥ 80%，误报率 < 15%，5 条用例有澄清回复 | `experiments/week4/week4_day4_ambiguity.md` |
| 状态追踪 | 准确率 ≥ 80%，Token 节省率 ≥ 30%，摘要实体保留率 ≥ 90% | `experiments/week4/week4_day2_state_tracking.md` |
| 多轮 RAG 端到端 | 3 轮追问正确；检索延迟 < 1s/轮；多轮拒答准确率 ≥ 85% | `experiments/week4/week4_day5_e2e_log.md` |

> 表中“验收目标”类指标以对应实验日志实测为准；README 只列目标与已有实测数字，避免把目标误写成结果。

## 可复现实验与验收清单

### 复现命令

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 运行 Week2 RAGAS 评估脚本（按脚本内路径与参数执行）
python experiments/week2/run_day6_ragas.py
```

### Week 2 验收清单

- [ ] 可运行 RAG：上传 PDF → 切分 → 向量化 → 提问 → 生成回答
- [ ] 20 条 Golden QA（easy 8 / medium 8 / hard 4）
- [ ] 9 组切分对比实验，输出 Recall@5 / MRR
- [ ] 3 种检索方案对比：纯向量 / BM25 / RRF 混合
- [ ] RAGAS 三指标结果
- [ ] ≥5 条 badcase 归因
- [ ] Dockerfile + README + GitHub 仓库
- [ ] 能回答 8 个基础 RAG 面试追问

### Week 3 验收清单

- [ ] 自适应 RAG：上传 PDF → Query 分类 → 路由 → 检索 → 条件 Rerank → 生成 → 引用溯源 → 拒答
- [ ] Query 分类器，准确率 ≥ 85%
- [ ] 4 条路由链路全部跑通，平均路由延迟 < 50ms
- [ ] 条件 Rerank vs 始终 Rerank 对比数据，延迟增加 < 200ms
- [ ] 引用溯源：每条回答带来源
- [ ] 冲突检测：5 条矛盾用例，召回率 ≥ 80%
- [ ] 自适应拒答：拒答准确率 ≥ 90%，误拒率 < 10%
- [ ] 分层归因表：Query 理解 / 检索 / 重排 / 生成四层错误率
- [ ] 对比实验：纯 LLM vs 基础 RAG vs 自适应 RAG
- [ ] RAGAS 三指标结果
- [ ] Dockerfile + README + GitHub
- [ ] 能回答 8 个自适应 RAG 面试追问

### Week 4 验收清单

- [ ] 多轮对话 RAG：支持 ≥3 轮追问，能处理指代消解
- [ ] 结构化对话状态对象：≥5 字段，含话题切换标记与澄清标记
- [ ] 20 条多轮对话测试集：覆盖 ≥5 种对话场景
- [ ] 状态追踪准确率 ≥ 80%，Token 节省率 ≥ 30%，摘要实体保留率 ≥ 90%
- [ ] Query Rewriting 质量评估：精确率 ≥ 80%，召回率 ≥ 85%，失效 case ≥5 条
- [ ] 歧义检测与主动澄清：召回率 ≥ 80%，误报率 < 15%
- [ ] 多轮失效分层归因表：Query 理解 / 检索 / 重排 / 生成四层错误率
- [ ] 对比实验：纯 LLM vs 基础 RAG vs 多轮 RAG（无状态） vs 多轮 RAG（有状态）
- [ ] RAGAS 三指标结果
- [ ] Dockerfile + README + GitHub
- [ ] 能回答 8 个多轮对话 RAG 面试追问

## 面试高频追问速答

### Week 2：基础 RAG 专项

| # | 问题 | 核心答案 |
|---|---|---|
| 1 | chunk_size 为什么选这个？ | 通过 9 组 chunk_size / overlap 矩阵对比 Recall@5 与 MRR，选最优配置；不是拍脑袋，而是实验驱动。 |
| 2 | 纯向量够用吗？ | 不够。纯向量对术语、公式、符号精确匹配弱；BM25 补关键词，混合检索更稳。 |
| 3 | BM25 和向量检索各自适合什么？ | BM25 适合术语/公式/符号精确匹配；向量适合“什么是”“为什么”等语义相似；混合用 RRF 融合。 |
| 4 | RRF 为什么有效？ | RRF 只依赖排名而非分数尺度，能把 BM25 与向量的候选排名稳健融合，降低单路噪音。 |
| 5 | 你怎么评估检索质量？ | 用 Golden QA 跑 Recall@5、MRR；再做 badcase 归因：漏召、误召、语义歧义、长距离依赖。 |
| 6 | 检索失败怎么排查？ | 看分层归因：Query 理解是否丢实体；检索是否未召回；重排是否排错；生成是否总结偏差。 |
| 7 | RAGAS 三个指标分别代表什么？ | Faithfulness 忠实度；Context Precision 检索精确率；Context Recall 检索召回率。 |
| 8 | 这个项目相比普通 RAG 的差异化在哪？ | 自适应路由 + 条件 Rerank + 拒答 + 冲突检测 + 分层归因，全链路可量化。 |

### Week 3：自适应 RAG 专项

| # | 问题 | 核心答案 |
|---|---|---|
| 1 | 有几种检索路径？怎么决定走哪条？ | 4 条：skip / exact / semantic / hybrid。基于正则规则分类器，无需训练即可解释。 |
| 2 | 什么情况下拒绝回答？ | 命中超范围模式，或检索置信度低于阈值（mean_sim < 0.45 且 var_sim 小），或生成置信度低。 |
| 3 | Reranker 什么时候反而降低效果？ | 原始检索已正确排序时可能引入新偏差；短 Query 或专有名词密集时效果下降。 |
| 4 | 怎么区分是 Query 理解问题还是检索器问题？ | 看分层归因：分类/改写错 → Query 理解层；分类对但 recall 低 → 检索层。 |
| 5 | 多结果矛盾怎么处理？ | 用 conflict_detector 检测两个 chunk 语义相似度低于阈值，触发“存在矛盾”提示。 |
| 6 | 分类器用什么实现？规则还是模型？ | 规则（正则）。零训练、可解释、迭代快；缺点是泛化弱于模型。 |
| 7 | 条件 Rerank 的触发条件是什么？ | 当向量检索结果的方差或一致性低于阈值时触发，避免每次都跑 Reranker。 |
| 8 | 相比普通 RAG 差异化在哪？ | 自适应路由 + 条件 Rerank + 拒答 + 冲突检测 + 分层归因，全链路可量化交付。 |

### Week 4：多轮对话 RAG 专项

| # | 问题 | 核心答案 |
|---|---|---|
| 1 | 多轮 RAG 为什么越聊越蠢？ | 历史噪音累积 + 指代不明导致检索漂移。解法：结构化状态追踪 + Query 改写 + 话题切换降级。 |
| 2 | 用户从手机切到电脑，query 用什么？ | 依然用改写后的包含核心实体的检索 Query；设备切换不影响对话状态继承，检索只走“当前意图 + 硬约束”。 |
| 3 | Query Rewriting 在什么情况下失效？ | 历史过长截断、指代极度模糊（如“就那个东西”）、话题突然跳跃；用实体重叠度检查与降级策略兜底。 |
| 4 | 指代不明时怎么处理？ | 触发 clarification 模块，主动向用户提问澄清（如“您说的是张三还是李四？”），不瞎猜。 |
| 5 | 多轮对话中检索失败，怎么定位是 query 问题还是检索问题？ | 看多轮分层归因表：改写丢实体 → Query 理解层；Query 正确但没搜到 → 检索层。 |
| 6 | 对话状态对象包含哪些字段？ | intent, hard_constraints, search_query, topic_switch_flag, clarification_needed；可扩展 current_entities、confirmed_info、pending_questions、history_summary。 |
| 7 | 话题切换检测的阈值怎么定？ | 基于改写后 Query 与当前 Query 的实体重叠度，低于阈值视为切换并降级为只用当前 Query 检索；阈值由多轮实验跑出。 |
| 8 | 相比普通多轮 RAG 差异化在哪？ | 自适应路由 + 条件 Rerank + 状态追踪 + 主动澄清 + 分层归因，全链路可量化交付。 |

## 技术决策与失效边界

| 决策点 | 选择 | 理由 | 失效边界 |
|---|---|---|---|
| Query 分类器 | 规则（正则） | 零训练、可解释、迭代快 | 复杂自然语言问法泛化弱，需持续补规则 |
| 条件 Rerank | 仅低一致性触发 | 平衡延迟与效果 | 原始排序已正确时可能引入偏差；短 Query/专有名词密集时下降 |
| 对话状态 | 结构化对象而非消息列表 | 分离“当前意图”与“历史上下文”，避免历史污染检索 | 状态更新规则覆盖不全时会漏实体；需日志持续修正 |
| 话题切换 | 实体重叠度低于阈值则降级 | 宁可少补，不可乱补 | 阈值过高会误降级，过低会放过错误改写 |
| 歧义处理 | 主动澄清而非猜测 | 降低瞎答风险 | 澄清过多影响体验，需控制误报率 <15% |
| 拒答 | 检索置信度 + 生成置信度双门控 | 降低超范围瞎答 | 阈值过高误拒正常题；需 3 组调参平衡 |
| 依赖管理 | requirements.in + uv 容器内解析 | 避免本地 Python 版本冲突 | 首次构建较慢；需挂载 HuggingFace 缓存 |

## 项目亮点

- **可复现**：Docker 打包 + `requirements.in` + `uv` 锁定版本 + 完整实验记录。
- **可评估**：RAGAS 三指标 + 分层归因表 + 路由轨迹分析 + 拒答/冲突/多轮/Rewrite/歧义专项评估。
- **可扩展**：RAG 已封装为 Agent 工具，未来可挂载多工具协同。
- **工程化**：分层目录（api / conversation / chunking / ingestion / retrieval / routing / generation / evaluation），前后 4 周迭代。
- **数据驱动**：所有结论带数字，例如“自适应 RAG 在 Faithfulness 上从 0.9 提升到 1.0”“Rewrite Precision 82% / Recall 88%”。
- **多轮增强**：结构化状态追踪 + Query 改写 + 主动澄清 + 话题切换降级，解决多轮对话越聊越蠢的行业痛点。
- **失效可诊断**：分层归因把 badcase 定位到 Query 理解 / 检索 / 重排 / 生成，避免“只调 Prompt 不治根因”。

## 常见问题

**Q：上传 PDF 后提问，回答“根据提供的资料无法回答”？**

A：说明该问题在已入库的 PDF 中找不到相关内容，或检索置信度过低触发了拒答。可先上传包含答案的 PDF，或检查 `chroma_db/day4_chunking_eval` 集合是否有数据。

**Q：首次运行卡在模型下载？**

A：`BAAI/bge-large-zh-v1.5` 约 1.3GB，首次会从 HuggingFace 下载。已设置国内镜像 [https://hf-mirror.com](https://hf-mirror.com)，若仍慢可手动下载到 `~/.cache/huggingface/`，然后用 Docker 挂载进去。

**Q：Docker build 很慢？**

A：`torch` 和 CUDA 依赖较大，首次 build 约 5-15 分钟。已用 `uv` 在容器内基于 Python 3.10 自动解析依赖，避免版本冲突。后续 build 命中缓存会快很多。

**Q：本地 Python 是 3.14，容器是 3.10，会冲突吗？**

A：不会。Docker 的价值就是环境隔离，本机 Python 版本与容器无关。`requirements.in` 只写顶层依赖，容器内用 `uv pip compile` 基于 Python 3.10 解析出兼容版本。

**Q：多轮对话历史应该传多少轮？**

A：不需要无限传。系统用滑动窗口 + Token 预算截断 + 摘要压缩；检索只使用结构化状态中的 `search_query` 与硬约束，避免历史噪音污染检索。具体窗口与预算见 `src/conversation/memory.py`。

## 许可

MIT License

## 作者

@kiyy0520