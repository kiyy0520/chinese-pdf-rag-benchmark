# chinese-pdf-rag-benchmark

> 一个面向中文强化学习教材的 PDF 知识库问答系统。支持上传 PDF、自适应路由检索、条件 Rerank、引用溯源、冲突检测、拒答、LLM 生成、FastAPI 服务化、RAGAS 自动评估。

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)

---

## 项目简介

本项目以中文强化学习课件（马尔可夫决策过程、强化学习基础）为语料，构建了一套完整的**自适应 RAG 流水线**，覆盖：

**PDF 解析 → 文本切分 → 向量化 → 混合检索 → 条件 Rerank → 引用溯源 → 冲突检测 → 拒答 → LLM 生成 → 分层归因 → RAGAS 评估 → API 服务化 → Docker 交付**

同时集成 **LangChain Agent** 与 **规则路由分类器**，让系统根据 Query 类型自主选择检索路径，实现"按需检索"，并通过分层归因表量化系统瓶颈。

## 核心特性

- **自适应路由**：4 条检索路径（skip / exact / semantic / hybrid），基于规则分类器自动分发
- **条件 Rerank**：仅在检索一致性低时触发 Rerank，平衡性能与延迟
- **引用溯源**：每条回答附带来源（source + chunk_id），可回溯原文
- **冲突检测**：多文档出现矛盾时主动提示用户
- **自适应拒答**：超出知识库范围或检索置信度低时拒绝作答
- **端到端 RAG 链路**：PDF → 切分 → 向量化 → 检索 → 生成，一步到位
- **FastAPI 服务化**：`/upload` 上传 PDF，`/ask` 提问，自动生成 Swagger 文档
- **RAGAS 自动评估**：Faithfulness / Context Precision / Context Recall 三指标
- **分层归因**：Query 理解 / 检索 / 重排 / 生成 四层错误率量化
- **Docker 一键部署**：`docker build` + `docker run` 即可在任何机器上复现
- **可复现实验记录**：切分对比、检索对比、重排对比、拒答评估、分层归因、方案对比

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
| 评估 | RAGAS 0.4.3 |
| 依赖管理 | uv（容器内基于 Python 3.10 自动解析） |
| 容器化 | Docker（python:3.10-slim） |

## 目录结构

```
chinese-pdf-rag-benchmark/
├── src/
│   ├── api/                  # FastAPI 服务与 RAG 链路
│   │   ├── main.py           # 上传、提问接口
│   │   └── rag_chain.py      # 检索 → Prompt → LLM 生成
│   ├── agent/                # Agent 工具化
│   │   ├── tools.py          # search_pdf 工具
│   │   └── agent.py          # AgentExecutor 入口
│   ├── chunking/             # 切分策略（fixed / recursive / regex / markdown）
│   ├── ingestion/            # PDF 解析、文本清洗、建库
│   ├── retrieval/            # 向量检索、BM25、混合检索、条件 Rerank
│   ├── routing/              # 规则路由分类器与路由器
│   │   ├── query_classifier.py  # 四类路由规则
│   │   └── router.py            # 路由分发与链路执行
│   ├── generation/           # 生成层增强
│   │   ├── citation.py          # 引用溯源
│   │   ├── conflict_detector.py # 冲突检测
│   │   └── refusal.py           # 自适应拒答
│   └── evaluation/          # 切分评估、检索评估、RAGAS 评估
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
│   └── week3/               # 第 3 周（自适应 RAG + 评估 + 交付）
│       ├── week3_day1_log.md               # 路由分类器
│       ├── week3_day2_routing_log.md       # 路由链路测试
│       ├── week3_day3_rerank_*.csv/md      # 条件 Rerank 对比
│       ├── week3_day4_citation_conflict.md  # 引用与冲突检测
│       ├── week3_day5_refusal_*.csv/md      # 自适应拒答评估
│       ├── week3_day6_layer_attribution*.csv # 分层归因表
│       ├── week3_day6_ragas_results.csv   # 三方案 RAGAS 结果
│       ├── week3_day6_comparison.md        # 三种方案对比报告
│       └── week3_final_report.md           # 最终实验报告（≥2500字）
├── data/
│   ├── raw/                  # 原始 PDF
│   ├── day2_cleaned/         # 清洗后文本
│   ├── golden_qa/            # 黄金测试集
│   ├── kb/                   # 知识库文档
│   └── uploaded_pdfs/        # API 上传的 PDF
├── chroma_db/                # ChromaDB 持久化目录（.gitignore）
├── Dockerfile
├── .dockerignore
├── requirements.in           # 顶层依赖（供 uv 解析）
├── requirements.txt          # 锁定版本（由 uv 生成）
└── README.md
```

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

访问 http://127.0.0.1:8000/docs 查看 Swagger 文档。

### 方式二：Docker 运行

```bash
# 1. 构建镜像（容器内自动用 uv 解析 Python 3.10 兼容依赖）
docker build -t rag-benchmark:week3 .

# 2. 运行容器（挂载模型缓存和向量库，避免重复下载）
docker run -p 8000:8000 --env-file .env \
  -v C:\Users\<你的用户名>\.cache\huggingface:/root/.cache/huggingface \
  -v F:\Projects\chinese-pdf-rag-benchmark\chroma_db:/app/chroma_db \
  rag-benchmark:week3
```

访问 http://127.0.0.1:8000/docs。

> **说明**：`requirements.in` 只写顶层依赖，Docker 构建时用 `uv pip compile` 在容器内基于 Python 3.10 解析并锁定版本，避免本地 Python 版本不一致导致的依赖冲突。

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

提问，返回答案、引用来源和路由轨迹。

请求：

```json
{"question": "什么是马尔可夫决策过程？"}
```

响应：

```json
{
  "answer": "马尔可夫决策过程是...",
  "sources": [{"source": "xxx.pdf", "chunk_id": 3}],
  "contexts": ["检索到的原文片段..."],
  "route_taken": "semantic"
}
```

## 路由决策表

系统根据 Query 特征自动选择检索路径：

| 路由类型 | 触发条件 | 检索方式 | 典型 Query |
|---|---|---|---|
| **skip** | 命中闲聊/超范围模式 | 不检索，直接回复 | "你好"、"今天天气怎么样" |
| **exact** | 命中术语/公式/符号 | BM25（关键词精确匹配） | "MDP 五元组是什么"、"γ 代表什么" |
| **semantic** | 命中"什么是"、"为什么"等 | 向量检索（语义相似） | "什么是马尔可夫过程"、"为什么要有折扣因子" |
| **hybrid** | 命中"区别"、"步骤"、"对比" | BM25 + 向量 RRF 融合 | "价值迭代和策略迭代的区别" |

分类器基于正则规则实现，无需训练即可解释，准确率见 `week3_day2_routing_log.md`。

## 评估结果

### 三种方案对比（10 个端到端问题，RAGAS 评估）

| 方案 | Faithfulness | Context Precision | Context Recall |
|---|---|---|---|
| 纯 LLM（无检索） | 0.0000 | 0.0000 | 0.0000 |
| 基础 RAG（向量检索） | 0.9000 | 0.9000 | 0.9000 |
| **自适应 RAG（路由 + 条件 Rerank）** | **1.0000** | 0.8656 | **1.0000** |

**结论**：

1. 纯 LLM 无检索辅助，在专业领域全面溃败。
2. 自适应 RAG 在 **Faithfulness** 和 **Context Recall** 两项优于基础 RAG，满足"至少在 1 个指标上优于基础 RAG"的验证条件。
3. 自适应 RAG 的 Context Precision 略低，说明混合检索引入了少量噪音，后续可通过调整 RRF 权重或加 Reranker 优化。

### 分层归因表

| 错误层级 | 错误数量 | 错误率 | 归因说明 |
|---|---|---|---|
| Query 理解 | 10 | 33.3% | 纯 LLM 无检索能力，全部答错归为此层 |
| 检索 | 1 | 3.3% | 基础 RAG 有 1 道题完全未召回正确文档 |
| 重排 | 0 | 0.0% | 检索到的资料均被正确排序 |
| 生成 | 0 | 0.0% | 资料给对后，大模型均能忠实作答 |

**根因定位**：系统核心瓶颈是**检索召回率**而非生成质量。一旦检索到正确文档，系统表现极度稳定。

### 路由轨迹分析（自适应 RAG）

| 路由 | Faithfulness | Context Precision | Context Recall | 触发次数 |
|---|---|---|---|---|
| exact | 1.0000 | 0.8222 | 1.0000 | 4 |
| hybrid | 0.8889 | 0.7611 | 1.0000 | 3 |
| semantic | 1.0000 | 1.0000 | 1.0000 | 3 |

**结论**：`semantic` 路由精度最高（1.0），`hybrid` 路由精度最低（0.76），后续应优先优化 `hybrid` 的 RRF 权重。

### 其他评估

| 评估项 | 结果 | 文件 |
|---|---|---|
| 切分策略对比 | 9 组配置，最优 recall@5 = 0.3 | `week2/day4_chunking_eval.csv` |
| 检索方案对比 | 向量 / BM25 / 混合 三路对比 | `week2/day5_retrieval_compare.md` |
| 条件 Rerank | 与 always Rerank 对比，节省延迟 | `week3_day3_rerank_compare.csv` |
| 自适应拒答 | 准确率 ≥ 90%，误拒率 < 10% | `week3_day5_refusal_eval.csv` |
| 冲突检测 | 5 条矛盾用例，召回率 ≥ 80% | `week3_day4_citation_conflict.md` |

## 面试高频追问速答

| # | 问题 | 核心答案 |
|---|---|---|
| 1 | 有几种检索路径？怎么决定走哪条？ | 4 条：skip / exact / semantic / hybrid。基于正则规则分类器，无需训练即可解释。 |
| 2 | 什么情况下拒绝回答？ | 命中超范围模式，或检索置信度低于阈值（mean_sim < 0.45 且 var_sim 小）。 |
| 3 | Reranker 什么时候反而降低效果？ | 当原始检索已经正确排序时，Reranker 可能引入新偏差；短 Query 或专有名词密集时效果下降。 |
| 4 | 怎么区分是 Query 理解问题还是检索器问题？ | 看分层归因表：query 分类错 → Query 理解层；分类对但 recall 低 → 检索层。 |
| 5 | 多结果矛盾怎么处理？ | 用 conflict_detector 检测两个 chunk 的语义相似度低于阈值，触发"存在矛盾"提示。 |
| 6 | 分类器用什么实现？规则还是模型？ | 规则（正则）。优点是零训练、可解释、迭代快；缺点是泛化能力弱于模型。 |
| 7 | 条件 Rerank 的触发条件是什么？ | 当向量检索结果的方差或一致性低于阈值时触发，避免每次都跑 Reranker。 |
| 8 | 相比普通 RAG 差异化在哪？ | 自适应路由 + 条件 Rerank + 拒答 + 冲突检测 + 分层归因，全链路可量化。 |

## 项目亮点

- **可复现**：Docker 打包 + `requirements.in` + `uv` 锁定版本 + 完整实验记录
- **可评估**：RAGAS 三指标 + 分层归因表 + 路由轨迹分析 + 拒答/冲突专项评估
- **可扩展**：RAG 已封装为 Agent 工具，未来可挂载多工具协同
- **工程化**：分层目录（api / chunking / ingestion / retrieval / routing / generation / evaluation），前后 7 天迭代
- **数据驱动**：所有结论带数字，例如"自适应 RAG 在 Faithfulness 上从 0.9 提升到 1.0"

## 常见问题

**Q：上传 PDF 后提问，回答"根据提供的资料无法回答"？**

A：说明该问题在已入库的 PDF 中找不到相关内容，或检索置信度过低触发了拒答。可先上传包含答案的 PDF，或检查 `chroma_db/day4_chunking_eval` 集合是否有数据。

**Q：首次运行卡在模型下载？**

A：`BAAI/bge-large-zh-v1.5` 约 1.3GB，首次会从 HuggingFace 下载。已设置国内镜像 https://hf-mirror.com ，若仍慢可手动下载到 `~/.cache/huggingface/`，然后用 Docker 挂载进去。

**Q：Docker build 很慢？**

A：`torch` 和 CUDA 依赖较大，首次 build 约 5-15 分钟。已用 `uv` 在容器内基于 Python 3.10 自动解析依赖，避免版本冲突。后续 build 命中缓存会快很多。

**Q：本地 Python 是 3.14，容器是 3.10，会冲突吗？**

A：不会。Docker 的价值就是环境隔离，本机 Python 版本与容器无关。`requirements.in` 只写顶层依赖，容器内用 `uv pip compile` 基于 Python 3.10 解析出兼容版本。

## 许可

MIT License

## 作者

@kiyy0520