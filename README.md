# chinese-pdf-rag-benchmark

> 一个面向中文强化学习教材的 PDF 知识库问答系统。支持上传 PDF、语义检索、LLM 生成、FastAPI 服务化、RAGAS 自动评估。

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)

---

## 项目简介

本项目以两篇中文强化学习课件（马尔可夫决策过程、sjtju_rl_lecture）为语料，构建了一套完整的 RAG（检索增强生成）流水线，覆盖 **PDF 解析 → 文本切分 → 向量化 → 向量检索 → Prompt 拼装 → LLM 生成 → API 服务化 → RAGAS 评估 → 错误分析** 全链路。

同时集成 **LangChain Agent**，让 LLM 自主判断是否需要调用 PDF 检索工具，实现"按需检索"。

## 核心特性

- **端到端 RAG 链路**：PDF → 切分 → 向量化 → 检索 → 生成，一步到位
- **FastAPI 服务化**：`/upload` 上传 PDF，`/ask` 提问，自动生成 Swagger 文档
- **RAGAS 自动评估**：Faithfulness / Context Precision / Context Recall 三指标
- **Agent 工具化**：`search_pdf` 工具可被 LLM 自主调用
- **Docker 一键部署**：`docker build` + `docker run` 即可在任何机器上复现
- **可复现实验记录**：9 组切分对比、多种检索方案对比、10 条端到端测试记录

## 技术栈

| 类别 | 选型 |
|---|---|
| LLM | Qwen-Plus（DashScope OpenAI 兼容模式） |
| Embedding | BAAI/bge-large-zh-v1.5（sentence-transformers） |
| 向量库 | ChromaDB 1.5.9（持久化） |
| PDF 解析 | pypdf / PyMuPDF |
| 后端 | FastAPI + Uvicorn |
| Agent | LangChain Tool + AgentExecutor |
| 评估 | RAGAS 0.4.3 |
| 容器化 | Docker（python:3.10-slim） |

## 目录结构

```
chinese-pdf-rag-benchmark/
├── src/
│   ├── api/                  # FastAPI 服务与 RAG 链路
│   │   ├── main.py          # 上传、提问接口
│   │   └── rag_chain.py     # 检索 → Prompt → Qwen 生成
│   ├── agent/                # Agent 工具化
│   │   ├── tools.py          # search_pdf 工具
│   │   └── agent.py          # AgentExecutor 入口
│   ├── chunking/             # 切分策略（fixed / recursive / regex / markdown）
│   ├── ingestion/            # PDF 解析、文本清洗、建库
│   ├── retrieval/            # 向量检索、混合检索
│   └── evaluation/            # 切分评估、检索评估
├── experiments/              # 每日实验记录与评估结果
│   ├── day1_log.md
│   ├── day3_chunking_matrix.md
│   ├── day4_chunking_eval.csv
│   ├── day5_retrieval_compare.md
│   ├── day6_testset.json
│   ├── day6_rag_outputs.csv
│   ├── day6_ragas_results.csv
│   ├── day6_test_record.md
│   ├── run_day6_ragas.py
│   └── final_report.md       # 汇总实验报告
├── data/
│   ├── raw/                  # 原始 PDF
│   ├── day2_cleaned/         # 清洗后文本
│   └── uploaded_pdfs/        # API 上传的 PDF
├── chroma_db/                # ChromaDB 持久化目录（.gitignore）
├── Dockerfile
├── .dockerignore
├── requirements.txt
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
# QWEN_API_KEY=sk-你的真实key

# 5. 启动服务
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

访问 http://127.0.0.1:8000/docs 查看 Swagger 文档。

### 方式二：Docker 运行

```bash
# 1. 构建镜像
docker build -t chinese-pdf-rag:latest .

# 2. 运行容器（把 .env 挂载进去，避免打进镜像）
docker run -p 8000:8000 --env-file .env chinese-pdf-rag:latest
```

访问 http://127.0.0.1:8000/docs。

## API 接口

### GET /

健康检查。

响应：

```json
{"status": "ok", "message": "Day6 RAG API is running"}
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

提问，返回答案和引用来源。

请求：

```json
{"question": "什么是马尔可夫决策过程？"}
```

响应：

```json
{
  "answer": "马尔可夫决策过程是...",
  "sources": [{"source": "xxx.pdf", "chunk_id": 3}],
  "contexts": ["检索到的原文片段..."]
}
```

## 评估结果

### RAGAS 三指标（10 个端到端问题）

| 指标 | 平均分 | 含义 |
|---|---|---|
| Faithfulness | 0.8683 | 答案对检索内容的忠实度，越高越少幻觉 |
| Context Precision | 0.9000 | 检索片段的精确率，越高噪声越少 |
| Context Recall | 1.0000 | 检索召回率，越高漏召越少 |

### 人工错误分析

已识别 3 类典型问题：

1. **检索漏召**：短小关键段落被高频章节挤掉 top-k
2. **轻度幻觉**：LLM 对公式自行展开解释
3. **PDF 解析噪声**：`\uf070` 等乱码字符影响 embedding 质量

**改进方向**：增大 n_results、加入 Rerank、优化 PDF 解析。

详见 `experiments/final_report.md`。

## 项目亮点

- **可复现**：Docker 打包 + requirements.txt 锁版本 + 完整实验记录
- **可评估**：RAGAS 自动评估 + 10 条端到端人工检查 + 错误归因
- **可扩展**：RAG 已封装为 Agent 工具，未来可挂载多工具协同
- **工程化**：分层目录（api / chunking / ingestion / retrieval / evaluation），前后 6 天迭代

## 常见问题

**Q：上传 PDF 后提问，回答"根据提供的资料无法回答"？**

A：说明该问题在已入库的 PDF 中找不到相关内容。可先上传包含答案的 PDF，或检查 `chroma_db/day4_chunking_eval` 集合是否有数据。

**Q：首次运行卡在模型下载？**

A：`BAAI/bge-large-zh-v1.5` 约 1.3GB，首次会从 HuggingFace 下载。已设置国内镜像 https://hf-mirror.com ，若仍慢可手动下载到 `~/.cache/huggingface/`。

**Q：Docker build 很慢？**

A：torch 等依赖较大，首次 build 约 20-40 分钟。后续 build 会命中缓存，速度快很多。

## 许可

MIT License

## 作者

@kiyy0520