# src/api/main.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pypdf

# 引入 Week 4 的核心类
from src.conversation.multi_turn_rag import MultiTurnRAGPipeline
from src.api.rag_chain import embed_model

app = FastAPI(title="Week 4 Multi-turn RAG API")

# 配置路径
UPLOAD_DIR = "data/uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 初始化全局多轮对话流水线（只加载一次模型和数据库）
print("正在初始化 FastAPI 依赖与多轮对话 Pipeline...")
pipeline = MultiTurnRAGPipeline()
collection = pipeline.collection # 复用内部的 Chroma collection

# --- 定义数据模型 ---

class Message(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    question: str
    history: Optional[List[Message]] = []

# --- 接口定义 ---

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # 1. 保存上传的 PDF
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. 提取文本
    try:
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF解析失败: {str(e)}")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="PDF中未提取到文字（可能是扫描版）")

    # 3. 切分（512字符，overlap 50）
    chunk_size = 512
    overlap = 50
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    
    # 4. 向量化
    embeddings = embed_model.encode(
        chunks, 
        normalize_embeddings=True
    ).tolist()
    
    # 5. 存入 Chroma
    ids = [f"{file.filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": file.filename, "chunk_id": i} for i in range(len(chunks))]
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )
    
    return {
        "message": "上传并入库成功",
        "filename": file.filename,
        "chunks_added": len(chunks)
    }

@app.post("/ask")
async def ask_question(request: AskRequest):
    try:
        # 【核心】重置全局 pipeline 的状态，防止并发/多用户数据污染
        pipeline.history = []
        
        # 将前端传来的历史记录转换成 pipeline 需要的格式并注入
        if request.history:
            for msg in request.history:
                if msg.role == "user":
                    pipeline.history.append({"user": msg.content, "assistant": ""})
                elif msg.role == "assistant" and pipeline.history:
                    pipeline.history[-1]["assistant"] = msg.content
                    
        # 执行 Week 4 的多轮处理逻辑
        # 内部会经历：状态更新 -> Query 改写 -> 检索 -> 生成 -> 拒答判断
        result = pipeline.process_turn(request.question)
        
        # 直接返回结果（包含 rewritten_query, current_state, is_refused 等全量字段）
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"多轮对话失败: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Week 4 Multi-turn RAG API is running"}