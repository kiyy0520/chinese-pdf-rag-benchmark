# src/api/main.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"   # 强制离线，不再联网检查
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pypdf
import chromadb
from src.api.rag_chain import rag_query, embed_model

app = FastAPI(title="Day6 RAG API")

# 配置路径
UPLOAD_DIR = "data/uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "day4_chunking_eval"

# 初始化模型和数据库（只加载一次）
print("正在初始化 FastAPI 依赖...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=None)

class QuestionRequest(BaseModel):
    question: str

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
async def ask_question(request: QuestionRequest):
    try:
        result = rag_query(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成回答失败: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Day6 RAG API is running"}