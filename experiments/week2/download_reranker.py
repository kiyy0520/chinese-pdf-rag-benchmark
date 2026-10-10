# experiments/download_reranker.py
from modelscope import snapshot_download

print("开始从 ModelScope 下载 bge-reranker-large（仅必要文件）...")
model_dir = snapshot_download(
    'BAAI/bge-reranker-large',
    cache_dir=r'F:\Projects\chinese-pdf-rag-benchmark\models',
    allow_patterns=[
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
        "model.safetensors",  # 只要 safetensors，不要 bin 和 onnx
    ]
)
print(f"\n下载完成，模型路径：{model_dir}")