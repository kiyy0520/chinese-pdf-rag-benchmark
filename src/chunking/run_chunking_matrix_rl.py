import os
import re
import json
import glob
from src.chunking.run_chunking_matrix import fixed_length_split, recursive_split, markdown_then_recursive_split, compute_stats

# ==========================================
# 1. 数据清洗函数（放在最顶部，模块级别）
# ==========================================
def clean_text(text):
    """
    数据清洗函数：去掉 PDF 解析带来的噪音，保留核心语义。
    """
    # 1. 去掉页码标记，如 "--- 第 8 页 ---" 或 "--- 第 8 页---"
    text = re.sub(r'---\s*第\s*\d+\s*页\s*---', '', text)
    
    # 2. 去掉页眉/作者信息（根据你的实际课件，去除类似"张伟楠 – 上海交通大学"的内容）
    text = re.sub(r'张伟楠\s*[–-]?\s*上海交通大学', '', text)
    
    # 3. 去掉无意义的哈希值（通常由数字和小写字母组成，长度在8位以上，如 f0954e19f）
    text = re.sub(r'\b[a-f0-9]{8,}\b', '', text)
    
    # 4. 去掉致谢或结束语（如 THANK YOU）
    text = re.sub(r'THANK\s*YOU', '', text, flags=re.IGNORECASE)
    
    # 5. 替换多余的空白行和空格，让文本更紧凑
    text = re.sub(r'\n\s*\n', '\n\n', text) # 多个空行合并为两个
    text = re.sub(r'[ \t]+', ' ', text)      # 多个空格合并为一个
    
    return text.strip()

# ==========================================
# 2. 主流程（放在 if __name__ 块中）
# ==========================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # 读取 day2_cleaned 下的真实 RL 文本 (合并所有 txt 文件)
    input_dir = os.path.join(project_root, "data", "day2_cleaned")
    text = ""
    for filepath in glob.glob(os.path.join(input_dir, "*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
            cleaned_text = clean_text(raw_text)  # 在这里注入清洗
            text += cleaned_text + "\n\n"
    
    print(f"成功读取 Day 2 RL 文本，总字符数：{len(text)}")

    # 输出到新目录，避免和之前的测试数据混淆
    output_dir = os.path.join(project_root, "data", "day3_outputs_rl")
    os.makedirs(output_dir, exist_ok=True)

    # 定义 9 组配置
    configs_to_run = [
        ("fixed", 512, 0),
        ("fixed", 512, 50),
        ("fixed", 1024, 0),
        ("markdown_recursive", 512, 0),
        ("markdown_recursive", 512, 100),
        ("markdown_recursive", 1024, 0),
        ("markdown_recursive", 1024, 50),
        ("markdown_recursive", 2048, 0),
        ("recursive", 512, 0)
    ]

    strategies = {
        "fixed": fixed_length_split,
        "recursive": recursive_split,
        "markdown_recursive": markdown_then_recursive_split
    }

    print("开始为 Day 2 RL 讲义批量切分...")
    print("=" * 60)

    for strategy_name, size, overlap in configs_to_run:
        split_func = strategies[strategy_name]
        chunks = split_func(text, size, overlap)
        stats = compute_stats(chunks)
        
        print(f"[{strategy_name}] size={size}, overlap={overlap} -> "
              f"块数:{stats['chunk_count']}, 平均:{stats['avg_len']}")
        
        # 导出 JSONL
        jsonl_filename = f"{strategy_name}_{size}_{overlap}.jsonl"
        jsonl_path = os.path.join(output_dir, jsonl_filename)
        with open(jsonl_path, "w", encoding="utf-8") as jf:
            for i, chunk_text in enumerate(chunks, 1):
                record = {
                    "chunk_id": f"{strategy_name}_{size}_{overlap}_{i}",
                    "strategy": strategy_name,
                    "chunk_size": size,
                    "overlap": overlap,
                    "text": chunk_text,
                    "length": len(chunk_text)
                }
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f"✅ 9 组 RL 切分数据已导出到：{output_dir}")