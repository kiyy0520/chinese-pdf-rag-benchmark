import os
import json
import statistics
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

# ---- 1. 定义我们的切分器 ----

def fixed_length_split(text, chunk_size, overlap):
    """固定长度切分"""
    chunks = []
    start = 0
    text_length = len(text)
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def recursive_split(text, chunk_size, overlap):
    """递归字符切分"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    )
    return splitter.split_text(text)

def markdown_then_recursive_split(text, chunk_size, overlap):
    """先按 Markdown 标题切，再套一层递归切分限制大小"""
    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_chunks = md_splitter.split_text(text)
    
    # 二级切分：把过大的 md_chunk 继续切小
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    )
    
    final_chunks = []
    for md_chunk in md_chunks:
        sub_chunks = recursive_splitter.split_text(md_chunk.page_content)
        final_chunks.extend(sub_chunks)
    return final_chunks

# ---- 2. 统计函数 ----

def compute_stats(chunks):
    """输入 chunk 列表，输出统计指标"""
    lengths = [len(c) for c in chunks]
    if not lengths:
        return {"chunk_count": 0, "avg_len": 0, "max_len": 0, "min_len": 0}
    return {
        "chunk_count": len(lengths),
        "avg_len": round(statistics.mean(lengths), 2),
        "max_len": max(lengths),
        "min_len": min(lengths)
    }

# ---- 3. 主流程 ----

if __name__ == "__main__":
    # 定位文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    sample_path = os.path.join(project_root, "data", "day3_sample.md")
    
    # 输出目录
    output_dir = os.path.join(project_root, "data", "day3_outputs")
    os.makedirs(output_dir, exist_ok=True) # 不存在就自动创建

    with open(sample_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 定义 9 组矩阵参数
    chunk_sizes = [512, 1024, 2048]
    overlaps = [0, 50, 100]
    
    # 策略字典：策略名 -> 切分函数
    strategies = {
        "fixed": fixed_length_split,
        "recursive": recursive_split,
        "markdown_recursive": markdown_then_recursive_split
    }

    # 准备记录表
    results = []

    print("开始批量跑数...")
    print("=" * 60)

    for strategy_name, split_func in strategies.items():
        for size in chunk_sizes:
            for overlap in overlaps:
                # 执行切分
                chunks = split_func(text, size, overlap)
                # 统计
                stats = compute_stats(chunks)
                
                # 打印一行日志
                print(f"[{strategy_name}] size={size}, overlap={overlap} -> "
                      f"块数:{stats['chunk_count']}, 平均:{stats['avg_len']}, "
                      f"最大:{stats['max_len']}, 最小:{stats['min_len']}")
                
                # 记录到表格
                results.append({
                    "strategy": strategy_name,
                    "chunk_size": size,
                    "overlap": overlap,
                    **stats
                })
                
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
    print(f"全部跑完！共 {len(results)} 组配置。")
    print(f"JSONL 数据集已导出到：{output_dir}")