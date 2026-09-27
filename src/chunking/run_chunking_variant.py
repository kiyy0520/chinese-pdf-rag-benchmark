import os
import json
import statistics
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- 1. 基础切分器（复用之前的递归切分） ----
def recursive_split(text, chunk_size, overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    )
    return splitter.split_text(text)

# ---- 2. 后处理：合并小碎片 ----
def merge_small_chunks(chunks, min_size=150):
    """
    如果某个 chunk 的长度小于 min_size，就把它合并到前一个 chunk 中。
    如果第一个 chunk 就太小，就和第二个合并。
    """
    if not chunks:
        return []
    
    merged = [chunks[0]]
    for i in range(1, len(chunks)):
        # 如果当前块太小，合并到前一个块
        if len(chunks[i]) < min_size:
            print(f"  >> 合并第{i+1}块（长度 {len(chunks[i])}）到前一块")
            print(f"     被合并块开头: {chunks[i][:30].strip()}...")
            merged[-1] += "\n" + chunks[i]
        else:
            merged.append(chunks[i])
    
    # 特殊处理：如果第一个块也太小，合并到第二个块
    if len(merged) > 1 and len(merged[0]) < min_size:
        print(f"  >> 合并第一块（长度 {len(merged[0])}）到第二块")
        merged[1] = merged[0] + "\n" + merged[1]
        merged.pop(0)
        
    return merged

# ---- 3. 统计函数 ----
def compute_stats(chunks):
    lengths = [len(c) for c in chunks]
    if not lengths:
        return {"chunk_count": 0, "avg_len": 0, "max_len": 0, "min_len": 0}
    return {
        "chunk_count": len(lengths),
        "avg_len": round(statistics.mean(lengths), 2),
        "max_len": max(lengths),
        "min_len": min(lengths)
    }

# ---- 4. 主流程 ----
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    sample_path = os.path.join(project_root, "data", "day3_challenge.md")

    with open(sample_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 使用基础递归切分（512, 50）
    print("使用 recursive_512_50 进行初始切分...")
    original_chunks = recursive_split(text, 512, 50)
    original_stats = compute_stats(original_chunks)
    print(f"【合并前】chunk数量: {original_stats['chunk_count']}, "
          f"平均: {original_stats['avg_len']}, "
          f"最大: {original_stats['max_len']}, "
          f"最小: {original_stats['min_len']}")

    # 执行合并
    print("\n执行小碎片合并（min_size=150）...")
    merged_chunks = merge_small_chunks(original_chunks, min_size=150)
    merged_stats = compute_stats(merged_chunks)
    print(f"【合并后】chunk数量: {merged_stats['chunk_count']}, "
          f"平均: {merged_stats['avg_len']}, "
          f"最大: {merged_stats['max_len']}, "
          f"最小: {merged_stats['min_len']}")

    # 打印一个合并前后的对比示例
    print("\n" + "=" * 50)
    print("合并前后对比示例：")
    print(f"合并前第1块结尾: ...{original_chunks[0][-50:]}")
    if len(original_chunks) > 1:
        print(f"合并前第2块开头: {original_chunks[1][:50]}...")
    print(f"合并后第1块结尾: ...{merged_chunks[0][-50:]}")
    if len(merged_chunks) > 1:
        print(f"合并后第2块开头: {merged_chunks[1][:50]}...")

    # 导出合并后的 JSONL
    output_dir = os.path.join(project_root, "data", "day3_variant_outputs")
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "recursive_512_50_merged.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for i, chunk_text in enumerate(merged_chunks, 1):
            record = {
                "chunk_id": f"recursive_512_50_merged_{i}",
                "strategy": "recursive_512_50_with_merge",
                "text": chunk_text,
                "length": len(chunk_text)
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n合并后的 JSONL 已导出到：{jsonl_path}")