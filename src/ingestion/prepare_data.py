import os
import json

# 1. 定义我们选定的 9 组切分配置（注意文件名要和你的实际文件完全一致）
configs = [
    "fixed_512_0",
    "fixed_512_50",
    "fixed_1024_0",
    "markdown_recursive_512_0",
    "markdown_recursive_512_100",
    "markdown_recursive_1024_0",
    "markdown_recursive_1024_50",
    "markdown_recursive_2048_0",
    "recursive_512_0"
]

DATA_DIR = "data/day3_outputs_rl"

def load_and_prepare():
    all_ids = []
    all_documents = []
    all_metadatas = []

    for config in configs:
        file_path = os.path.join(DATA_DIR, f"{config}.jsonl")
        
        # 检查文件是否存在，防止你名字打错
        if not os.path.exists(file_path):
            print(f"警告：找不到文件 {file_path}，请检查文件名是否正确！")
            continue

        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                
                # 提取我们需要的字段
                all_ids.append(data["chunk_id"])
                all_documents.append(data["text"])
                
                # 把除了 chunk_id 和 text 之外的信息作为元数据，存入 metadata
                # 这一步非常重要，后面我们要靠 metadata 里的 strategy 来做过滤
                meta = {k: v for k, v in data.items() if k not in ["chunk_id", "text"]}
                meta["config_name"] = config  # 额外增加一个 config_name 方便过滤
                all_metadatas.append(meta)
                
                count += 1
                
        print(f"✅ 已读取 {config}.jsonl，共 {count} 个 chunk")

    print("-" * 30)
    print(f"总计：{len(all_ids)} 个 chunk 准备完毕！")
    print(f"第一个 chunk 的 metadata 示例：{all_metadatas[0] if all_metadatas else '无'}")
    
    return all_ids, all_documents, all_metadatas

if __name__ == "__main__":
    load_and_prepare()