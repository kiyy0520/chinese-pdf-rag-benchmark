import os

def fixed_length_split(text, chunk_size=512, overlap=50):
    """
    固定长度切分器
    :param text: 要切分的完整文本
    :param chunk_size: 每个 chunk 的最大长度（按字符算）
    :param overlap: 相邻 chunk 之间重叠的字符数
    :return: 切分后的 chunk 列表
    """
    chunks = []
    start = 0
    text_length = len(text)

    # 防止 overlap 大于等于 chunk_size 导致死循环
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size，否则会陷入死循环！")

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # 下一块的起点 = 当前起点 + chunk_size - overlap
        start = start + chunk_size - overlap

    return chunks


if __name__ == "__main__":
    # 1. 自动定位项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))  # src/chunking
    project_root = os.path.dirname(os.path.dirname(current_dir)) # 根目录
    sample_path = os.path.join(project_root, "data", "day3_sample.md") # 匹配你的新文件名

    print(f"正在读取文件：{sample_path}")
    print("-" * 40)

    # 2. 读取样本文档
    with open(sample_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 3. 用固定长度切分（默认 512 / 50）
    chunks = fixed_length_split(text, chunk_size=512, overlap=50)

    # 4. 打印结果
    print(f"总共切出了 {len(chunks)} 个 chunk")
    print("=" * 40)
    print("第一个 chunk 的内容（前200字）：")
    print(chunks[0][:200] + "...")
    print("=" * 40)
    print("第二个 chunk 的内容（前200字）：")
    print(chunks[1][:200] + "...")