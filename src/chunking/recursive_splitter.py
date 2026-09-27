import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

if __name__ == "__main__":
    # 1. 智能定位样本文件（沿用我们刚才的路径写法）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    sample_path = os.path.join(project_root, "data", "day3_sample.md")

    print(f"正在读取文件：{sample_path}")
    print("-" * 40)

    with open(sample_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2. 初始化递归字符切分器
    # 参数解释：
    # chunk_size=512：目标每块最大 512 个字符
    # chunk_overlap=50：相邻块之间重叠 50 个字符
    # separators：优先按段落、换行、句子切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    )

    # 3. 执行切分
    chunks = splitter.split_text(text)

    # 4. 打印结果
    print(f"总共切出了 {len(chunks)} 个 chunk")
    print("=" * 40)
    for i, chunk in enumerate(chunks[:3], 1):  # 只看前3块
        print(f"第 {i} 块（长度 {len(chunk)}）：")
        print(chunk)
        print("-" * 40)