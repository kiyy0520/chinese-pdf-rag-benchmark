import os
from langchain_text_splitters import MarkdownHeaderTextSplitter

if __name__ == "__main__":
    # 1. 智能定位样本文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    sample_path = os.path.join(project_root, "data", "day3_sample.md")

    print(f"正在读取文件：{sample_path}")
    print("-" * 40)

    with open(sample_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2. 定义标题层级映射
    # 告诉切分器：遇到 # 就是一级标题，## 就是二级标题，### 就是三级标题
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # 3. 初始化 Markdown 标题切分器
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    # 4. 执行切分
    # 注意：这里返回的 chunks 是一个个 Document 对象，不是纯字符串
    md_chunks = markdown_splitter.split_text(text)

    # 5. 打印结果
    print(f"总共切出了 {len(md_chunks)} 个 chunk")
    print("=" * 40)
    for i, chunk in enumerate(md_chunks, 1):
        # chunk.page_content 是文本内容
        # chunk.metadata 是标题信息（比如它是哪个 ## 下面的）
        print(f"第 {i} 块（标题信息：{chunk.metadata}，长度：{len(chunk.page_content)}）")
        print(chunk.page_content[:150] + "...") # 只打印前150字预览
        print("-" * 40)