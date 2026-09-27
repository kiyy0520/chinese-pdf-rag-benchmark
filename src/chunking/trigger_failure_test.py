import os
import re

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    sample_path = os.path.join(project_root, "data", "day3_sample.md")

    with open(sample_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("=" * 60)
    print("【精准打击 1：切断代码块】")
    print("=" * 60)
    # 单独提取代码块
    code_match = re.search(r"```python.*?```", text, re.DOTALL)
    if code_match:
        code_block = code_match.group(0)
        # 用 chunk_size=50 硬切代码块
        chunk_size = 50
        for i in range(0, len(code_block), chunk_size):
            chunk = code_block[i:i + chunk_size]
            print(f"第 {i//chunk_size + 1} 块: {repr(chunk)}")

    print("\n" + "=" * 60)
    print("【精准打击 2：切断问答对】")
    print("=" * 60)
    # 单独提取问答对
    qa_match = re.search(r"Q:.*", text, re.DOTALL)
    if qa_match:
        qa_block = qa_match.group(0)
        # 用 chunk_size=30 硬切问答对
        chunk_size = 30
        for i in range(0, len(qa_block), chunk_size):
            chunk = qa_block[i:i + chunk_size]
            print(f"第 {i//chunk_size + 1} 块: {repr(chunk)}")