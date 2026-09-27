import os
import re

if __name__ == "__main__":
    # 1. 定位样本文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    sample_path = os.path.join(project_root, "data", "day3_sample.md")

    with open(sample_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 为了看清效果，我们只截取问答对那部分
    qa_text = text.split("## 第四部分：问答对测试")[1]
    
    print("【错误示范】：直接用 re.split 切分 Q:")
    # 错误：正则会把 'Q:' 吃掉，导致切出来的块丢失了 'Q:'
    bad_chunks = re.split(r"Q:", qa_text)
    for i, c in enumerate(bad_chunks[:2], 1):
        print(f"第{i}块开头: {c[:20].strip()}...")
    
    print("\n" + "="*40 + "\n")

    print("【正确修复】：使用零宽断言 (?=Q:) 切分")
    # 正确：(?=Q:) 表示“匹配 Q: 前面的位置，但不消耗 Q: 本身”
    # 这样 Q: 会被保留在下一个块的开头
    good_chunks = re.split(r"(?=Q:)", qa_text)
    for i, c in enumerate(good_chunks[:3], 1):
        # 去掉空字符串
        if c.strip():
            print(f"第{i}块开头: {c.strip()[:30]}...")
            print(f"   -> 长度: {len(c.strip())}")