from pathlib import Path
from collections import Counter
import pymupdf

pdf_path = "data/raw/sjtu_rl_lecture.pdf"

print(f"正在分析页眉页脚: {pdf_path}...")
doc = pymupdf.open(pdf_path)

# 用来存放所有页面的每一行文字
all_lines = []

for page in doc:
    text = page.get_text()
    lines = text.split("\n")
    for line in lines:
        line = line.strip()  # 去掉首尾空格
        if line:  # 如果是空行就跳过
            all_lines.append(line)

doc.close()

# 使用 Counter 统计每一行出现的次数
line_counts = Counter(all_lines)

# 按出现次数从高到低排序，打印前 20 行
print("\n=== 出现频率最高的 20 行文字 ===")
for line, count in line_counts.most_common(20):
    print(f"[{count} 次] {line}")