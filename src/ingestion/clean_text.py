import re
from pathlib import Path
from collections import Counter

input_path = "data/day2_raw/pymupdf/2-马尔可夫决策过程.txt"
output_dir = Path("data/day2_cleaned")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "2-马尔可夫决策过程_cleaned.txt"
# 1. 读取原始文本
with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 2. 统计每一行（去除首尾空格后）的出现频率
stripped_lines = [line.strip() for line in lines if line.strip()]
line_counts = Counter(stripped_lines)

# 3. 定义清洗规则
cleaned_lines = []
for line in lines:
    line = line.strip()
    if not line:
        continue  # 跳过空行

    # 规则 1：如果是纯数字（页码），或者单字符如"2"，直接删掉
    if re.fullmatch(r"\d+", line):
        continue

    # 规则 2：如果是高频出现且非常短的行（长度 <= 3），比如 "1." "2." 等，直接删掉
    #if line_counts[line] > 5 and len(line) <= 3:
        #continue

    # 规则 3：合并被拆开的列表序号，如 "1." 单独一行，下一行是内容
    # 如果当前行是 "数字." 或 "•" 这种列表符号，且长度 <= 3，就把它和下一行拼起来
    # 由于 Python 的 for 循环不便预读下一行，这里改用索引遍历
    cleaned_lines.append(line)

# 对列表序号进行合并处理
final_lines = []
i = 0
while i < len(cleaned_lines):
    line = cleaned_lines[i]
    # 如果当前行是 "数字." 或 "•" 这种纯符号行，并且不是最后一行
    if re.fullmatch(r"(\d+\.|•)", line) and i + 1 < len(cleaned_lines):
        # 把序号和下一行内容合并，并加上空格
        merged = line + " " + cleaned_lines[i + 1]
        final_lines.append(merged)
        i += 2  # 跳过下一行
    else:
        final_lines.append(line)
        i += 1

# 4. 统一换行和空格
# 把所有行拼起来，遇到句号、问号、感叹号后加换行，其他保持一行
text = "\n".join(final_lines)
# 把多个连续空格替换成一个空格
text = re.sub(r" +", " ", text)
# 把多个连续换行替换成两个换行（段落之间保留一个空行）
text = re.sub(r"\n{3,}", "\n\n", text)

# 5. 保存清洗后的文本
with open(output_path, "w", encoding="utf-8") as f:
    f.write(text)

# 6. 统计清洗前后字符数变化（去掉我们自己加的页眉标记，只对比正文）
original_text = "".join([line for line in lines if not line.strip().startswith("---")])
original_len = len(original_text)
cleaned_len = len(text)
change_rate = abs(cleaned_len - original_len) / original_len * 100

print(f"清洗完成！")
print(f"清洗前字符数: {original_len}")
print(f"清洗后字符数: {cleaned_len}")
print(f"文本量变化率: {change_rate:.2f}% （目标 < 5%）")
print(f"结果已保存至: {output_path}")