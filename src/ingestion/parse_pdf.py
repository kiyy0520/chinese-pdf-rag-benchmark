import os
import re
from langchain_community.document_loaders import PyPDFLoader

# ================= 配置区 =================
# 换新 PDF 时，只需要改这两行
RAW_PDF_NAME = "sjtu_rl_lecture.pdf"        # <--- 改这里
OUTPUT_TXT_NAME = "day1_pdf_text.txt"

RAW_PDF_PATH = os.path.join("data", "raw", RAW_PDF_NAME)
OUTPUT_TXT_PATH = os.path.join("data", "processed", OUTPUT_TXT_NAME)
# ==========================================

def main():
    print(f"开始解析: {RAW_PDF_NAME}")
    
    # 1. 加载 PDF
    loader = PyPDFLoader(RAW_PDF_PATH)
    pages = loader.load()
    
    # 2. 提取原始文本
    raw_text = "\n".join([page.page_content for page in pages])
    
    # 3. 基础清洗：去空行和纯数字页码
        # 3. 基础清洗：去空行、去页码、去特定页眉
    cleaned_lines = []
    for line in raw_text.splitlines():
        line = line.strip()
        
        # 过滤空行
        if not line:
            continue
            
        # 过滤纯数字页码（比如 "1", "28"）
        if line.isdigit():
            continue
            
        # 过滤 "第X章" 页眉（用字符串匹配，简单直观）
        if "第" in line and "章" in line:
            continue
            
        # 过滤 "- 1 -" 这种带横杠的页码（用正则，精准防误杀）
        # 解释：^表示开头，- 表示横杠，\s* 表示任意个空格，\d+ 表示1个或多个数字，$表示结尾
        if re.match(r"^-\s*\d+\s*-$", line):
            continue
            
        cleaned_lines.append(line)
    
    cleaned_text = "\n".join(cleaned_lines)
    
    # 4. 写入文件（自动创建目录）
    os.makedirs(os.path.dirname(OUTPUT_TXT_PATH), exist_ok=True)
    with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    
    # 5. 量化验证（Day1 验收指标）
    raw_len = len(raw_text)
    clean_len = len(cleaned_text)
    change_rate = abs(raw_len - clean_len) / raw_len * 100
    
    print("-" * 30)
    print(f"原始字符数: {raw_len}")
    print(f"清洗后字符数: {clean_len}")
    print(f"字符变化率: {change_rate:.2f}%")
    print(f"产出文件: {OUTPUT_TXT_PATH}")
    print("-" * 30)

if __name__ == "__main__":
    main()