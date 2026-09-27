import time
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# 1. 设定输入和输出路径
pdf_path = "data/raw/sjtu_rl_lecture.pdf"  # 复用Day1的PDF
output_dir = Path("data/day2_raw/pypdf")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "sjtu_rl_lecture.txt"

# 2. 开始计时
start_time = time.perf_counter()

# 3. 初始化Loader并加载PDF
print(f"正在使用 PyPDFLoader 解析: {pdf_path}...")
loader = PyPDFLoader(pdf_path)
pages = loader.load()  # 返回一个 Document 对象列表

# 4. 结束计时
end_time = time.perf_counter()
elapsed_time = end_time - start_time

# 5. 将内容拼接并保存
full_text = ""
for i, page in enumerate(pages):
    full_text += f"\n--- 第 {i+1} 页 ---\n"
    full_text += page.page_content

with open(output_path, "w", encoding="utf-8") as f:
    f.write(full_text)

# 6. 打印统计信息
char_count = len(full_text)
print(f"解析完成！")
print(f"总页数: {len(pages)}")
print(f"总字符数: {char_count}")
print(f"耗时: {elapsed_time:.4f} 秒")
print(f"结果已保存至: {output_path}")