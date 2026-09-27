import time
from pathlib import Path
import pymupdf

pdf_path = "data/raw/2-马尔可夫决策过程.pdf"
output_dir = Path("data/day2_raw/pymupdf")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "2-马尔可夫决策过程.txt"
start_time = time.perf_counter()

print(f"正在使用 PyMuPDF 解析: {pdf_path}...")
doc = pymupdf.open(pdf_path)  # 打开PDF

full_text = ""
for i, page in enumerate(doc):
    # 提取当前页的文字
    page_text = page.get_text()
    full_text += f"\n--- 第 {i+1} 页 ---\n"
    full_text += page_text

end_time = time.perf_counter()
elapsed_time = end_time - start_time

with open(output_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"\n解析完成！")
print(f"总页数: {len(doc)}")  # 注意：doc.close()后可能取不到长度，我们直接打印统计
print(f"总字符数: {len(full_text)}")
print(f"耗时: {elapsed_time:.4f} 秒")
print(f"结果已保存至: {output_path}")

doc.close()  # 3. 挪到最后