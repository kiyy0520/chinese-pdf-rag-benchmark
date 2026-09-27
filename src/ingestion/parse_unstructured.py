import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time
from pathlib import Path
from langchain_community.document_loaders import UnstructuredPDFLoader

pdf_path = "data/raw/sjtu_rl_lecture.pdf"
output_dir = Path("data/day2_raw/unstructured")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "sjtu_rl_lecture.txt"

start_time = time.perf_counter()

print(f"正在使用 UnstructuredLoader 解析: {pdf_path}...")
# 注意这里加了 mode="elements"，这是它比 PyPDF 强大的关键所在
loader = UnstructuredPDFLoader(pdf_path, mode="elements", strategy="fast")
docs = loader.load()

end_time = time.perf_counter()
elapsed_time = end_time - start_time

full_text = ""
for i, doc in enumerate(docs):
    # 获取这个元素的类别（比如 Title, NarrativeText, ListItem 等）
    category = doc.metadata.get('category', '未知')
    full_text += f"\n--- 元素 {i+1} (类型: {category}) ---\n"
    full_text += doc.page_content

with open(output_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"\n解析完成！")
print(f"总元素数: {len(docs)}")
print(f"总字符数: {len(full_text)}")
print(f"耗时: {elapsed_time:.4f} 秒")
print(f"结果已保存至: {output_path}")