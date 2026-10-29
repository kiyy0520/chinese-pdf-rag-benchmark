FROM python:3.10-slim

WORKDIR /app

# 换 apt 为清华源
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list 2>/dev/null || true

# 装系统级依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple uv

# 拷贝依赖声明文件
COPY requirements.in .

# 解析并安装依赖
RUN uv pip compile requirements.in -o requirements.txt \
        --index-strategy unsafe-best-match \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        --extra-index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/ \
    && uv pip install --system -r requirements.txt \
        --index-strategy unsafe-best-match \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        --extra-index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/

# 拷贝源码与数据
COPY src/ ./src/
COPY experiments/ ./experiments/
COPY data/ ./data/
# 【新增】把你根目录下的运行脚本也拷进去，不然容器里找不到
COPY run_week4_day6_comparison.py .
COPY run_week4_day6_ragas_eval.py .

ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# 默认启动 API 服务（如果要在容器跑评测，可以 docker run 时覆盖此命令）
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]