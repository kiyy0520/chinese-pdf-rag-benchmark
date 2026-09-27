FROM python:3.14-slim

WORKDIR /app

# 换 apt 为清华源，并把 stretch 替换成当前 Debian 版本
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list 2>/dev/null || true

# 装系统级依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 用清华 pip 源安装依赖，加速下载
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

COPY src/ ./src/
COPY experiments/ ./experiments/
COPY data/ ./data/

ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]