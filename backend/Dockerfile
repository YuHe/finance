# ─── Base ───
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# ─── Dependencies ───
FROM base AS deps

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Runtime ───
FROM deps AS runtime

WORKDIR /app

# 复制核心引擎代码
COPY data_layer/ /app/data_layer/
COPY factor_layer/ /app/factor_layer/
COPY regime_layer/ /app/regime_layer/
COPY selection_layer/ /app/selection_layer/
COPY execution_layer/ /app/execution_layer/
COPY strategy_layer/ /app/strategy_layer/
COPY llm_layer/ /app/llm_layer/
COPY backtest.py /app/backtest.py

# 复制后端API代码
COPY backend/main.py /app/main.py
COPY backend/app/ /app/app/

# 数据目录
RUN mkdir -p /app/data

# 非root用户
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -s /bin/bash appuser && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
