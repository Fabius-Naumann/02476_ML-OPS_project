FROM ghcr.io/astral-sh/uv:python3.12-bookworm

# -------------------------------------------------
# Docker / CI ONLY behaviour (does not affect local)
# -------------------------------------------------
ENV FORCE_CPU=1
ENV CUDA_VISIBLE_DEVICES=""

# Avoid large persistent caches in CI
ENV TORCH_HOME=/tmp/torch
ENV HF_HOME=/tmp/huggingface

WORKDIR /app

# ----------------------------
# Install dependencies
# ----------------------------
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# ----------------------------
# Copy source
# ----------------------------
COPY src src/
COPY README.md LICENSE ./

RUN uv sync --frozen

# ----------------------------
# Run API
# ----------------------------
ENTRYPOINT ["uv", "run", "uvicorn", "src.sign_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
