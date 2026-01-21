FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

# ----------------------------
# Install dependencies
# ----------------------------
COPY pyproject.toml ./

# Force CPU torch/torchvision for Linux-in-Docker by rewriting the uv sources mapping
RUN sed -i "s/{ index = \"pytorch-cu118\", marker = \"sys_platform != 'darwin'\" }/{ index = \"pytorch-cpu\", marker = \"sys_platform != 'darwin'\" }/g" pyproject.toml

RUN uv sync --no-install-project

# ----------------------------
# Copy source
# ----------------------------
COPY src src/
COPY README.md LICENSE ./

RUN uv sync

# ----------------------------
# Run API
# ----------------------------
ENTRYPOINT ["uv", "run", "uvicorn", "src.sign_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
