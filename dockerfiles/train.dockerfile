FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

ENV SIGN_ML_BASE_DIR=/app

# Install dependencies from lockfile
COPY pyproject.toml ./

# Force CPU torch/torchvision for Linux-in-Docker by rewriting the uv sources mapping
RUN sed -i "s/{ index = \"pytorch-cu118\", marker = \"sys_platform != 'darwin'\" }/{ index = \"pytorch-cpu\", marker = \"sys_platform != 'darwin'\" }/g" pyproject.toml

RUN uv sync --no-install-project


# Copy project code
COPY src src/
COPY configs configs/
COPY README.md README.md
COPY LICENSE LICENSE

# Copy data so training can run inside the container
COPY data/processed data/processed
COPY data/raw/traffic_signs_merged.zip data/raw/

RUN uv sync

# Run training correctly
CMD ["uv", "run", "python", "-m", "sign_ml.train"]
