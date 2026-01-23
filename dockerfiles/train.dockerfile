FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

# Install dependencies from lockfile
COPY pyproject.toml ./

# Force CPU torch/torchvision for Linux-in-Docker by rewriting the uv sources mapping
RUN sed -i "s/{ index = \"pytorch-cu124\", marker = \"sys_platform != 'darwin'\" }/{ index = \"pytorch-cpu\", marker = \"sys_platform != 'darwin'\" }/g" pyproject.toml

RUN uv sync --no-install-project


# Copy project code
COPY src src/
COPY configs configs/
COPY README.md README.md
COPY LICENSE LICENSE

RUN uv sync

# Run training correctly
CMD ["uv", "run", "python", "-m", "sign_ml.train"]
