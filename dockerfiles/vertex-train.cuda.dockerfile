FROM ghcr.io/astral-sh/uv:latest AS uv_bin

FROM nvcr.io/nvidia/pytorch:25.01-py3

COPY --from=uv_bin /uv /uvx /bin/

WORKDIR /app

ENV UV_PYTHON=/usr/bin/python3.12
ENV PYTHONPATH=/app/src

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY src src/
COPY configs configs/
COPY README.md README.md
COPY LICENSE LICENSE

CMD ["uv", "run", "python", "-m", "sign_ml.vertex_train"]
