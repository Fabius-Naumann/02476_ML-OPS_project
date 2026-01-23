FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app


COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
RUN uv sync --frozen --no-install-project

COPY src src/
COPY configs configs/
COPY README.md README.md
COPY LICENSE LICENSE
ENV PYTHONPATH=/app/src

CMD ["uv", "run", "python", "-m", "sign_ml.vertex_train"]