FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY src src/
COPY README.md LICENSE ./

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "uvicorn", "src.sign_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
