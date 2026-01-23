FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

COPY pyproject.toml ./
RUN sed -i "s/{ index = \"pytorch-cu124\", marker = \"sys_platform != 'darwin'\" }/{ index = \"pytorch-cpu\", marker = \"sys_platform != 'darwin'\" }/g" pyproject.toml

RUN uv sync --no-install-project

COPY src src/
COPY README.md LICENSE ./

RUN uv sync

ENV PYTHONPATH=/app/src
ENV PORT=8080

CMD ["uv", "run", "uvicorn", "sign_ml.apicloud:app", "--host", "0.0.0.0", "--port", "8080"]

