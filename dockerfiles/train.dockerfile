FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

# Install dependencies from lockfile
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
RUN uv sync --frozen --no-install-project

# Copy project code
COPY src src/
COPY configs configs/
COPY README.md README.md
COPY LICENSE LICENSE

# Run training correctly
CMD ["uv", "run", "python", "-m", "sign_ml.train"]
