FROM ghcr.io/astral-sh/uv:python3.12-bookworm

# Set working directory
WORKDIR /app

# Copy dependency files first (for better layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies (CPU-only, reproducible)
RUN uv sync --frozen --no-install-project

# Copy application code and metadata
COPY src src/
COPY README.md LICENSE ./

# Install project itself
RUN uv sync --frozen

# Run FastAPI app
ENTRYPOINT [
  "uv",
  "run",
  "uvicorn",
  "src.sign_ml.api:app",
  "--host", "0.0.0.0",
  "--port", "8000"
]
