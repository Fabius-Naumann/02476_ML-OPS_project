import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "sign_ml"
PYTHON_VERSION = "3.12"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/merge_data.py", echo=True, pty=not WINDOWS)
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py --preprocess", echo=True, pty=not WINDOWS)


@task
def viz_data(ctx: Context, samples: int = 16, output: str | None = None) -> None:
    """Visualize data samples.

    Args:
        samples: Number of samples to visualize.

        output: Optional output image path for the plot.
    """
    cmd = f"uv run src/{PROJECT_NAME}/data.py --samples {samples}"
    if output:
        cmd += f" --output {output}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def sweep(ctx: Context) -> None:
    """Create a W&B sweep from configs/sweep.yaml."""
    ctx.run("uv run python -m wandb sweep configs/sweep.yaml", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
