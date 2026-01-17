import os
from pathlib import Path

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "sign_ml"
PYTHON_VERSION = "3.12"

PROFILING_DIR = Path("src") / PROJECT_NAME / "profiling"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def profile_data(ctx: Context, out: str = "profile_data.prof") -> None:
    """Profile the data preprocessing step using cProfile."""
    PROFILING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(Path(out) if Path(out).is_absolute() else (PROFILING_DIR / out))
    ctx.run(f"uv run python -m cProfile -o {out_path} src/{PROJECT_NAME}/data.py", echo=True, pty=not WINDOWS)


@task
def profile_train(ctx: Context, out: str = "profile_train.prof", epochs: int = 1, batch_size: int | None = None) -> None:
    """Profile a short training run using cProfile.

    Args:
        ctx: Invoke context.
        out: Output path for the .prof file.
        epochs: Number of epochs to run for profiling.
        batch_size: Optional batch size override.
    """

    PROFILING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(Path(out) if Path(out).is_absolute() else (PROFILING_DIR / out))

    overrides = [
        "hydra.job.chdir=false",
        "hydra.run.dir=.",
        f"experiment.training.epochs={epochs}",
    ]
    if batch_size is not None:
        overrides.append(f"experiment.training.batch_size={batch_size}")

    ctx.run(
        f"uv run python -m cProfile -o {out_path} src/{PROJECT_NAME}/train.py " + " ".join(overrides),
        echo=True,
        pty=not WINDOWS,
    )


@task
def profile_evaluate(ctx: Context, out: str = "profile_evaluate.prof", batch_size: int | None = None) -> None:
    """Profile evaluation using cProfile.

    Args:
        ctx: Invoke context.
        out: Output path for the .prof file.
        batch_size: Optional batch size override.
    """

    PROFILING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(Path(out) if Path(out).is_absolute() else (PROFILING_DIR / out))

    overrides = [
        "hydra.job.chdir=false",
        "hydra.run.dir=.",
    ]
    if batch_size is not None:
        overrides.append(f"experiment.training.batch_size={batch_size}")

    ctx.run(
        f"uv run python -m cProfile -o {out_path} src/{PROJECT_NAME}/evaluate.py " + " ".join(overrides),
        echo=True,
        pty=not WINDOWS,
    )


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
