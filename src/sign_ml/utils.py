import os
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from types import ModuleType
from typing import IO, Any, cast

import torch
import wandb
from omegaconf import DictConfig, OmegaConf


def _find_repo_root(start: Path) -> Path:
    """Find the repository root by searching for common marker files.

    Args:
        start: Starting path (file or directory) to begin searching from.

    Returns:
        Path to the detected repository root.
    """

    current = start.resolve()
    if current.is_file():
        current = current.parent

    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent

    return current


def _next_counter_value(counter_file: Path) -> int:
    """Return the next integer value for a local counter file.

    The counter is stored on disk so consecutive sweep runs can be named
    deterministically (e.g., sweep1, sweep2, ...).

    Notes:
        This uses a best-effort file lock to reduce collisions when multiple
        sweep agents run in parallel on the same machine.

        This does not coordinate across machines or containers that do not share
        the same filesystem.

    Args:
        counter_file: Path to a text file storing the last used counter value.

    Returns:
        The next counter value (starting from 1).
    """

    @contextmanager
    def _exclusive_lock(file_obj: IO[str]) -> Iterator[None]:
        fd = file_obj.fileno()

        fcntl_module: ModuleType | None
        try:
            import fcntl as fcntl_module  # type: ignore[import-not-found]
        except ImportError:
            fcntl_module = None

        if fcntl_module is not None:
            fcntl_module.flock(fd, fcntl_module.LOCK_EX)
            try:
                yield
            finally:
                fcntl_module.flock(fd, fcntl_module.LOCK_UN)
            return

        try:
            import msvcrt
        except ImportError:
            yield
            return

        file_obj.seek(0)
        msvcrt_any = cast(Any, msvcrt)
        msvcrt_any.locking(fd, msvcrt_any.LK_LOCK, 1)
        try:
            yield
        finally:
            file_obj.seek(0)
            msvcrt_any.locking(fd, msvcrt_any.LK_UNLCK, 1)

    counter_file.parent.mkdir(parents=True, exist_ok=True)

    fd = os.open(counter_file, os.O_RDWR | os.O_CREAT)
    with os.fdopen(fd, "r+", encoding="utf-8") as f, _exclusive_lock(f):
        f.seek(0)
        raw = f.read().strip()
        try:
            last_value = int(raw) if raw else 0
        except ValueError:
            last_value = 0

        next_value = last_value + 1
        f.seek(0)
        f.write(str(next_value))
        f.truncate()
        f.flush()
        with suppress(OSError):
            # Best-effort durability: ignore fsync failures as this counter file
            # is non-critical and an unsynced write only risks losing the latest increment.
            os.fsync(f.fileno())

        return next_value


def device_from_cfg(device: str) -> torch.device:
    """Return torch.device based on config value.

    Args:
        device: Device string, e.g. 'auto', 'cpu', 'cuda'.

    Returns:
        torch.device: Selected device.
    """
    if device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def get_wandb_init_kwargs(cfg: DictConfig, run_name: str | None = None, group: str | None = None) -> dict[str, Any]:
    """Build keyword arguments for ``wandb.init`` from config and environment.

    This helper centralizes how Weights & Biases runs are configured, including
    reading environment variables and resolving the run configuration.

    Args:
        cfg: Hydra configuration for the current run.
        run_name: Optional name for the wandb run.

    Returns:
        Dict[str, Any]: Keyword arguments suitable for ``wandb.init``.
    """

    wandb_project = os.getenv("WANDB_PROJECT", "sign-ml")
    wandb_entity = os.getenv("WANDB_ENTITY")

    wandb_dir = os.getenv("WANDB_DIR")
    if wandb_dir:
        Path(wandb_dir).mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = {
        "project": wandb_project,
        "entity": wandb_entity,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    if run_name is not None:
        kwargs["name"] = run_name
    if group is not None:
        kwargs["group"] = group

    if wandb_dir:
        kwargs["dir"] = wandb_dir

    return kwargs


def init_wandb(cfg: DictConfig, run_name: str | None = None, group: str | None = None) -> tuple[bool, Exception | None]:
    """Initialize Weights & Biases (fail-soft).

    Args:
        cfg: Hydra configuration for the current run.
        run_name: Optional W&B run name. If omitted, W&B will auto-generate a unique name.
        group: Optional W&B group name (useful to group sweep runs).

    Returns:
        Tuple (use_wandb, error). If initialization fails, use_wandb is False and error contains the exception.
    """

    try:
        run = wandb.init(**get_wandb_init_kwargs(cfg, run_name=run_name, group=group))
    except Exception as exc:
        return False, exc

    if run is not None and run_name is None:
        # By default we let W&B generate unique names, but for sweeps it's often
        # nicer to have deterministic local names like sweep1, sweep2, ...
        sweep_id = getattr(run, "sweep_id", None) or os.getenv("WANDB_SWEEP_ID")
        is_sweep_run = sweep_id is not None

        if is_sweep_run:
            repo_root = _find_repo_root(Path(__file__))
            counter_file = repo_root / f".wandb_sweep_counter_{sweep_id}.txt"
            idx = _next_counter_value(counter_file)
            run.name = f"sweep{idx}"
        else:
            prefix = group or "run"
            run.name = f"{prefix}-{run.id}"

    return True, None
