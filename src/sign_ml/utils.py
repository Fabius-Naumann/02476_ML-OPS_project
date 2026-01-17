import os
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import IO, Any, cast

import torch
from omegaconf import DictConfig, OmegaConf


BASE_DIR = Path(__file__).resolve().parent.parent.parent


def _bool_from_cfg(cfg: DictConfig, key: str, default: bool = False) -> bool:
    """Return a boolean from config, accepting several truthy strings.

    Args:
        cfg: Hydra configuration object.
        key: Dot-separated key to look up.
        default: Fallback value when the key is missing or invalid.

    Returns:
        Parsed boolean value.
    """

    value = OmegaConf.select(cfg, key, default=default)
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _int_from_cfg(cfg: DictConfig, key: str, default: int) -> int:
    """Return an integer from config, falling back on parse errors.

    Args:
        cfg: Hydra configuration object.
        key: Dot-separated key to look up.
        default: Fallback value when the key is missing or invalid.

    Returns:
        Parsed integer value.
    """

    value = OmegaConf.select(cfg, key, default=default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _torch_profile_dir(cfg: DictConfig) -> Path:
    """Resolve an absolute output directory for torch.profiler traces."""

    out_dir = OmegaConf.select(cfg, "profiling.torch.out_dir", default=None)
    if out_dir is None:
        # Default under project-root ./log so traces stay out of src/
        return BASE_DIR / "log" / "sign_ml" / "profiling" / "torch"
    path = Path(str(out_dir))
    return path if path.is_absolute() else BASE_DIR / path


def _torch_tb_log_dir(cfg: DictConfig) -> Path:
    """Resolve an absolute output directory for TensorBoard profiler logs."""

    out_dir = OmegaConf.select(cfg, "profiling.torch.tensorboard_dir", default=None)
    if out_dir is None:
        # Default to project-root ./log so users can run: tensorboard --logdir=./log
        return BASE_DIR / "log" / "sign_ml"
    path = Path(str(out_dir))
    return path if path.is_absolute() else BASE_DIR / path


def _get_torch_profiler_config(
    cfg: DictConfig,
    device: torch.device,
    *,
    steps: int,
    timestamp: datetime,
    export_tensorboard: bool,
) -> tuple[list[Any], Any, Any, Path, Optional[Path]]:
    """Build shared torch.profiler configuration for training/evaluation.

    Returns the activities list, schedule, on_trace_ready callback, trace directory,
    and optional TensorBoard directory.
    """

    from torch.profiler import ProfilerActivity, schedule as profiler_schedule, tensorboard_trace_handler

    trace_dir = _torch_profile_dir(cfg) / timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    trace_dir.mkdir(parents=True, exist_ok=True)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    tb_dir: Optional[Path] = None
    schedule = None
    on_trace_ready = None
    if export_tensorboard:
        tb_root = _torch_tb_log_dir(cfg)
        tb_dir = tb_root / timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        tb_dir.mkdir(parents=True, exist_ok=True)
        on_trace_ready = tensorboard_trace_handler(str(tb_dir))
        active_steps = max(steps - 1, 1)
        schedule = profiler_schedule(wait=0, warmup=1, active=active_steps, repeat=1)

    return activities, schedule, on_trace_ready, trace_dir, tb_dir


def _is_truthy_env(var_name: str) -> bool:
    """Return True when an environment variable is set to a truthy value."""
    value = os.getenv(var_name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _is_wandb_disabled(cfg: Optional[DictConfig] = None) -> bool:
    """Return True when W&B should be disabled.

    Disabling can be controlled via environment variables or via Hydra config.
    By default, W&B is disabled unless explicitly enabled in config.
    """
    if _is_truthy_env("WANDB_DISABLED"):
        return True

    mode = os.getenv("WANDB_MODE")
    if mode is not None and mode.strip().lower() == "disabled":
        return True

    is_sweep_agent_run = bool(os.getenv("WANDB_SWEEP_ID"))

    if cfg is None:
        # For sweeps, we default to enabling W&B so the sweep UI receives metrics.
        return not is_sweep_agent_run

    enabled_root = OmegaConf.select(cfg, "wandb.enabled", default=None)
    enabled_experiment = OmegaConf.select(cfg, "experiment.wandb.enabled", default=None)

    if enabled_root is None and enabled_experiment is None:
        # No explicit config: for sweeps, enable; otherwise default to disabled.
        return not is_sweep_agent_run
    if enabled_root is False or enabled_experiment is False:
        return True

    return False


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

    if _is_wandb_disabled(cfg):
        return False, None

    try:
        import wandb  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return False, exc

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
