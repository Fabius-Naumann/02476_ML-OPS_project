import os

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import wandb
from omegaconf import DictConfig, OmegaConf


def _next_counter_value(counter_file: Path) -> int:
    """Return the next integer value for a local counter file.

    The counter is stored on disk so consecutive sweep runs can be named
    deterministically (e.g., sweep1, sweep2, ...).

    Notes:
        This is best-effort and is primarily intended for a single agent running
        locally. If you run multiple agents in parallel, counters may collide.

    Args:
        counter_file: Path to a text file storing the last used counter value.

    Returns:
        The next counter value (starting from 1).
    """

    counter_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = counter_file.read_text(encoding="utf-8").strip()
        last_value = int(raw) if raw else 0
    except FileNotFoundError:
        last_value = 0
    except ValueError:
        last_value = 0

    next_value = last_value + 1
    counter_file.write_text(str(next_value), encoding="utf-8")
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


def get_wandb_init_kwargs(
    cfg: DictConfig, run_name: Optional[str] = None, group: Optional[str] = None
) -> Dict[str, Any]:
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

    kwargs: Dict[str, Any] = {
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


def init_wandb(
    cfg: DictConfig, run_name: Optional[str] = None, group: Optional[str] = None
) -> tuple[bool, Optional[Exception]]:
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
    except Exception as exc:  # noqa: BLE001
        return False, exc

    if run is not None and run_name is None:
        # By default we let W&B generate unique names, but for sweeps it's often
        # nicer to have deterministic local names like sweep1, sweep2, ...
        sweep_id = getattr(run, "sweep_id", None) or os.getenv("WANDB_SWEEP_ID")
        is_sweep_run = sweep_id is not None

        if is_sweep_run:
            repo_root = Path(__file__).resolve().parents[2]
            counter_file = repo_root / f".wandb_sweep_counter_{sweep_id}.txt"
            idx = _next_counter_value(counter_file)
            run.name = f"sweep{idx}"
        else:
            prefix = group or "run"
            run.name = f"{prefix}-{run.id}"

    return True, None
