from pathlib import Path
from typing import Any, Dict, Optional

import os
import torch
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf


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


def get_wandb_init_kwargs(cfg: DictConfig, run_name: Optional[str] = None) -> Dict[str, Any]:
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

    return {
        "project": wandb_project,
        "entity": wandb_entity,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "name": run_name,
    }


def init_wandb(cfg: DictConfig, run_name: Optional[str] = None) -> tuple[bool, Optional[Exception]]:
    # Initialize Weights & Biases for experiment tracking

    load_dotenv()

    try:
        wandb.init(**get_wandb_init_kwargs(cfg, run_name))
    except Exception as exc:  # noqa: BLE001
        return False, exc

    return True, None
