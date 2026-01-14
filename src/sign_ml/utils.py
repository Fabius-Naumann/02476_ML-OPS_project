import torch

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
