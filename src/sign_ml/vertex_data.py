import torch
import gcsfs
from torch.utils.data import Dataset, DataLoader


class VertexTrafficSignsDataset(Dataset):
    def __init__(self, gcs_path: str, project_id: str):
        fs = gcsfs.GCSFileSystem(project=project_id)
        with fs.open(gcs_path, "rb") as f:
            data = torch.load(f, map_location="cpu")

        self.images = data["images"]
        self.targets = data["labels"]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


def load_vertex_data(cfg):
    train_ds = VertexTrafficSignsDataset(
        cfg.data.train_path,
        cfg.gcp.project_id,
    )

    val_ds = VertexTrafficSignsDataset(
        cfg.data.val_path,
        cfg.gcp.project_id,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.device == "cuda",
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.device == "cuda",
    )

    return train_loader, val_loader
