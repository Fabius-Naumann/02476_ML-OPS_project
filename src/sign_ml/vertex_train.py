from pathlib import Path

import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
import gcsfs

from sign_ml.model import build_model
from sign_ml.vertex_data import load_vertex_data
from sign_ml.utils import device_from_cfg, init_wandb


def upload_to_gcs(local_path: Path, gcs_path: str, project_id: str) -> None:
    print(f"Uploading file to GCS: {gcs_path}")
    fs = gcsfs.GCSFileSystem(project=project_id)
    with local_path.open("rb") as f:
        with fs.open(gcs_path, "wb") as gcs_file:
            gcs_file.write(f.read())
    print(f"Upload finished: {gcs_path}")


@hydra.main(config_path="../../configs", config_name="vertex", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Vertex training job started")

    device = device_from_cfg(cfg.training.device)
    print(f"Using device: {device}")

    train_loader, val_loader = load_vertex_data(cfg)
    print("Loaded training and validation data")

    model = build_model(cfg.model.num_classes).to(device)
    print(f"Model initialized with num_classes={cfg.model.num_classes}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    criterion = nn.CrossEntropyLoss()

    use_wandb, _ = init_wandb(cfg, group="vertex")
    print(f"WandB enabled: {use_wandb}")

    for epoch in range(cfg.training.epochs):
        print(f"Starting epoch {epoch + 1}/{cfg.training.epochs}")

        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = 100.0 * train_correct / train_total

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = 100.0 * val_correct / val_total

        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.2f}% | "
            f"Val Loss={val_loss:.4f} Acc={val_acc:.2f}%"
        )

        if use_wandb:
            import wandb

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                }
            )

    output_dir = Path("/tmp/model")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model output directory created at {output_dir}")

    gcs_base = f"gs://{cfg.gcp.bucket}/models/sign_ml"

  
    local_pt = output_dir / "traffic_sign_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": cfg.model.num_classes,
        },
        local_pt,
    )
    print(f"PT model saved locally at {local_pt}")
    upload_to_gcs(local_pt, f"{gcs_base}/traffic_sign_model.pt", cfg.gcp.project_id)
    print(f"Uploaded PT model to {gcs_base}/traffic_sign_model.pt")

    # existing PKL (still kept, but optional)
    local_pkl = output_dir / "model.pkl"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": cfg.model.num_classes,
        },
        local_pkl,
    )
    upload_to_gcs(local_pkl, f"{gcs_base}/model.pkl", cfg.gcp.project_id)
    print(f"Uploaded PKL model to {gcs_base}/model.pkl")

    dummy_input = torch.randn(
        1,
        3,
        cfg.model.image_size,
        cfg.model.image_size,
        device=device,
    )
    print(f"Dummy input created with shape {dummy_input.shape}")

    local_onnx = output_dir / "model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        local_onnx,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
    )
    print(f"ONNX model exported locally at {local_onnx}")

    upload_to_gcs(local_onnx, f"{gcs_base}/model.onnx", cfg.gcp.project_id)
    print(f"Uploaded ONNX model to {gcs_base}/model.onnx")

    print("Vertex training job completed successfully")


if __name__ == "__main__":
    main()
