import matplotlib.pyplot as plt
import torch
import typer
import wandb
from my_project.data import corrupt_mnist
from my_project.model import MyAwesomeModel
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(lr: float = 0.001, batch_size: int = 32, epochs: int = 5) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")
    # TODO is done,  initialize a wandb run here
    run = wandb.init(
        project="corrupt_mnist",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        preds, targets = [], []
        for i, (img, target) in enumerate(train_dataloader):
            global_step = epoch * len(train_dataloader) + i
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            # TODO is done, here log the loss and accuracy to wandb
            wandb.log(
                {"train_loss": loss.item(), "train_accuracy": accuracy, "epoch": epoch},
                step=global_step,
            )

            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

                # TODO is done, add a plot of the input images
                images = [wandb.Image(im.detach().cpu(), caption=f"epoch={epoch} step={global_step}") for im in img[:5]]
                wandb.log({"images": images}, step=global_step)

                #TODO is done, add a plot of histogram of the gradients
                grads = torch.cat(
                    [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None],
                    0,
                ).cpu()
                wandb.log({"gradients": wandb.Histogram(grads)}, step=global_step)

        # add a custom matplotlib plot of the ROC curves
        preds = torch.cat(preds, 0)
        targets = torch.cat(targets, 0)

        # Convert logits to probabilities for ROC curves
        preds_probs = preds.softmax(dim=1)

        fig, ax = plt.subplots()

        for class_id in range(10):
            one_hot = (targets == class_id).int()
            _ = RocCurveDisplay.from_predictions(
                one_hot.numpy(),
                preds_probs[:, class_id].numpy(),
                name=f"ROC curve for {class_id}",
                plot_chance_level=(class_id == 2),
                ax=ax,
            )
        # TODO is done, roc log the figure to wandb
        wandb.log({"roc": wandb.Image(fig)}, step=(epoch + 1) * len(train_dataloader) - 1)
        plt.close(fig)  # close the plot to avoid memory leaks and overlapping figures

    y_true = targets.numpy()
    y_pred_labels = preds.argmax(dim=1).numpy()
    final_accuracy = accuracy_score(y_true, y_pred_labels)
    final_precision = precision_score(y_true, y_pred_labels, average="weighted", zero_division=0)
    final_recall = recall_score(y_true, y_pred_labels, average="weighted", zero_division=0)
    final_f1 = f1_score(y_true, y_pred_labels, average="weighted", zero_division=0)

    # first we save the model to a file then log it as an artifact
    torch.save(model.state_dict(), "model.pth")
    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="A model trained to classify corrupt MNIST images",
        metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    )
    # TODO is done, add the model file to the artifact and log it
    artifact.add_file("model.pth")
    run.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":
    typer.run(train)