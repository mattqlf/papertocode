import torch as tc
from torch import nn, optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.cuda import amp
from tqdm import tqdm
from typing import Callable
from config import CONFIG


def train_val(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    metric: Callable[[tc.Tensor, tc.Tensor], float],
    optimizer: optim.Optimizer,
    scaler: amp.GradScaler,
    epochs: int = 20,
    scheduler: _LRScheduler = None,
    device: str = "cpu",
) -> list[tuple[float, float, float, float]]:
    """Training and validation function for a PyTorch model

    Args:
        model: PyTorch model to be trained
        train_loader: Dataloader for training dataset that returns (image, label) such that the shape of image and label are (N, C, H, W)
        val_loader: Dataloader for training dataset that returns (image, label) such that the shape of image and label are (N, C, H, W)
        loss_fn: Loss function used for training
        metric: Evaluation metric function
        optimizer: Optimizer used for training
        scaler: Gradient scaler for mixed precision training
        epochs: How epochs to train for
        scheduler: Learning rate schedule
        device: Device to run training on

    Returns:
        List of tuples in the form of (average training loss, average validation loss, average training score, average validation score) with length equal to number of epochs
    """
    model.to(device)
    results = []

    for epoch in range(epochs):
        print(f"---------------Epoch {epoch + 1}---------------")

        # Training
        model.train()
        avg_train_loss, avg_train_score = 0.0, 0.0
        train_progress = tqdm(enumerate(train_loader), total = len(train_loader))

        for i, (images, labels) in train_progress:
            images, labels = images.to(device), labels.to(device)

            with amp.autocast():
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            score = metric(outputs, labels)

            avg_train_loss = (avg_train_loss * i + loss.item()) / (i + 1)
            avg_train_score = (avg_train_score * i + score.item()) / (i + 1)
            train_progress.set_description(
                f"Average Train Loss: {avg_train_loss} and Average Train Score {avg_train_score}"
            )

        if scheduler:
            scheduler.step()

        # Validation
        model.eval()
        avg_val_loss, avg_val_score = 0.0, 0.0
        val_progress = tqdm(enumerate(val_loader), total = len(val_loader))

        with tc.no_grad():
            for i, (images, labels) in val_progress:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                score = metric(outputs, labels)

                avg_val_loss = (avg_val_loss * i + loss.item()) / (i + 1)
                avg_val_score = (avg_val_score * i + score.item()) / (i + 1)

        results.append((avg_train_loss, avg_val_loss, avg_train_score, avg_val_score))
        tc.save(model.state_dict(), f"./checkpoints/{CONFIG.MODEL}_{epoch + 1}.pth.tar")

    return results
