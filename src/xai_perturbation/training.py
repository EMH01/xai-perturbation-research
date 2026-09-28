from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from .models import PerturbationClassifier


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    loss: float
    accuracy: float


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += logits.argmax(dim=1).eq(labels).sum().item()
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Cannot train on an empty dataset.")
    return total_loss / total_examples, total_correct / total_examples


def train_classifier(
    model: nn.Module,
    loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    momentum: float,
    device: torch.device,
) -> list[EpochMetrics]:
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
    )
    criterion = nn.CrossEntropyLoss()
    history: list[EpochMetrics] = []

    for epoch in range(1, epochs + 1):
        loss, accuracy = _train_epoch(model, loader, optimizer, criterion, device)
        history.append(EpochMetrics(epoch, loss, accuracy))
    return history


def train_explainer(
    model: PerturbationClassifier,
    loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    momentum: float,
    device: torch.device,
) -> list[EpochMetrics]:
    model.to(device)
    model.freeze_classifier()
    optimizer = torch.optim.SGD(
        model.explainer.parameters(),
        lr=learning_rate,
        momentum=momentum,
    )
    criterion = nn.CrossEntropyLoss()
    history: list[EpochMetrics] = []

    for epoch in range(1, epochs + 1):
        loss, accuracy = _train_epoch(model, loader, optimizer, criterion, device)
        history.append(EpochMetrics(epoch, loss, accuracy))
    return history
