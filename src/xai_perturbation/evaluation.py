from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    accuracy: float
    correct: int
    total: int


@torch.inference_mode()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> ClassificationMetrics:
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images).argmax(dim=1)
        correct += predictions.eq(labels).sum().item()
        total += labels.shape[0]

    return ClassificationMetrics(
        accuracy=correct / total if total else 0.0,
        correct=correct,
        total=total,
    )
