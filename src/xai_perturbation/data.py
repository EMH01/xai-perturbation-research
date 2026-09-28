from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.models import VGG19_BN_Weights


def legacy_transform(image_size: int = 224) -> transforms.Compose:
    """Preprocessing used by the original research code."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def vgg_transform() -> transforms.Compose:
    """Current preprocessing associated with the pretrained VGG19-BN weights."""
    return VGG19_BN_Weights.DEFAULT.transforms()


def imagefolder_splits(
    root: str | Path,
    *,
    transform,
    split: Sequence[float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    if len(split) != 3 or not torch.isclose(torch.tensor(sum(split)), torch.tensor(1.0)):
        raise ValueError("split must contain three fractions summing to 1.0")

    dataset = datasets.ImageFolder(root=str(root), transform=transform)
    total = len(dataset)
    train_size = int(split[0] * total)
    eval_size = int(split[1] * total)
    test_size = total - train_size - eval_size
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        random_split(
            dataset,
            [train_size, eval_size, test_size],
            generator=generator,
        )
    )


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


@torch.inference_mode()
def correctly_classified_subset(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    batch_size: int,
    device: torch.device,
) -> Subset:
    """Return a non-destructive subset containing only correctly classified examples."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    keep: list[int] = []
    offset = 0

    model.eval()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images).argmax(dim=1)
        matches = predictions.eq(labels).cpu()
        keep.extend(offset + index for index, match in enumerate(matches) if bool(match))
        offset += len(labels)

    return Subset(dataset, keep)
