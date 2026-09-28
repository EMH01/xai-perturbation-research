from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_state_dict(
    model: nn.Module,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": metadata or {},
        },
        target,
    )


def load_state_dict(
    model: nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return dict(checkpoint.get("metadata", {}))
