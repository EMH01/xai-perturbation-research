from dataclasses import dataclass

import torch

from .models import PerturbationClassifier


@dataclass(frozen=True, slots=True)
class ExplanationBatch:
    original: torch.Tensor
    mask: torch.Tensor
    perturbed: torch.Tensor
    predictions: torch.Tensor


@torch.inference_mode()
def explain_batch(
    model: PerturbationClassifier,
    images: torch.Tensor,
) -> ExplanationBatch:
    model.eval()
    mask = model.explainer.mask(images)
    perturbed = images * mask
    predictions = model.classifier(perturbed).argmax(dim=1)
    return ExplanationBatch(
        original=images,
        mask=mask,
        perturbed=perturbed,
        predictions=predictions,
    )
