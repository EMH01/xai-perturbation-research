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
    """Collect the same visual-analysis elements produced by the thesis notebook."""
    model.eval()
    mask = model.explainer.mask(images)
    perturbed = images * mask

    # The notebook's save() function recorded the classifier prediction for the
    # original image alongside the relevance map and perturbed image.
    predictions = model.classifier(images).argmax(dim=1)

    return ExplanationBatch(
        original=images,
        mask=mask,
        perturbed=perturbed,
        predictions=predictions,
    )
