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

    # Reproduce the notebook's save() function: the stored relevance vector was
    # computed directly from fc1 -> fc2 -> sigmoid, while the perturbed image was
    # produced through ExplainerNetwork.forward(), which includes ReLU after fc1.
    flat = images.flatten(start_dim=1)
    fc1_output = model.explainer.fc1(flat)
    mask = torch.sigmoid(model.explainer.fc2(fc1_output)).view_as(images)

    perturbed = model.explainer(images)

    # The notebook recorded the classifier prediction for the original image
    # alongside the relevance map and perturbed image.
    predictions = model.classifier(images).argmax(dim=1)

    return ExplanationBatch(
        original=images,
        mask=mask,
        perturbed=perturbed,
        predictions=predictions,
    )
