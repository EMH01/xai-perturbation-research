import torch
from torch import nn

from xai_perturbation.models import DensePixelMaskExplainer, PerturbationClassifier


def test_dense_explainer_produces_bounded_mask_and_same_shape():
    explainer = DensePixelMaskExplainer(image_size=4, channels=3, hidden_dim=8)
    images = torch.rand(2, 3, 4, 4)

    mask = explainer.mask(images)
    perturbed = explainer(images)

    assert mask.shape == images.shape
    assert perturbed.shape == images.shape
    assert torch.all(mask >= 0)
    assert torch.all(mask <= 1)


def test_classifier_remains_frozen_and_in_eval_mode():
    explainer = DensePixelMaskExplainer(image_size=4, channels=3, hidden_dim=8)
    classifier = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    model = PerturbationClassifier(explainer, classifier)

    model.train()

    assert model.training
    assert not model.classifier.training
    assert all(not parameter.requires_grad for parameter in model.classifier.parameters())
    assert all(parameter.requires_grad for parameter in model.explainer.parameters())
