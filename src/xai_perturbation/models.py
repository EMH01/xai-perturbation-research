import torch
from torch import nn
from torchvision.models import VGG19_BN_Weights, vgg19_bn


class DensePixelMaskExplainer(nn.Module):
    """Faithful modernization of the dense explainer used in the original research."""

    def __init__(
        self,
        image_size: int = 224,
        channels: int = 3,
        hidden_dim: int = 1200,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        features = channels * image_size * image_size
        self.fc1 = nn.Linear(features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, features)

    def mask(self, images: torch.Tensor) -> torch.Tensor:
        flat = images.flatten(start_dim=1)
        hidden = torch.relu(self.fc1(flat))
        mask = torch.sigmoid(self.fc2(hidden))
        return mask.view(
            -1,
            self.channels,
            self.image_size,
            self.image_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images * self.mask(images)


class PerturbationClassifier(nn.Module):
    """Apply a learned perturbation before a frozen image classifier."""

    def __init__(self, explainer: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.explainer = explainer
        self.classifier = classifier
        self.freeze_classifier()

    def freeze_classifier(self) -> None:
        self.classifier.requires_grad_(False)
        self.classifier.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.classifier.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        perturbed = self.explainer(images)
        return self.classifier(perturbed)


def build_vgg19_classifier(
    num_classes: int,
    *,
    pretrained: bool = True,
) -> nn.Module:
    weights = VGG19_BN_Weights.DEFAULT if pretrained else None
    model = vgg19_bn(weights=weights)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model
