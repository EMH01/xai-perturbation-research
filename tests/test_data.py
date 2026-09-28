import torch
from torch import nn
from torch.utils.data import TensorDataset

from xai_perturbation.data import correctly_classified_subset


class ThresholdClassifier(nn.Module):
    def forward(self, images):
        score = images.flatten(start_dim=1).mean(dim=1)
        return torch.stack([1 - score, score], dim=1)


def test_correct_only_filter_is_non_destructive():
    images = torch.tensor([[[[0.0]]], [[[1.0]]], [[[0.9]]], [[[0.1]]]])
    labels = torch.tensor([0, 1, 0, 1])
    dataset = TensorDataset(images, labels)

    subset = correctly_classified_subset(
        ThresholdClassifier(),
        dataset,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert len(dataset) == 4
    assert subset.indices == [0, 1]
