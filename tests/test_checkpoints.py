import torch
from torch import nn

from xai_perturbation.checkpoints import load_state_dict, save_state_dict


def test_state_dict_round_trip(tmp_path):
    original = nn.Linear(3, 2)
    path = tmp_path / "model.pt"
    save_state_dict(original, path, metadata={"experiment": "test"})

    restored = nn.Linear(3, 2)
    metadata = load_state_dict(restored, path)

    assert metadata == {"experiment": "test"}
    for expected, actual in zip(original.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)
