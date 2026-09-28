from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    num_classes: int = 5
    image_size: int = 224
    hidden_dim: int = 1200
    batch_size: int = 32
    learning_rate: float = 1e-3
    momentum: float = 0.9
    seed: int = 42
