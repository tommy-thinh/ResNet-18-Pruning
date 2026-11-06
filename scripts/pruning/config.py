"""The reference training and small, explicit inference budgets."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 16
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.005
    use_ema: bool = True
    early_stopping_patience: int = 20
    lr_patience: int = 10
    lr_factor: float = 0.5
    min_lr: float = 1e-5

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("Epochs and batch size must be positive.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Weight decay must be finite and nonnegative.")
        if self.early_stopping_patience < 0 or self.lr_patience < 0:
            raise ValueError("Patience cannot be negative.")
        if not 0 < self.lr_factor < 1 or not 0 <= self.momentum <= 1:
            raise ValueError("Invalid LR reduction factor or momentum.")


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_size: int = 16
    warmup: int = 10
    repeats: int = 50
    device: str = "/CPU:0"

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.warmup < 1 or self.repeats < 2:
            raise ValueError("Use batch_size >= 1, warmup >= 1, repeats >= 2.")
