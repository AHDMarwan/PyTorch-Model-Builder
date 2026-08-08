"""Small, deterministic training utilities for AI Lab Junior.

The datasets are generated locally with PyTorch. This keeps the classroom demo
fast, private, and independent from external dataset downloads while students
learn the training loop itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Any

import torch
import torch.nn as nn

from .core import build_torch_model


@dataclass
class DatasetBundle:
    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    description: str


@dataclass
class TrainingResult:
    losses: tuple[float, ...]
    train_accuracies: tuple[float, ...]
    test_accuracies: tuple[float, ...]
    final_train_accuracy: float
    final_test_accuracy: float
    confusion_matrix: torch.Tensor
    state_dict: dict[str, torch.Tensor]


def _split_dataset(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int,
    test_fraction: float = 0.2,
) -> DatasetBundle:
    generator = torch.Generator().manual_seed(seed + 101)
    order = torch.randperm(len(x), generator=generator)
    x = x[order]
    y = y[order]
    test_count = max(1, int(len(x) * test_fraction))
    train_count = len(x) - test_count
    if train_count < 1:
        raise ValueError("Not enough samples to create a training split.")
    return DatasetBundle(
        train_x=x[:train_count],
        train_y=y[:train_count],
        test_x=x[train_count:],
        test_y=y[train_count:],
        description="",
    )


def make_vector_dataset(
    features: int,
    classes: int,
    *,
    samples: int = 300,
    noise: float = 0.45,
    seed: int = 42,
    kind: str = "clusters",
) -> DatasetBundle:
    """Generate a classification dataset that matches a vector model input."""

    features = int(features)
    classes = int(classes)
    samples = int(samples)
    if features < 1:
        raise ValueError("features must be at least 1")
    if classes < 2:
        raise ValueError("classes must be at least 2")
    if samples < classes * 4:
        raise ValueError("Use at least four samples per class.")

    generator = torch.Generator().manual_seed(seed)

    if kind == "xor":
        if features != 2 or classes != 2:
            raise ValueError("XOR requires exactly 2 input features and 2 classes.")
        signs = torch.randint(0, 2, (samples, 2), generator=generator).float() * 2.0 - 1.0
        x = signs + float(noise) * torch.randn(samples, 2, generator=generator)
        y = (signs[:, 0] * signs[:, 1] < 0).long()
        bundle = _split_dataset(x, y, seed=seed)
        bundle.description = "XOR: four groups that need a non-linear boundary."
        return bundle

    labels = torch.arange(samples, dtype=torch.long) % classes
    centers = torch.zeros((classes, features), dtype=torch.float32)

    if features == 1:
        centers[:, 0] = torch.linspace(-2.5, 2.5, classes)
    else:
        angles = torch.arange(classes, dtype=torch.float32) * (2.0 * math.pi / classes)
        centers[:, 0] = 2.5 * torch.cos(angles)
        centers[:, 1] = 2.5 * torch.sin(angles)
        if features > 2:
            centers[:, 2:] = 0.4 * torch.randn(classes, features - 2, generator=generator)

    x = centers[labels] + float(noise) * torch.randn(samples, features, generator=generator)
    bundle = _split_dataset(x, labels, seed=seed)
    bundle.description = "Clusters: each class forms a small group of similar points."
    return bundle


def make_image_dataset(
    input_shape: Iterable[int],
    classes: int,
    *,
    samples: int = 240,
    noise: float = 0.18,
    seed: int = 42,
) -> DatasetBundle:
    """Generate simple class-specific bright patterns for a CNN classroom demo."""

    channels, height, width = (int(v) for v in input_shape)
    classes = int(classes)
    samples = int(samples)
    if channels < 1 or height < 4 or width < 4:
        raise ValueError("Image input must be channels × height × width with size at least 4×4.")
    if classes < 2:
        raise ValueError("classes must be at least 2")
    if samples < classes * 4:
        raise ValueError("Use at least four samples per class.")

    generator = torch.Generator().manual_seed(seed)
    y = torch.arange(samples, dtype=torch.long) % classes
    x = float(noise) * torch.randn(samples, channels, height, width, generator=generator)

    grid = int(math.ceil(math.sqrt(classes)))
    half = max(1, min(height, width) // 12)
    for index in range(samples):
        label = int(y[index].item())
        row = label // grid
        col = label % grid
        cy = int((row + 0.5) * height / grid)
        cx = int((col + 0.5) * width / grid)
        y0, y1 = max(0, cy - half), min(height, cy + half + 1)
        x0, x1 = max(0, cx - half), min(width, cx + half + 1)
        channel = label % channels
        x[index, channel, y0:y1, x0:x1] += 2.5

    bundle = _split_dataset(x, y, seed=seed)
    bundle.description = "Synthetic images: each class contains a bright pattern in a different region."
    return bundle


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean().item())


def train_classifier(
    layers: Iterable[dict[str, Any]],
    input_shape: Iterable[int],
    output_size: int,
    dataset: DatasetBundle,
    *,
    epochs: int = 50,
    learning_rate: float = 0.01,
    seed: int = 42,
) -> TrainingResult:
    """Train the student's current architecture on a small generated dataset."""

    epochs = int(epochs)
    if epochs < 1 or epochs > 300:
        raise ValueError("epochs must be between 1 and 300")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    torch.manual_seed(seed)
    model = build_torch_model(layers, input_shape, int(output_size))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))

    losses: list[float] = []
    train_accuracies: list[float] = []
    test_accuracies: list[float] = []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(dataset.train_x)
        loss = criterion(logits, dataset.train_y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_logits = model(dataset.train_x)
            test_logits = model(dataset.test_x)
            losses.append(float(loss.item()))
            train_accuracies.append(accuracy_from_logits(train_logits, dataset.train_y))
            test_accuracies.append(accuracy_from_logits(test_logits, dataset.test_y))

    with torch.no_grad():
        test_logits = model(dataset.test_x)
        predictions = test_logits.argmax(dim=1)

    confusion = torch.zeros((int(output_size), int(output_size)), dtype=torch.int64)
    for target, prediction in zip(dataset.test_y, predictions):
        confusion[int(target.item()), int(prediction.item())] += 1

    state_dict = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return TrainingResult(
        losses=tuple(losses),
        train_accuracies=tuple(train_accuracies),
        test_accuracies=tuple(test_accuracies),
        final_train_accuracy=train_accuracies[-1],
        final_test_accuracy=test_accuracies[-1],
        confusion_matrix=confusion,
        state_dict=state_dict,
    )
