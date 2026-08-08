"""Core architecture logic for the educational PyTorch Model Builder.

The Streamlit UI intentionally stays thin. All tensor-shape reasoning,
validation, parameter counting and code generation live here so they can be
tested independently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import prod
from typing import Any, Iterable

import torch
import torch.nn as nn


SUPPORTED_ACTIVATIONS = ("ReLU", "Sigmoid", "Tanh", "GELU", "None")
SUPPORTED_LAYER_TYPES = ("linear", "conv2d", "maxpool2d", "dropout", "flatten")


class ShapeError(ValueError):
    """Raised when a layer cannot accept the tensor produced before it."""


@dataclass(frozen=True)
class LayerReport:
    index: int
    kind: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    parameters: int
    is_output: bool = False


@dataclass(frozen=True)
class LayerStackAnalysis:
    input_shape: tuple[int, ...]
    reports: tuple[LayerReport, ...]
    total_parameters: int
    output_shape: tuple[int, ...]


@dataclass(frozen=True)
class ArchitectureAnalysis:
    input_shape: tuple[int, ...]
    reports: tuple[LayerReport, ...]
    total_parameters: int
    output_shape: tuple[int, ...]


def _positive_int(value: Any, label: str) -> int:
    value = int(value)
    if value <= 0:
        raise ShapeError(f"{label} must be greater than 0.")
    return value


def _conv_out_dim(size: int, kernel: int, stride: int, padding: int) -> int:
    output = (size + 2 * padding - kernel) // stride + 1
    if output <= 0:
        raise ShapeError(
            "Kernel/stride/padding make the spatial size zero or negative. "
            "Use a smaller kernel or more padding."
        )
    return output


def _activation_module(name: str) -> nn.Module | None:
    if name == "None":
        return None
    if name not in SUPPORTED_ACTIVATIONS:
        raise ShapeError(f"Unsupported activation: {name}")
    return getattr(nn, name)()


def _apply_layer(
    layer: dict[str, Any], shape: tuple[int, ...], index: int
) -> tuple[tuple[int, ...], int]:
    kind = layer.get("type")
    params = layer.get("params", {})

    if kind not in SUPPORTED_LAYER_TYPES:
        raise ShapeError(f"Layer {index}: unsupported layer type '{kind}'.")

    if kind == "linear":
        if len(shape) != 1:
            raise ShapeError(
                f"Layer {index}: Fully Connected expects a vector, but received {shape}. "
                "Add Flatten before it."
            )
        out_features = _positive_int(params.get("out_features", 0), "out_features")
        parameters = shape[0] * out_features + out_features
        return (out_features,), parameters

    if kind == "conv2d":
        if len(shape) != 3:
            raise ShapeError(
                f"Layer {index}: Conv2D expects an image (channels, height, width), "
                f"but received {shape}."
            )
        in_channels, height, width = shape
        out_channels = _positive_int(params.get("out_channels", 0), "out_channels")
        kernel = _positive_int(params.get("kernel_size", 0), "kernel_size")
        stride = _positive_int(params.get("stride", 0), "stride")
        padding = int(params.get("padding", 0))
        if padding < 0:
            raise ShapeError("padding cannot be negative.")
        out_h = _conv_out_dim(height, kernel, stride, padding)
        out_w = _conv_out_dim(width, kernel, stride, padding)
        parameters = out_channels * in_channels * kernel * kernel + out_channels
        return (out_channels, out_h, out_w), parameters

    if kind == "maxpool2d":
        if len(shape) != 3:
            raise ShapeError(
                f"Layer {index}: MaxPool2D expects an image tensor, but received {shape}."
            )
        channels, height, width = shape
        kernel = _positive_int(params.get("kernel_size", 0), "kernel_size")
        stride = _positive_int(params.get("stride", kernel), "stride")
        out_h = _conv_out_dim(height, kernel, stride, 0)
        out_w = _conv_out_dim(width, kernel, stride, 0)
        return (channels, out_h, out_w), 0

    if kind == "dropout":
        probability = float(params.get("p", 0.5))
        if not 0.0 <= probability <= 1.0:
            raise ShapeError("Dropout probability must be between 0 and 1.")
        return shape, 0

    if kind == "flatten":
        if len(shape) == 1:
            return shape, 0
        return (prod(shape),), 0

    raise AssertionError("Unreachable layer type")


def analyze_layers(
    layers: Iterable[dict[str, Any]], input_shape: Iterable[int]
) -> LayerStackAnalysis:
    """Validate user-added layers without requiring a classifier output yet."""

    shape = tuple(_positive_int(v, "input dimension") for v in input_shape)
    if len(shape) not in (1, 3):
        raise ShapeError("Input must be a vector (features,) or image (channels, height, width).")

    reports: list[LayerReport] = []
    total_parameters = 0
    for index, layer in enumerate(layers, start=1):
        input_to_layer = shape
        shape, parameters = _apply_layer(layer, shape, index)
        reports.append(
            LayerReport(
                index=index,
                kind=str(layer["type"]),
                input_shape=input_to_layer,
                output_shape=shape,
                parameters=parameters,
            )
        )
        total_parameters += parameters

    return LayerStackAnalysis(
        input_shape=tuple(input_shape),
        reports=tuple(reports),
        total_parameters=total_parameters,
        output_shape=shape,
    )


def analyze_architecture(
    layers: Iterable[dict[str, Any]],
    input_shape: Iterable[int],
    output_size: int,
) -> ArchitectureAnalysis:
    """Validate the architecture and propagate shapes from input to output."""

    layers = list(layers)
    stack = analyze_layers(layers, input_shape)
    shape = stack.output_shape
    output_size = _positive_int(output_size, "output_size")
    reports = list(stack.reports)
    total_parameters = stack.total_parameters

    if len(shape) != 1:
        raise ShapeError(
            f"The classifier output expects a vector, but the current shape is {shape}. "
            "Add a Flatten layer before the output."
        )

    output_parameters = shape[0] * output_size + output_size
    reports.append(
        LayerReport(
            index=len(reports) + 1,
            kind="output",
            input_shape=shape,
            output_shape=(output_size,),
            parameters=output_parameters,
            is_output=True,
        )
    )
    total_parameters += output_parameters

    return ArchitectureAnalysis(
        input_shape=tuple(input_shape),
        reports=tuple(reports),
        total_parameters=total_parameters,
        output_shape=(output_size,),
    )


def build_torch_model(
    layers: Iterable[dict[str, Any]], input_shape: Iterable[int], output_size: int
) -> nn.Sequential:
    """Build a validated ``nn.Sequential`` model."""

    layers = list(layers)
    input_shape = tuple(int(v) for v in input_shape)
    analyze_architecture(layers, input_shape, output_size)

    modules: list[nn.Module] = []
    shape = input_shape

    for index, layer in enumerate(layers, start=1):
        kind = layer["type"]
        params = layer.get("params", {})

        if kind == "linear":
            modules.append(nn.Linear(shape[0], int(params["out_features"])))
            activation = _activation_module(params.get("activation", "None"))
            if activation is not None:
                modules.append(activation)
        elif kind == "conv2d":
            modules.append(
                nn.Conv2d(
                    shape[0],
                    int(params["out_channels"]),
                    kernel_size=int(params["kernel_size"]),
                    stride=int(params["stride"]),
                    padding=int(params["padding"]),
                )
            )
            activation = _activation_module(params.get("activation", "None"))
            if activation is not None:
                modules.append(activation)
        elif kind == "maxpool2d":
            modules.append(
                nn.MaxPool2d(
                    kernel_size=int(params["kernel_size"]),
                    stride=int(params["stride"]),
                )
            )
        elif kind == "dropout":
            modules.append(nn.Dropout(p=float(params["p"])))
        elif kind == "flatten":
            modules.append(nn.Flatten())

        shape, _ = _apply_layer(layer, shape, index)

    modules.append(nn.Linear(shape[0], int(output_size)))
    return nn.Sequential(*modules)


def validate_forward_pass(
    layers: Iterable[dict[str, Any]], input_shape: Iterable[int], output_size: int
) -> tuple[int, ...]:
    """Run one synthetic sample through the model and return its output shape."""

    model = build_torch_model(layers, input_shape, output_size)
    model.eval()
    sample = torch.zeros((1, *tuple(int(v) for v in input_shape)), dtype=torch.float32)
    with torch.no_grad():
        output = model(sample)
    return tuple(int(v) for v in output.shape)


def generate_model_code(
    layers: Iterable[dict[str, Any]], input_shape: Iterable[int], output_size: int
) -> str:
    """Generate a standalone, readable PyTorch model file."""

    layers = list(layers)
    shape = tuple(int(v) for v in input_shape)
    analyze_architecture(layers, shape, output_size)

    lines = [
        "import torch.nn as nn",
        "",
        "",
        "class StudentModel(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
        "        self.network = nn.Sequential(",
    ]

    indent = " " * 12
    for index, layer in enumerate(layers, start=1):
        kind = layer["type"]
        params = layer.get("params", {})

        if kind == "linear":
            lines.append(f"{indent}nn.Linear({shape[0]}, {int(params['out_features'])}),")
            activation = params.get("activation", "None")
            if activation != "None":
                lines.append(f"{indent}nn.{activation}(),")
        elif kind == "conv2d":
            lines.append(
                f"{indent}nn.Conv2d({shape[0]}, {int(params['out_channels'])}, "
                f"kernel_size={int(params['kernel_size'])}, stride={int(params['stride'])}, "
                f"padding={int(params['padding'])}),"
            )
            activation = params.get("activation", "None")
            if activation != "None":
                lines.append(f"{indent}nn.{activation}(),")
        elif kind == "maxpool2d":
            lines.append(
                f"{indent}nn.MaxPool2d(kernel_size={int(params['kernel_size'])}, "
                f"stride={int(params['stride'])}),"
            )
        elif kind == "dropout":
            lines.append(f"{indent}nn.Dropout(p={float(params['p'])}),")
        elif kind == "flatten":
            lines.append(f"{indent}nn.Flatten(),")

        shape, _ = _apply_layer(layer, shape, index)

    lines.extend(
        [
            f"{indent}nn.Linear({shape[0]}, {int(output_size)}),",
            "        )",
            "",
            "    def forward(self, x):",
            "        return self.network(x)",
            "",
        ]
    )
    return "\n".join(lines)


def export_architecture_json(
    layers: Iterable[dict[str, Any]], input_shape: Iterable[int], output_size: int
) -> str:
    payload = {
        "version": 1,
        "input_shape": list(input_shape),
        "output_size": int(output_size),
        "layers": list(layers),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def import_architecture_json(raw: str) -> dict[str, Any]:
    payload = json.loads(raw)
    if payload.get("version") != 1:
        raise ValueError("Unsupported architecture file version.")
    if not isinstance(payload.get("layers"), list):
        raise ValueError("Architecture file must contain a layers list.")
    input_shape = tuple(int(v) for v in payload["input_shape"])
    output_size = int(payload["output_size"])
    analyze_architecture(payload["layers"], input_shape, output_size)
    return {
        "input_shape": input_shape,
        "output_size": output_size,
        "layers": payload["layers"],
    }


def analysis_as_dict(analysis: ArchitectureAnalysis) -> dict[str, Any]:
    return {
        "input_shape": list(analysis.input_shape),
        "total_parameters": analysis.total_parameters,
        "output_shape": list(analysis.output_shape),
        "reports": [asdict(report) for report in analysis.reports],
    }
