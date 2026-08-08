"""Educational PyTorch model builder package."""

from .core import (
    ArchitectureAnalysis,
    LayerReport,
    ShapeError,
    analyze_architecture,
    analyze_layers,
    build_torch_model,
    export_architecture_json,
    generate_model_code,
    import_architecture_json,
    validate_forward_pass,
)

__all__ = [
    "ArchitectureAnalysis",
    "LayerReport",
    "ShapeError",
    "analyze_architecture",
    "analyze_layers",
    "build_torch_model",
    "export_architecture_json",
    "generate_model_code",
    "import_architecture_json",
    "validate_forward_pass",
]
