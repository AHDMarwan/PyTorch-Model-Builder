import pytest

from model_builder.core import (
    ShapeError,
    analyze_architecture,
    generate_model_code,
    validate_forward_pass,
)


def test_vector_classifier_shape_and_parameter_count():
    layers = [
        {"type": "linear", "params": {"out_features": 16, "activation": "ReLU"}},
        {"type": "dropout", "params": {"p": 0.2}},
    ]
    analysis = analyze_architecture(layers, (8,), 3)

    assert analysis.output_shape == (3,)
    assert analysis.reports[0].output_shape == (16,)
    assert analysis.total_parameters == (8 * 16 + 16) + (16 * 3 + 3)
    assert validate_forward_pass(layers, (8,), 3) == (1, 3)


def test_cnn_shape_inference_is_correct():
    layers = [
        {
            "type": "conv2d",
            "params": {
                "out_channels": 8,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "activation": "ReLU",
            },
        },
        {"type": "maxpool2d", "params": {"kernel_size": 2, "stride": 2}},
        {"type": "flatten", "params": {}},
        {"type": "linear", "params": {"out_features": 32, "activation": "ReLU"}},
    ]
    analysis = analyze_architecture(layers, (1, 28, 28), 10)

    assert analysis.reports[0].output_shape == (8, 28, 28)
    assert analysis.reports[1].output_shape == (8, 14, 14)
    assert analysis.reports[2].output_shape == (1568,)
    assert analysis.reports[3].output_shape == (32,)
    assert validate_forward_pass(layers, (1, 28, 28), 10) == (1, 10)


def test_invalid_conv_after_linear_is_rejected():
    layers = [
        {"type": "linear", "params": {"out_features": 16, "activation": "ReLU"}},
        {
            "type": "conv2d",
            "params": {
                "out_channels": 4,
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "activation": "ReLU",
            },
        },
    ]

    with pytest.raises(ShapeError, match="Conv2D expects an image"):
        analyze_architecture(layers, (8,), 2)


def test_image_classifier_requires_flatten_before_output():
    with pytest.raises(ShapeError, match="Add a Flatten"):
        analyze_architecture([], (1, 28, 28), 10)


def test_generated_code_compiles():
    layers = [
        {"type": "linear", "params": {"out_features": 12, "activation": "GELU"}},
    ]
    code = generate_model_code(layers, (6,), 4)
    compile(code, "student_model.py", "exec")
    assert "class StudentModel" in code
    assert "nn.Linear(6, 12)" in code


def test_export_import_round_trip():
    from model_builder.core import export_architecture_json, import_architecture_json

    layers = [
        {"type": "linear", "params": {"out_features": 10, "activation": "ReLU"}},
    ]
    raw = export_architecture_json(layers, (4,), 2)
    restored = import_architecture_json(raw)

    assert restored["input_shape"] == (4,)
    assert restored["output_size"] == 2
    assert restored["layers"] == layers


def test_generated_model_executes():
    import torch

    layers = [
        {"type": "linear", "params": {"out_features": 12, "activation": "ReLU"}},
    ]
    namespace = {}
    exec(generate_model_code(layers, (6,), 4), namespace)
    model = namespace["StudentModel"]()
    output = model(torch.zeros(2, 6))
    assert tuple(output.shape) == (2, 4)
