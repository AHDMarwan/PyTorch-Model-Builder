import torch

from model_builder.training import make_image_dataset, make_vector_dataset, train_classifier


def test_vector_dataset_matches_model_input():
    dataset = make_vector_dataset(2, 2, samples=120, noise=0.25, seed=7)
    assert dataset.train_x.ndim == 2
    assert dataset.train_x.shape[1] == 2
    assert set(dataset.train_y.tolist()).issubset({0, 1})
    assert len(dataset.train_x) + len(dataset.test_x) == 120


def test_image_dataset_matches_cnn_input():
    dataset = make_image_dataset((1, 12, 12), 3, samples=120, noise=0.1, seed=5)
    assert tuple(dataset.train_x.shape[1:]) == (1, 12, 12)
    assert set(dataset.train_y.tolist()).issubset({0, 1, 2})
    assert len(dataset.train_x) + len(dataset.test_x) == 120


def test_training_reduces_loss_on_easy_clusters():
    layers = [
        {"type": "linear", "params": {"out_features": 8, "activation": "ReLU"}},
    ]
    dataset = make_vector_dataset(2, 2, samples=180, noise=0.2, seed=11)
    result = train_classifier(
        layers,
        (2,),
        2,
        dataset,
        epochs=35,
        learning_rate=0.01,
        seed=11,
    )
    assert result.losses[-1] < result.losses[0]
    assert result.final_test_accuracy >= 0.9
    assert tuple(result.confusion_matrix.shape) == (2, 2)
    assert int(result.confusion_matrix.sum().item()) == len(dataset.test_y)
    assert all(torch.isfinite(value).item() for value in result.confusion_matrix.float())
