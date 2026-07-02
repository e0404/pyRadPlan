import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from pyRadPlan.ml import BasePreprocessor, load_model  # noqa: E402


def test_load_example_model_returns_model_and_preprocessor(example_model_dir):
    model, preprocessor = load_model(local_dir=example_model_dir)

    assert isinstance(model, torch.nn.Module)
    assert isinstance(preprocessor, BasePreprocessor)
    assert model.training is False


def test_example_model_forward_pass(example_model_dir):
    model, preprocessor = load_model(local_dir=example_model_dir)

    inputs = {name: np.zeros((8, 8, 8)) for name in ("dose", "ct", "mask")}
    x = preprocessor(inputs)
    assert tuple(x.shape) == (1, 3, 8, 8, 8)

    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (1, 1)


def test_trust_remote_code_disabled_raises(example_model_dir):
    with pytest.raises(ValueError):
        load_model(local_dir=example_model_dir, trust_remote_code=False)
