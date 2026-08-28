"""End-to-end load of the committed dummy model.

Requires torch and safetensors (the ``ai`` extra plus a torch build). The
loader machinery itself is covered without them in
``test_loader_machinery.py``.
"""

import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from pyRadPlan.ai.modelhub import BasePreprocessor, load_model  # noqa: E402


def test_load_dummy_model_returns_model_and_preprocessor(dummy_model_dir):
    model, preprocessor = load_model(local_dir=dummy_model_dir)

    assert isinstance(model, torch.nn.Module)
    assert isinstance(preprocessor, BasePreprocessor)
    assert model.training is False


def test_dummy_model_forward_pass(dummy_model_dir):
    model, preprocessor = load_model(local_dir=dummy_model_dir)

    inputs = {name: np.zeros((8, 8, 8)) for name in ("dose", "ct", "mask")}
    x = preprocessor(inputs)
    assert tuple(x.shape) == (1, 3, 8, 8, 8)

    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (1, 1)


def test_loaded_model_survives_a_round_trip(dummy_model_dir):
    """__module__ must stay resolvable, or pickling and torch.save break."""
    import pickle

    model, _ = load_model(local_dir=dummy_model_dir)

    assert type(model).__module__ in sys.modules
    restored = pickle.loads(pickle.dumps(model))
    assert type(restored) is type(model)


def test_weights_are_loaded_not_freshly_initialised(dummy_model_dir):
    model, _ = load_model(local_dir=dummy_model_dir)
    weight = model.head.weight
    assert torch.isfinite(weight).all()
    assert not torch.equal(weight, torch.zeros_like(weight))
