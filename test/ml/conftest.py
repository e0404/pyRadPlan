from pathlib import Path

import pytest


@pytest.fixture
def dummy_model_dir():
    """Path to the committed dummy model folder used for local-load tests."""
    return str(Path(__file__).resolve().parents[1] / "data" / "ml_models" / "dummy_model")
