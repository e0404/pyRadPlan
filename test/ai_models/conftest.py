from pathlib import Path

import pytest

from pyRadPlan._settings import get_settings


@pytest.fixture
def dummy_model_dir():
    """Path to the committed dummy model folder used for local-load tests."""
    return str(Path(__file__).resolve().parents[1] / "data" / "ai_models" / "dummy_model")


@pytest.fixture(autouse=True)
def isolated_model_settings(tmp_path, monkeypatch):
    """Keep the suite off the developer's real data root and environment.

    Without this, tests that exercise the defaults read ``~/.pyradplan`` and
    whatever ``PYRADPLAN_*`` variables happen to be set, so their outcome
    depends on the machine they run on.
    """
    for var in (
        "PYRADPLAN_DATA_DIR",
        "PYRADPLAN_AI_MODELS_LOCAL_MODELS_DIR",
        "PYRADPLAN_HUGGINGFACE_PATH",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PYRADPLAN_DATA_DIR", str(tmp_path / "data_root"))

    cfg = get_settings().ai_models
    monkeypatch.setattr(cfg, "local_models_dir", tmp_path / "data_root" / "ai_models")
    monkeypatch.setattr(cfg, "cache_dir", None)
    monkeypatch.setattr(cfg, "offline", False)
    monkeypatch.setattr(cfg, "trust_remote_code", False)
    return cfg
