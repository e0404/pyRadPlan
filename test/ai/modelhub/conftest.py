from pathlib import Path

import pytest

from pyRadPlan._settings import get_settings
from pyRadPlan.ai.modelhub import _resolve


@pytest.fixture
def dummy_model_dir():
    """Path to the committed dummy model folder used for local-load tests."""
    return str(Path(__file__).resolve().parents[2] / "data" / "ai_models" / "dummy_model")


@pytest.fixture(autouse=True)
def isolated_model_settings(tmp_path, monkeypatch):
    """Keep the suite off the developer's real data root, environment and the Hub.

    Without this, tests that exercise the defaults read ``~/.pyradplan`` (where
    the trust records live, too) and whatever ``PYRADPLAN_*`` variables happen
    to be set, so their outcome depends on the machine they run on. Tests that
    download replace ``snapshot_download`` by a fake of their own.
    """
    for var in (
        "PYRADPLAN_DATA_DIR",
        "PYRADPLAN_AI_MODELHUB_LOCAL_MODELS_DIR",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PYRADPLAN_DATA_DIR", str(tmp_path / "data_root"))

    cfg = get_settings().ai
    monkeypatch.setattr(cfg, "modelhub_local_models_dir", tmp_path / "data_root" / "ai_models")
    monkeypatch.setattr(cfg, "modelhub_cache_dir", None)
    monkeypatch.setattr(cfg, "modelhub_offline", False)
    monkeypatch.setattr(cfg, "modelhub_trust_remote_code", False)

    def _no_hub():
        def _snapshot_download(**kwargs):
            raise ConnectionError("the test suite never reaches the HuggingFace Hub")

        return _snapshot_download

    monkeypatch.setattr(_resolve, "_import_snapshot_download", _no_hub)
    return cfg
