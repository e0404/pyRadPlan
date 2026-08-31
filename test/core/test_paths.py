"""Tests for the writable data directory resolution."""

from pathlib import Path

from pyRadPlan.core import DEFAULT_DATA_DIR, get_data_dir, get_data_subdir


def test_default_data_dir(monkeypatch):
    monkeypatch.delenv("PYRADPLAN_DATA_DIR", raising=False)
    assert get_data_dir() == DEFAULT_DATA_DIR
    assert DEFAULT_DATA_DIR == Path.home() / ".pyradplan"


def test_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PYRADPLAN_DATA_DIR", str(tmp_path))
    assert get_data_dir() == tmp_path


def test_env_override_expands_user(monkeypatch):
    monkeypatch.setenv("PYRADPLAN_DATA_DIR", "~/some_pyrp_dir")
    assert get_data_dir() == Path.home() / "some_pyrp_dir"


def test_blank_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("PYRADPLAN_DATA_DIR", "   ")
    assert get_data_dir() == DEFAULT_DATA_DIR


def test_get_data_subdir_creates_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("PYRADPLAN_DATA_DIR", str(tmp_path))
    sub = get_data_subdir("ai_models")
    assert sub == tmp_path / "ai_models"
    assert sub.is_dir()
    # second call must not fail on an existing directory
    assert get_data_subdir("ai_models") == sub


def test_get_data_subdir_no_create(tmp_path, monkeypatch):
    monkeypatch.setenv("PYRADPLAN_DATA_DIR", str(tmp_path))
    sub = get_data_subdir("phantoms", create=False)
    assert sub == tmp_path / "phantoms"
    assert not sub.exists()
