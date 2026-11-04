"""The download of matRad's phantoms for the installers and the container images."""

import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
BRIEFCASE = ROOT / "packaging" / "briefcase"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fetcher(tmp_path, monkeypatch):
    """The script, with its files served from a local mirror of two fake matRad files."""
    script = _load("fetch_matrad_phantoms", BRIEFCASE / "scripts" / "fetch_matrad_phantoms.py")
    mirror = tmp_path / "mirror"
    contents = {"matRad/phantoms/BOX.mat": b"box phantom", "LICENSE.md": b"BSD-3-Clause"}
    files = {}
    for source, content in contents.items():
        (mirror / source).parent.mkdir(parents=True, exist_ok=True)
        (mirror / source).write_bytes(content)
        target = "phantoms/BOX.mat" if source.endswith(".mat") else "matRad-LICENSE.md"
        files[source] = (target, hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(script, "FILES", files)
    monkeypatch.setattr(script, "ATTEMPTS", 1)
    return script, mirror.as_uri() + "/"


def test_pinned_files_are_phantoms_and_the_license():
    script = _load("fetch_matrad_phantoms", BRIEFCASE / "scripts" / "fetch_matrad_phantoms.py")
    targets = [target for target, _ in script.FILES.values()]

    assert "matRad-LICENSE.md" in targets
    assert {Path(t).name for t in targets if t.startswith("phantoms/")} == {
        "BOXPHANTOM.mat",
        "HEAD_AND_NECK.mat",
        "LIVER.mat",
        "PROSTATE.mat",
    }
    for _, sha256 in script.FILES.values():
        assert len(sha256) == 64 and int(sha256, 16) >= 0
    assert script.MATRAD_RELEASE in script.BASE_URL


def test_fetch_downloads_checks_and_keeps_files(fetcher, tmp_path):
    script, base_url = fetcher
    folder = tmp_path / "data"

    assert script.main([str(folder), "--base-url", base_url]) == 0
    assert (folder / "phantoms" / "BOX.mat").read_bytes() == b"box phantom"
    assert (folder / "matRad-LICENSE.md").read_bytes() == b"BSD-3-Clause"
    assert "CORT dataset" in (folder / "README.md").read_text("utf-8")
    assert not list(folder.rglob("*.part"))

    # Files in place with the right hash are not downloaded again, a damaged one is.
    (folder / "phantoms" / "BOX.mat").write_bytes(b"damaged")
    (tmp_path / "mirror" / "LICENSE.md").unlink()
    assert script.main([str(folder), "--base-url", base_url]) == 0
    assert (folder / "phantoms" / "BOX.mat").read_bytes() == b"box phantom"


def test_fetch_rejects_a_file_with_another_hash(fetcher, tmp_path, capsys):
    script, base_url = fetcher
    (tmp_path / "mirror" / "matRad" / "phantoms" / "BOX.mat").write_bytes(b"something else")
    folder = tmp_path / "data"

    assert script.main([str(folder), "--base-url", base_url]) == 1
    assert "expected" in capsys.readouterr().err
    assert not (folder / "phantoms" / "BOX.mat").exists()
    assert not list(folder.rglob("*.part"))


def test_fetch_reports_a_missing_file(fetcher, tmp_path, capsys):
    script, base_url = fetcher
    (tmp_path / "mirror" / "LICENSE.md").unlink()

    assert script.main([str(tmp_path / "data"), "--base-url", base_url]) == 1
    assert "Download of" in capsys.readouterr().err


def test_app_searches_its_data_after_the_configured_folders(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from pyRadPlan import settings

    monkeypatch.syspath_prepend(str(BRIEFCASE / "src"))
    monkeypatch.delitem(sys.modules, "pyradplan_app.app", raising=False)
    from pyradplan_app import app

    data = tmp_path / "data"
    monkeypatch.setattr(app, "BASE_DATA", data)
    monkeypatch.setattr(settings, "base_data_path", "/course")

    app.add_bundled_base_data()
    assert settings.base_data_path == "/course"  # without the folder, nothing changes

    data.mkdir()
    app.add_bundled_base_data()
    assert settings.base_data_path == os.pathsep.join(["/course", str(data)])

    monkeypatch.setattr(settings, "base_data_path", "")
    app.add_bundled_base_data()
    assert settings.base_data_path == str(data)
