"""The users of the JupyterHub values: make_hub_users.py and the authenticator agree."""

import asyncio
import importlib.util
import os
from pathlib import Path
import stat
import sys
import types

import pytest

CONTAINER = Path(__file__).resolve().parents[2] / "packaging" / "container"


def _make_hub_users():
    spec = importlib.util.spec_from_file_location(
        "make_hub_users", CONTAINER / "make_hub_users.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_make_hub_users_writes_passwords_and_prints_hashes(tmp_path, capsys):
    script = _make_hub_users()
    output = tmp_path / "users.csv"

    assert script.main(["Admin", "anna", "--output", str(output)]) == 0

    rows = [line.split(",") for line in output.read_text("utf-8").splitlines()]
    assert rows[0] == ["user", "password"]
    assert [row[0] for row in rows[1:]] == ["admin", "anna"]
    assert len({row[1] for row in rows[1:]}) == 2
    printed = capsys.readouterr().out
    assert '"admin": pbkdf2_sha256$' in printed
    assert '"anna": pbkdf2_sha256$' in printed
    if os.name == "posix":
        assert stat.S_IMODE(output.stat().st_mode) == 0o600
    # An existing file is never overwritten, so no handed-out password is lost.
    with pytest.raises(SystemExit):
        script.main(["ben", "--output", str(output)])


def test_values_authenticator_checks_the_script_hashes(monkeypatch):
    yaml = pytest.importorskip("yaml")
    traitlets_config = pytest.importorskip("traitlets.config")

    # Stand-in for JupyterHub's Authenticator: configurable, with a logger and lower-case names.
    class Authenticator(traitlets_config.LoggingConfigurable):
        def normalize_username(self, username):
            return username.lower()

    auth_module = types.ModuleType("jupyterhub.auth")
    auth_module.Authenticator = Authenticator
    monkeypatch.setitem(sys.modules, "jupyterhub", types.ModuleType("jupyterhub"))
    monkeypatch.setitem(sys.modules, "jupyterhub.auth", auth_module)

    values = yaml.safe_load((CONTAINER / "jupyterhub-values.yaml").read_text("utf-8"))
    config = traitlets_config.Config()
    exec(values["hub"]["extraConfig"]["passwordList"], {"c": config})
    authenticator_class = config.JupyterHub.authenticator_class

    script = _make_hub_users()
    authenticator = authenticator_class(
        password_hashes={
            "anna": script.password_hash("secret-1", iterations=1000),
            "placeholder": "CHANGE-ME",
        }
    )

    def login(name, password):
        data = {"username": name, "password": password}
        return asyncio.run(authenticator.authenticate(None, data))

    assert login("Anna", "secret-1") == "anna"
    assert login("anna", "secret-2") is None
    assert login("ben", "secret-1") is None
    assert login("placeholder", "CHANGE-ME") is None
