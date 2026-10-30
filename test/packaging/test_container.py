"""Regression checks for the browser container without requiring Docker or Xpra."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def _shell():
    shell = shutil.which("sh")
    if shell is None and os.name == "nt":
        git = shutil.which("git")
        if git:
            candidate = Path(git).resolve().parents[1] / "bin" / "sh.exe"
            if candidate.is_file():
                shell = str(candidate)
    if shell is None or shutil.which("python3") is None:
        pytest.skip("A POSIX shell and python3 are required")
    return shell


@pytest.mark.parametrize(
    "arguments",
    [
        ["/shared/My Case.mat", "$(printf expanded)", "a;b", ""],
        pytest.param(
            ["a'b", 'a"b'],
            marks=pytest.mark.skipif(
                os.name == "nt",
                reason="Git Bash cannot preserve embedded quotes for native Python",
            ),
        ),
    ],
)
def test_entrypoint_preserves_command_arguments(tmp_path, arguments):
    shell = _shell()
    # Emulate Xpra's execution of its single --start-child command string.
    xpra = tmp_path / "xpra"
    xpra.write_text(
        "#!/bin/sh\nfor arg do\n"
        '    case "$arg" in --start-child=*) exec sh -c "${arg#--start-child=}" ;; esac\n'
        "done\nexit 1\n",
        encoding="utf-8",
    )
    xpra.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(tmp_path) + os.pathsep + env["PATH"]
    env["XPRA_PASSWORD"] = "test-password"
    # Prevent Git Bash from turning the Linux data path into a Windows path.
    env["MSYS_NO_PATHCONV"] = "1"
    env["MSYS2_ARG_CONV_EXCL"] = "*"
    entrypoint = Path(__file__).resolve().parents[2] / "packaging/container/entrypoint.sh"
    result = subprocess.run(
        [
            shell,
            str(entrypoint),
            "python3",
            "-c",
            "import json, sys; print(json.dumps(sys.argv[1:]))",
            *arguments,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(result.stdout) == arguments
