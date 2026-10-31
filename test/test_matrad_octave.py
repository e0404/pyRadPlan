import os
import sys
import types

import pytest

from pyRadPlan.matRad import MatRadEngineOctave


def test_default_octave_executable_is_found_on_the_path(tmp_path, monkeypatch):
    """The default used to be a Windows path, so the engine never started elsewhere."""
    if sys.platform == "win32":
        pytest.skip("needs an executable file without extension on the PATH")
    octave = tmp_path / "octave-cli"
    octave.write_text("#!/bin/sh\n")
    octave.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("OCTAVE_EXECUTABLE", raising=False)
    # Stop right after the executable was chosen; no real Octave is started.
    fake = types.SimpleNamespace(Oct2Py=lambda: (_ for _ in ()).throw(RuntimeError("stop")))
    monkeypatch.setitem(sys.modules, "oct2py", fake)

    with pytest.raises(RuntimeError, match="stop"):
        MatRadEngineOctave()
    assert os.environ["OCTAVE_EXECUTABLE"] == str(octave)
