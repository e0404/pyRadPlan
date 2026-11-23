"""The TG119 pipeline gives the same result with the shim as with real SimpleITK (natively)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PIPELINE = Path(__file__).resolve().parents[1] / "pipeline.py"


def _run(tmp_path, mode, shim):
    out = tmp_path / f"{mode}_{'shim' if shim else 'sitk'}.npz"
    cmd = [sys.executable, str(PIPELINE), "--mode", mode, "--preset", "fast"]
    cmd += ["--max-iter", "30", "--out", str(out)]
    if shim:
        cmd.append("--sitk-shim")
    subprocess.run(cmd, check=True, capture_output=True)
    return np.load(out)


@pytest.mark.parametrize("mode", ["photons", "protons"])
def test_pipeline_identical_with_shim(tmp_path, mode):
    pytest.importorskip("SimpleITK")
    pytest.importorskip("pyRadPlan")
    real = _run(tmp_path, mode, shim=False)
    shim = _run(tmp_path, mode, shim=True)
    assert (
        json.loads(str(shim["info"]))["num_bixels"] == json.loads(str(real["info"]))["num_bixels"]
    )
    for key in ("ray_positions", "bixel_sums", "unit_dose", "weights", "dose"):
        np.testing.assert_array_equal(shim[key], real[key], err_msg=key)
