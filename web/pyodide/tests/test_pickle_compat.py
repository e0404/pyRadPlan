"""Pickles move between real SimpleITK and the shim in both directions.

Each side runs in its own interpreter, because both must be importable as ``SimpleITK``.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

WORKER = Path(__file__).resolve().with_name("_pickle_worker.py")
PROTOCOLS = (2, 4, 5)


@pytest.fixture(scope="module", autouse=True)
def _needs_sitk():
    pytest.importorskip("SimpleITK")


def run(runtime, action, *paths):
    subprocess.run(
        [sys.executable, str(WORKER), runtime, action, *map(str, paths)],
        check=True,
        capture_output=True,
    )


def load_description(path):
    data = np.load(path)
    return {k: data[k] for k in data.files if k != "meta"}, json.loads(str(data["meta"]))


def assert_same(a, b):
    arrays_a, meta_a = a
    arrays_b, meta_b = b
    assert meta_a == meta_b
    assert arrays_a.keys() == arrays_b.keys()
    for key in arrays_a:
        assert arrays_a[key].dtype == arrays_b[key].dtype, key
        np.testing.assert_array_equal(arrays_a[key], arrays_b[key], err_msg=key)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_image_pickles_cross_runtime(tmp_path, protocol):
    run("real", "make-images", tmp_path / "real.pkl")
    run("shim", "make-images", tmp_path / "shim.pkl")
    results = {}
    for maker in ("real", "shim"):
        for loader in ("real", "shim"):
            out = tmp_path / f"{maker}_to_{loader}.npz"
            run(loader, "load-images", tmp_path / f"{maker}.pkl.p{protocol}", out)
            results[maker, loader] = load_description(out)
    reference = results["real", "real"]
    assert reference[1]["dose"]["metadata"] == {"unit": "Gy"}
    for value in results.values():
        assert_same(reference, value)


def test_shim_pickle_refers_to_simpleitk_module(tmp_path):
    run("shim", "make-images", tmp_path / "shim.pkl")
    data = (tmp_path / "shim.pkl.p4").read_bytes()
    assert b"SimpleITK.SimpleITK" in data
    assert b"_image" not in data


def test_pyradplan_pickle_cross_runtime(tmp_path):
    pytest.importorskip("pyRadPlan")
    run("real", "save-tg119", tmp_path / "real.pkl")
    run("shim", "save-tg119", tmp_path / "shim.pkl")
    results = {}
    for maker in ("real", "shim"):
        for loader in ("real", "shim"):
            out = tmp_path / f"{maker}_to_{loader}.npz"
            run(loader, "load-tg119", tmp_path / f"{maker}.pkl", out)
            results[maker, loader] = load_description(out)
    reference = results["real", "real"]
    assert reference[1]["ct_type"] == "CT"
    assert len(reference[1]["voi_types"]) == 3
    for value in results.values():
        assert_same(reference, value)
