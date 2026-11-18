import os

import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan.app import ViewCache
from pyRadPlan.app._views import beam_geometry


@pytest.fixture(scope="module")
def pipeline(tg119_pipeline):
    client, _, _ = tg119_pipeline
    return client, client.manager.get(client.session)


def _array(response, field="data"):
    descriptor = response["result"][field]
    buffer = response["buffers"][descriptor["$array"]]
    return np.frombuffer(buffer, dtype=descriptor["dtype"]).reshape(descriptor["shape"])


def test_volume_of_a_component_key(pipeline):
    client, session = pipeline
    response = client.ok("view.volume", {"name": "physical_dose_beam 1"})
    assert response["result"]["name"] == "physical_dose_beam 1"
    assert response["result"]["unit"] == "Gy"
    expected = sitk.GetArrayFromImage(session.workspace.result["physical_dose_beam"][1])
    np.testing.assert_allclose(_array(response), expected.astype(np.float32))
    assert response["result"]["range"] == [float(expected.min()), pytest.approx(expected.max())]
    response = client.call("view.volume", {"name": "physical_dose_beam 7"})
    assert response["error"]["type"] == "KeyError"
    response = client.call("view.volume", {"name": "ct", "encoding": "int8"})
    assert response["error"]["type"] == "ValueError"


class _FakeCT:
    def __init__(self, array):
        self.cube_hu = sitk.GetImageFromArray(array)


@pytest.mark.parametrize(
    "values, dtype",
    [
        (np.array([-1000.0, 0.0, 1500.0]), "int16"),
        (np.array([-1000.0, 0.5, 1500.0]), "float32"),
        (np.array([-1000.0, 0.0, 40000.0]), "float32"),
    ],
)
def test_ct_auto_encoding(values, dtype):
    array = np.resize(values, (2, 3, 4)).astype(np.float32)
    ct = _FakeCT(array)
    views = ViewCache({"ct": ct}.get)
    volume = views.volume("ct", encoding="auto")
    assert volume["data"].dtype == np.dtype(dtype)
    np.testing.assert_array_equal(volume["data"].astype(np.float32), array)
    assert volume["range"] == [float(array.min()), float(array.max())]
    assert views.volume("ct")["data"].dtype == np.float32


def _uint8_masks(client):
    response = client.ok("view.structure_masks")
    return {
        mask["name"]: np.frombuffer(
            response["buffers"][mask["data"]["$array"]], dtype=np.uint8
        ).reshape(mask["data"]["shape"])
        for mask in response["result"]["masks"]
    }


def test_bit_planes_decode_to_the_v0_masks(pipeline):
    client, _ = pipeline
    expected = _uint8_masks(client)
    response = client.ok("view.structure_masks", {"encoding": "bits"})
    result = response["result"]
    assert result["encoding"] == "bits" and len(result["planes"]) == 1
    planes = [
        np.frombuffer(response["buffers"][p["$array"]], dtype=p["dtype"]).reshape(p["shape"])
        for p in result["planes"]
    ]
    assert planes[0].dtype == np.uint8
    for mask in result["masks"]:
        decoded = (planes[mask["plane"]] >> mask["bit"]) & 1
        np.testing.assert_array_equal(decoded, expected[mask["name"]])
        assert mask["num_voxels"] == int(expected[mask["name"]].sum())


def test_many_structures_use_wider_planes(pipeline):
    from pyRadPlan.app._views import BITS_PER_PLANE, _plane_dtype

    assert BITS_PER_PLANE == 32
    assert [_plane_dtype(n).itemsize for n in (1, 8, 9, 16, 17, 32)] == [1, 1, 2, 2, 4, 4]


def test_crops_decode_to_the_v0_masks(pipeline):
    client, _ = pipeline
    expected = _uint8_masks(client)
    response = client.ok("view.structure_masks", {"encoding": "crops", "names": ["Core"]})
    (mask,) = response["result"]["masks"]
    z0, y0, x0, z1, y1, x1 = mask["bbox"]
    crop = np.frombuffer(response["buffers"][mask["data"]["$array"]], dtype=np.uint8).reshape(
        mask["data"]["shape"]
    )
    full = np.zeros_like(expected["Core"])
    full[z0:z1, y0:y1, x0:x1] = crop
    np.testing.assert_array_equal(full, expected["Core"])
    nonzero = np.argwhere(expected["Core"])
    assert mask["bbox"] == [*nonzero.min(axis=0).tolist(), *(nonzero.max(axis=0) + 1).tolist()]
    response = client.call("view.structure_masks", {"encoding": "png"})
    assert response["error"]["type"] == "ValueError"


def test_crop_of_an_empty_structure():
    from pyRadPlan.app._views import _bbox

    assert _bbox(np.zeros((2, 2, 2), dtype=np.uint8)) is None


# ----------------------------------------------------------------------
# Beams against the Qt viewer's helpers
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def qt_beams():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from pyRadPlan.gui.widgets import _result_widget

    return _result_widget


def _to_lps_expectation(qt, ct, entry):
    origin = np.array(ct.cube_hu.GetOrigin())
    spacing = np.array(ct.cube_hu.GetSpacing())
    return {
        "iso_center": qt._to_viewer_vox(entry["iso_center"], origin, spacing),
        "source_direction": qt._to_viewer_direction(entry["source_direction"], spacing),
        "orbit": None
        if "orbit_u" not in entry
        else (
            qt._to_viewer_direction(entry["orbit_u"], spacing),
            qt._to_viewer_direction(entry["orbit_w"], spacing),
        ),
        "orbit_axis": None
        if "orbit_axis" not in entry
        else qt._to_viewer_direction(entry["orbit_axis"], spacing),
    }


def _compare(qt, ct, ours, theirs):
    assert len(ours) == len(theirs)
    for mine, qt_entry in zip(ours, theirs):
        mapped = _to_lps_expectation(qt, ct, mine)
        np.testing.assert_allclose(mapped["iso_center"], qt_entry["iso_center"])
        np.testing.assert_allclose(mapped["source_direction"], qt_entry["source_direction"])
        assert mine["name"] == qt_entry["name"] and mine["label"] == qt_entry["label"]
        if "orbit" in qt_entry:
            np.testing.assert_allclose(mapped["orbit"][0], qt_entry["orbit"][0], atol=1e-12)
            np.testing.assert_allclose(mapped["orbit"][1], qt_entry["orbit"][1], atol=1e-12)
            np.testing.assert_allclose(mapped["orbit_axis"], qt_entry["orbit_axis"], atol=1e-12)
            assert (mine["gantry"], mine["couch"]) == (qt_entry["gantry"], qt_entry["couch"])
        else:
            assert mapped["orbit"] is None
        assert np.isclose(np.linalg.norm(mine["source_direction"]), 1.0)


def test_beams_from_the_stf_match_the_qt_viewer(pipeline, qt_beams):
    client, session = pipeline
    ws = session.workspace
    result = client.ok("view.beams")["result"]
    assert result["source"] == "stf"
    iso = qt_beams._plan_iso_centers(ws.pln, ws.cst)
    np.testing.assert_allclose(result["iso_centers"], iso)
    expected = qt_beams._beam_geometry(ws.ct, ws.pln, ws.stf, iso)
    _compare(qt_beams, ws.ct, result["beams"], expected)
    assert not any(beam["pending"] for beam in result["beams"])


def test_beams_from_the_plan_match_the_qt_viewer(pipeline, qt_beams):
    _, session = pipeline
    ws = session.workspace
    pln = ws.pln.model_copy(
        update={
            "prop_stf": {**ws.pln.prop_stf, "gantry_angles": [10.0, 200.0], "couch_angles": [15.0]}
        }
    )
    result = beam_geometry(pln, ws.stf, ws.cst)
    assert result["source"] == "plan"
    iso = qt_beams._plan_iso_centers(pln, ws.cst)
    _compare(qt_beams, ws.ct, result["beams"], qt_beams._beam_geometry(ws.ct, pln, ws.stf, iso))
    assert [b["couch"] for b in result["beams"]] == [15.0, 15.0]


def test_draft_beams_are_pending_where_they_differ(pipeline, qt_beams):
    client, session = pipeline
    ws = session.workspace
    draft = {"gantry_angles": [0.0, 90.0, 240.0], "couch_angles": [0.0, 0.0, 0.0]}
    result = client.ok("view.beams", {"draft": draft})["result"]
    assert result["source"] == "plan"
    assert [b["pending"] for b in result["beams"]] == [False, True, False]
    edited = ws.pln.model_copy(update={"prop_stf": {**ws.pln.prop_stf, **draft}})
    iso = qt_beams._plan_iso_centers(edited, ws.cst)
    _compare(qt_beams, ws.ct, result["beams"], qt_beams._beam_geometry(ws.ct, edited, ws.stf, iso))

    moved = client.ok("view.beams", {"draft": {**draft, "iso_center": [1.0, 2.0, 3.0]}})["result"]
    assert moved["iso_centers"] == [[1.0, 2.0, 3.0]]
    assert all(b["iso_center"] == [1.0, 2.0, 3.0] for b in moved["beams"])


def test_no_beams_without_angles(client):
    result = client.ok("view.beams")["result"]
    assert result == {"source": None, "iso_centers": [], "beams": []}
