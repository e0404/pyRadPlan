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
# Beams (the geometry both viewers draw)
# ----------------------------------------------------------------------


def _check_orbit(beam):
    """The source direction lies on the beam's gantry orbit, whose axis is u x w."""
    u, w, axis = (np.asarray(beam[k]) for k in ("orbit_u", "orbit_w", "orbit_axis"))
    gantry = np.deg2rad(beam["gantry"])
    np.testing.assert_allclose(
        np.cos(gantry) * u + np.sin(gantry) * w, beam["source_direction"], atol=1e-9
    )
    np.testing.assert_allclose(axis, np.cross(u, w), atol=1e-12)
    assert np.isclose(np.linalg.norm(axis), 1.0) and abs(u @ w) < 1e-12


def test_beams_from_the_stf_follow_the_steering_information(pipeline):
    from pyRadPlan.app._views import plan_iso_centers

    client, session = pipeline
    ws = session.workspace
    result = client.ok("view.beams")["result"]
    assert result["source"] == "stf"
    iso = plan_iso_centers(ws.pln, ws.cst)
    np.testing.assert_allclose(result["iso_centers"], iso)
    assert len(result["beams"]) == len(ws.stf.beams)
    for beam, stf_beam in zip(result["beams"], ws.stf.beams):
        source = np.asarray(stf_beam.source_point, dtype=float)
        np.testing.assert_allclose(beam["source_direction"], source / np.linalg.norm(source))
        np.testing.assert_allclose(beam["iso_center"], iso[0])
        assert (beam["gantry"], beam["couch"]) == (stf_beam.gantry_angle, stf_beam.couch_angle)
        assert beam["label"] == f"Beam #{beam['index']}: gantry {beam['gantry']:g}°, couch 0°"
        assert not beam["pending"]
        _check_orbit(beam)


def test_beams_from_the_plan_agree_with_the_stf(pipeline):
    _, session = pipeline
    ws = session.workspace
    from_stf = beam_geometry(ws.pln, ws.stf, ws.cst)["beams"]
    from_plan = beam_geometry(ws.pln, None, ws.cst)
    assert from_plan["source"] == "plan"
    for derived, generated in zip(from_plan["beams"], from_stf):
        np.testing.assert_allclose(
            derived["source_direction"], generated["source_direction"], atol=1e-9
        )
        np.testing.assert_allclose(derived["iso_center"], generated["iso_center"])

    pln = ws.pln.model_copy(
        update={
            "prop_stf": {**ws.pln.prop_stf, "gantry_angles": [10.0, 200.0], "couch_angles": [15.0]}
        }
    )
    result = beam_geometry(pln, ws.stf, ws.cst)
    assert result["source"] == "plan"
    assert [b["couch"] for b in result["beams"]] == [15.0, 15.0]
    for beam in result["beams"]:
        _check_orbit(beam)


def test_draft_beams_are_pending_where_they_differ(pipeline):
    client, _ = pipeline
    draft = {"gantry_angles": [0.0, 90.0, 240.0], "couch_angles": [0.0, 0.0, 0.0]}
    result = client.ok("view.beams", {"draft": draft})["result"]
    assert result["source"] == "plan"
    assert [b["pending"] for b in result["beams"]] == [False, True, False]
    assert [b["gantry"] for b in result["beams"]] == draft["gantry_angles"]

    moved = client.ok("view.beams", {"draft": {**draft, "iso_center": [1.0, 2.0, 3.0]}})["result"]
    assert moved["iso_centers"] == [[1.0, 2.0, 3.0]]
    assert all(b["iso_center"] == [1.0, 2.0, 3.0] for b in moved["beams"])


def test_no_beams_without_angles(client):
    result = client.ok("view.beams")["result"]
    assert result == {"source": None, "iso_centers": [], "beams": []}
