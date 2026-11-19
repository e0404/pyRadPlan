import numpy as np
import pytest

from pyRadPlan.analysis import DEFAULT_REF_VOLS, DVH, QICollection
from pyRadPlan.analysis._units import safe_unit


@pytest.fixture(scope="module")
def pipeline(tg119_pipeline):
    client, _, _ = tg119_pipeline
    return client, client.manager.get(client.session)


def _arrays(response, curve):
    def get(field):
        descriptor = curve[field]
        return np.frombuffer(response["buffers"][descriptor["$array"]], dtype=np.float32)

    return get("dose"), get("volume")


def test_per_structure_dvh_sets_match_dvh_compute(pipeline):
    client, session = pipeline
    keys = ["physical_dose", "physical_dose_beam 2"]
    response = client.ok(
        "analysis.dvh", {"keys": keys, "num_points": 1000, "range": "per_structure"}
    )
    sets = response["result"]["sets"]
    assert [s["key"] for s in sets] == keys
    assert sets[0]["label"] == "Dose" and sets[0]["name"] == "dose"
    assert sets[1]["name"] == "dose, beam 2" and sets[1]["unit"] == "Gy"
    assert sets[0]["unit_info"] == {"symbol": "Gy", "dimensionless": False}
    views = session.views
    for dvh_set in sets:
        values = views.volume_array(dvh_set["key"])
        assert {c["name"] for c in dvh_set["curves"]} == {
            v.name for v in session.workspace.cst.vois
        }
        for curve in dvh_set["curves"]:
            mask = views.mask_array(curve["index"])
            expected = DVH.compute(values, mask, num_points=1000)
            dose, volume = _arrays(response, curve)
            np.testing.assert_allclose(dose, expected.bins.astype(np.float32))
            np.testing.assert_allclose(volume, expected.cum_volume.astype(np.float32), atol=1e-4)
            inside = values[mask.astype(bool)]
            assert curve["stats"]["max"] == pytest.approx(float(inside.max()))


def test_v0_dvh_keeps_its_shape_and_adds_labels(pipeline):
    client, _ = pipeline
    result = client.ok("analysis.dvh", {"names": ["Core"]})["result"]
    assert result["quantity"] == "physical_dose" and result["unit"] == "Gy"
    assert result["label"] == "Dose" and result["unit_info"]["symbol"] == "Gy"
    (curve,) = result["curves"]
    assert curve["dose"]["shape"] == [500]
    per = client.ok("analysis.dvh", {"names": ["Core"], "range": "per_structure"})["result"]
    assert per["curves"][0]["dose"]["shape"] == [500]
    response = client.call("analysis.dvh", {"keys": "physical_dose"})
    assert response["error"]["type"] == "TypeError"
    response = client.call("analysis.dvh", {"range": "global"})
    assert response["error"]["type"] == "ValueError"


def _qi_reference(session, names, ref_vols, ref_doses):
    views = session.views
    values = views.volume_array("physical_dose")
    vois = list(session.workspace.cst.vois)
    masks = {
        voi.name: views.mask_array(i)
        for i, voi in enumerate(vois)
        if names is None or voi.name in names
    }
    return QICollection.from_masks(
        masks, values, ref_vols=ref_vols, ref_doses=ref_doses, dose_unit=safe_unit("Gy")
    )


def test_quality_indicators_match_the_analysis_package(pipeline):
    client, session = pipeline
    result = client.ok("analysis.qi", {"quantity": "physical_dose"})["result"]
    values = session.views.volume_array("physical_dose")
    assert result["ref_vols"] == list(DEFAULT_REF_VOLS)
    assert result["ref_doses"] == QICollection.default_ref_doses(values)
    assert result["label"] == "Dose" and result["unit_symbol"] == "Gy"
    expected = _qi_reference(session, None, None, None)
    ids = [m["id"] for m in result["metrics"]]
    assert ids == expected.metric_ids()
    labels = {m["id"]: m["label"] for m in result["metrics"]}
    assert labels["mean"] == "mean [Gy]" and labels["D98"] == "D98 [Gy]"
    kinds = {m["id"]: (m["kind"], m.get("parameter"), m["unit"]) for m in result["metrics"]}
    assert kinds["D95"] == ("dx", 95.0, "Gy") and kinds["max"] == ("max", None, "Gy")
    vx = [m for m in result["metrics"] if m["kind"] == "vx"]
    assert vx and all(m["unit"] == "%" and m["label"].endswith("[%]") for m in vx)
    for row in result["rows"]:
        for metric, value in row["values"].items():
            assert value == pytest.approx(expected[row["name"]][metric].value), metric


def test_quality_indicator_parameters_fall_back_like_qt(pipeline):
    client, session = pipeline
    params = {"quantity": "physical_dose", "names": ["Core", "OuterTarget"]}
    custom = client.ok("analysis.qi", {**params, "ref_vols": [10, 90], "ref_doses": [20]})
    custom = custom["result"]
    assert [row["name"] for row in custom["rows"]] == ["Core", "OuterTarget"]
    assert custom["ref_vols"] == [10.0, 90.0] and custom["ref_doses"] == [20.0]
    expected = _qi_reference(session, params["names"], [10.0, 90.0], [20.0])
    assert custom["rows"][0]["values"]["V20Gy"] == pytest.approx(expected["Core"]["V20Gy"].value)
    invalid = client.ok("analysis.qi", {**params, "ref_vols": [150], "ref_doses": [-1]})
    invalid = invalid["result"]
    assert invalid["ref_vols"] == list(DEFAULT_REF_VOLS)
    values = session.views.volume_array("physical_dose")
    assert invalid["ref_doses"] == QICollection.default_ref_doses(values)
    empty = client.ok("analysis.qi", {**params, "ref_vols": [], "ref_doses": []})["result"]
    assert empty["ref_vols"] == list(DEFAULT_REF_VOLS)


def test_compare_units(pipeline):
    client, _ = pipeline
    result = client.ok(
        "analysis.compare_units", {"a": "physical_dose", "b": "physical_dose_beam 0"}
    )
    assert result["result"] == {"shareable": True, "scale": 1.0}
    response = client.call("analysis.compare_units", {"a": "physical_dose", "b": "nope"})
    assert response["error"]["type"] == "KeyError"


def test_compare_units_of_incompatible_quantities(client):
    import SimpleITK as sitk

    client.ok("data.load_phantom", {"name": "TG119"})
    session = client.manager.get(client.session)
    image = sitk.Image(session.workspace.ct.cube_hu)
    session.workspace.result = {"physical_dose": image, "let": image, "rbe_x_dose": image}
    params = {"a": "physical_dose", "b": "let"}
    assert client.ok("analysis.compare_units", params)["result"] == {
        "shareable": False,
        "scale": None,
    }
    params = {"a": "physical_dose", "b": "rbe_x_dose"}
    assert client.ok("analysis.compare_units", params)["result"]["shareable"] is True
