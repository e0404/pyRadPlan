import csv
import json

import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan.analysis import DVH, DVHCollection, QICollection


@pytest.fixture
def dose():
    rng = np.random.default_rng(3)
    image = sitk.GetImageFromArray(rng.random((6, 8, 10)) * 60.0)
    image.SetSpacing((2.0, 3.0, 4.0))
    return image


@pytest.fixture
def masks():
    first = np.zeros((6, 8, 10), dtype=bool)
    first[1:4, 2:6, 2:8] = True
    second = np.zeros((6, 8, 10), dtype=bool)
    second[3:6, 0:4, 0:5] = True
    return {"PTV": first, "Rectum": second}


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def _collection(dose, masks):
    return DVHCollection(
        dvhs=[
            DVH.compute(dose, mask=mask, name=name, num_points=20, max_value=70.0)
            for name, mask in masks.items()
        ]
    )


def test_total_volume_from_voxel_volume(dose, masks):
    dvh = DVH.compute(dose, mask=masks["PTV"], name="PTV")
    assert dvh.total_volume == pytest.approx(masks["PTV"].sum() * 2 * 3 * 4 / 1000.0)
    assert DVH.compute(sitk.GetArrayFromImage(dose), mask=masks["PTV"]).total_volume is None


def test_dvh_csv_cumulative_and_absolute_volume(dose, masks, tmp_path):
    dvh = DVH.compute(dose, mask=masks["PTV"], name="PTV", num_points=10)
    dvh.to_csv(tmp_path / "dvh.csv", absolute_volume=True)
    rows = _read(tmp_path / "dvh.csv")
    assert rows[0] == ["dose [Gy]", "volume [%]", "volume [cm3]"]
    values = np.array(rows[1:], dtype=float)
    np.testing.assert_array_equal(values[:, 0], dvh.bins)
    np.testing.assert_array_equal(values[:, 1], dvh.cum_volume)
    np.testing.assert_allclose(values[:, 2], dvh.cum_volume / 100 * dvh.total_volume)
    assert values[0, 1] == pytest.approx(100.0)


def test_dvh_csv_differential(dose, masks):
    dvh = DVH.compute(dose, mask=masks["PTV"], num_points=10)
    values = np.array(dvh.to_rows(kind="differential")[1:], dtype=float)
    np.testing.assert_array_equal(values[:, 0], dvh.bin_centers)
    np.testing.assert_array_equal(values[:, 1], dvh.diff_volume)
    assert values[:, 1].sum() == pytest.approx(100.0)


def test_dvh_rows_errors(dose, masks):
    dvh = DVH.compute(sitk.GetArrayFromImage(dose), mask=masks["PTV"], num_points=5)
    with pytest.raises(ValueError, match="absolute volume"):
        dvh.to_rows(absolute_volume=True)
    with pytest.raises(ValueError, match="kind"):
        dvh.to_rows(kind="other")


def test_collection_csv_wide(dose, masks, tmp_path):
    collection = _collection(dose, masks)
    collection.to_csv(tmp_path / "wide.csv", absolute_volume=True)
    rows = _read(tmp_path / "wide.csv")
    assert rows[0] == ["dose [Gy]", "PTV [%]", "PTV [cm3]", "Rectum [%]", "Rectum [cm3]"]
    values = np.array(rows[1:], dtype=float)
    assert values.shape == (20, 5)
    np.testing.assert_array_equal(values[:, 0], collection.dvhs[0].bins)
    np.testing.assert_array_equal(values[:, 3], collection.dvhs[1].cum_volume)


def test_collection_csv_long_when_bins_differ(dose, masks, tmp_path):
    dvhs = [
        DVH.compute(dose, mask=masks["PTV"], name="PTV", num_points=5),
        DVH.compute(dose, mask=masks["Rectum"], name="Rectum", num_points=7),
    ]
    collection = DVHCollection(dvhs=dvhs)
    collection.to_csv(tmp_path / "long.csv")
    rows = _read(tmp_path / "long.csv")
    assert rows[0] == ["structure", "dose [Gy]", "volume [%]"]
    assert len(rows) == 1 + 5 + 7
    assert [r[0] for r in rows[1:]] == ["PTV"] * 5 + ["Rectum"] * 7
    with pytest.raises(ValueError, match="Wide layout"):
        collection.to_rows(layout="wide")


def test_collection_long_layout_forced(dose, masks):
    rows = _collection(dose, masks).to_rows(layout="long", absolute_volume=True)
    assert rows[0] == ["structure", "dose [Gy]", "volume [%]", "volume [cm3]"]
    assert len(rows) == 1 + 40


def test_collection_long_layout_converts_units(dose, masks):
    gray = DVH.compute(dose, mask=masks["PTV"], name="PTV", num_points=5, max_value=70.0)
    centigray = DVH(
        name="PTV cGy", unit="cGy", bin_edges=gray.bin_edges * 100.0, diff_volume=gray.diff_volume
    )
    rows = DVHCollection(dvhs=[gray, centigray]).to_rows()
    assert rows[0][1] == "dose [Gy]"
    doses = [float(r[1]) for r in rows[1:]]
    np.testing.assert_allclose(doses[5:], doses[:5])


def test_collection_long_layout_rejects_incompatible_units(dose, masks):
    gray = DVH.compute(dose, mask=masks["PTV"], name="PTV", num_points=5)
    let = DVH(name="LET", unit="keV/um", bin_edges=gray.bin_edges, diff_volume=gray.diff_volume)
    with pytest.raises(ValueError, match="incompatible units"):
        DVHCollection(dvhs=[gray, let]).to_rows()


def test_empty_collection_has_no_rows():
    assert DVHCollection(dvhs=[]).to_rows() == []


def test_dvh_json_round_trip(dose, masks, tmp_path):
    dvh = DVH.compute(dose, mask=masks["PTV"], name="PTV", num_points=10)
    restored = DVH.model_validate_json(dvh.model_dump_json())
    assert restored.unit == dvh.unit
    assert restored.name == "PTV"
    assert restored.total_volume == dvh.total_volume
    np.testing.assert_array_equal(restored.bin_edges, dvh.bin_edges)
    np.testing.assert_array_equal(restored.diff_volume, dvh.diff_volume)

    dvh.to_json(tmp_path / "dvh.json")
    from_file = DVH.model_validate_json((tmp_path / "dvh.json").read_text(encoding="utf-8"))
    np.testing.assert_array_equal(from_file.cum_volume, dvh.cum_volume)


def test_dvh_json_keeps_non_gray_unit():
    dvh = DVH(
        diff_volume=np.array([50.0, 50.0]),
        bin_edges=np.array([0.0, 1.0, 2.0]),
        unit="keV/um",
        name="LET",
    )
    restored = DVH.model_validate_json(dvh.model_dump_json())
    assert restored.unit == dvh.unit


def test_collection_json_round_trip(dose, masks, tmp_path):
    collection = _collection(dose, masks)
    restored = DVHCollection.model_validate_json(collection.model_dump_json())
    assert [d.name for d in restored.dvhs] == ["PTV", "Rectum"]
    for new, old in zip(restored.dvhs, collection.dvhs):
        np.testing.assert_array_equal(new.diff_volume, old.diff_volume)
        assert new.unit == old.unit

    collection.to_json(tmp_path / "c.json", indent=2)
    json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))


def test_qi_rows(dose, masks):
    qis = QICollection.from_masks(masks, dose, ref_vols=[50.0], ref_doses=[30.0])
    rows = qis.to_rows()
    assert rows[0][0] == "Structure"
    assert rows[0][1:] == [
        "mean [Gy]",
        "std [Gy]",
        "max [Gy]",
        "min [Gy]",
        "D50 [Gy]",
        "V30Gy [%]",
    ]
    assert [r[0] for r in rows[1:]] == ["PTV", "Rectum"]
    assert rows[1][1] == f"{qis['PTV']['mean'].value:.2f}"

    subset = qis.to_rows(structures=["Rectum", "missing"], metrics=["D50", "mean"])
    assert subset[0] == ["Structure", "mean [Gy]", "D50 [Gy]"]
    assert [r[0] for r in subset[1:]] == ["Rectum"]
    assert qis.to_rows(structures=["missing"]) == []
    assert QICollection().to_rows() == []

    precise = qis.to_rows(precision=None)
    assert float(precise[1][1]) == qis["PTV"]["mean"].value


def test_qi_csv_and_missing_values(dose, masks, tmp_path):
    empty = dict(masks, Empty=np.zeros((6, 8, 10), dtype=bool))
    qis = QICollection.from_masks(empty, dose, ref_vols=[50.0], ref_doses=[])
    qis.to_csv(tmp_path / "qi.csv")
    rows = _read(tmp_path / "qi.csv")
    assert rows == qis.to_rows()
    assert rows[-1][0] == "Empty"
    assert set(rows[-1][1:]) == {"-"}


def test_qi_json_round_trip(dose, masks, tmp_path):
    empty = dict(masks, Empty=np.zeros((6, 8, 10), dtype=bool))
    qis = QICollection.from_masks(empty, dose, ref_vols=[50.0], ref_doses=[20.0])
    restored = QICollection.model_validate_json(qis.model_dump_json())
    assert list(restored.structures) == list(qis.structures)
    for name, structure in qis.structures.items():
        for metric, qi in structure.metrics.items():
            other = restored[name][metric]
            assert type(other) is type(qi)
            assert other.unit == qi.unit
            np.testing.assert_equal(other.value, qi.value)

    qis.to_json(tmp_path / "qi.json")
    from_file = QICollection.model_validate_json((tmp_path / "qi.json").read_text("utf-8"))
    assert restored.to_rows() == from_file.to_rows() == qis.to_rows()
