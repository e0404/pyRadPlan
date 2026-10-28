import warnings

import pytest
import numpy as np
import SimpleITK as sitk
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from pyRadPlan import load_tg119  # noqa: E402
from pyRadPlan.analysis import (  # noqa: E402
    DEFAULT_REF_VOLS,
    DX,
    QICollection,
    StructureQIs,
    Max,
    Mean,
    Min,
    Std,
    VX,
    format_metric_label,
    format_unit_symbol,
)
from pyRadPlan.analysis._dvh import ureg  # shared registry  # noqa: E402


@pytest.fixture
def cst():
    """Get StructureSet from TG119"""
    return load_tg119()[1]


@pytest.fixture
def dose():
    """Create a simple dose distribution for testing"""
    ct, _ = load_tg119()
    dose_array = np.zeros(ct.size[::-1])
    dose_array[60:80, 80:120, 80:120] = 1.0

    dose_image = sitk.GetImageFromArray(dose_array)
    dose_image.SetSpacing((ct.resolution["x"], ct.resolution["y"], ct.resolution["z"]))
    dose_image.SetOrigin((ct.origin[0], ct.origin[1], ct.origin[2]))
    dose_image.SetDirection(tuple(ct.direction[i] for i in range(9)))
    return dose_image


@pytest.fixture
def simple_dose_array():
    """Create a simple numpy array with known dose distribution"""
    return np.linspace(0, 10, 1000)


@pytest.fixture
def simple_mask():
    """Create a simple mask for testing"""
    mask = np.ones(1000, dtype=bool)
    mask[900:] = False  # Exclude top 10%
    return mask


def test_basic_qi_statistics(simple_dose_array, simple_mask):
    """Test basic QI statistics with and without mask"""
    mean_qi = Mean.compute_from(quantity=simple_dose_array)
    assert isinstance(mean_qi, Mean)
    assert mean_qi.metric == "mean"
    assert np.isclose(mean_qi.value, 5.0, rtol=1e-5)
    assert mean_qi.unit == ureg.gray

    std_qi = Std.compute_from(quantity=simple_dose_array)
    assert std_qi.metric == "std"
    assert np.isclose(std_qi.value, 2.8896395421142094, rtol=1e-5)

    max_qi = Max.compute_from(quantity=simple_dose_array)
    assert max_qi.metric == "max"
    assert np.isclose(max_qi.value, 10.0, rtol=1e-5)

    min_qi = Min.compute_from(quantity=simple_dose_array)
    assert min_qi.metric == "min"
    assert np.isclose(min_qi.value, 0.0, rtol=1e-5)

    mean_masked = Mean.compute_from(quantity=simple_dose_array, mask=simple_mask)
    assert np.isclose(mean_masked.value, 4.499499499499499, rtol=1e-5)

    max_masked = Max.compute_from(quantity=simple_dose_array, mask=simple_mask)
    assert np.isclose(max_masked.value, 8.998998998999, rtol=1e-5)


def test_qi_with_sitk_images(dose, cst):
    """Test all QI types work with SimpleITK images and masks"""
    mean_qi = Mean.compute_from(quantity=dose, mask=cst.vois[0].mask)
    assert isinstance(mean_qi, Mean)
    assert np.isclose(mean_qi.value, 0.5, rtol=1e-5)

    dx_qi = DX.compute_from(quantity=dose, mask=cst.vois[0].mask, ref_vol=50)
    assert isinstance(dx_qi, DX)
    assert np.isclose(dx_qi.value, 0.5, rtol=1e-5)

    vx_qi = VX.compute_from(quantity=dose, mask=cst.vois[0].mask, ref_dose=0.5)
    assert isinstance(vx_qi, VX)
    assert np.isclose(vx_qi.value, 50.0, rtol=1e-5)


def test_dx_metrics(simple_dose_array, simple_mask):
    """Test DX computation for various percentiles"""
    dx50 = DX.compute_from(quantity=simple_dose_array, ref_vol=50.0)
    assert dx50.metric == "D50"
    assert np.isclose(dx50.value, 5.0, rtol=1e-5)
    assert dx50.ref_vol == 50.0

    dx95 = DX.compute_from(quantity=simple_dose_array, ref_vol=95)
    assert dx95.metric == "D95"
    assert np.isclose(dx95.value, 0.5, rtol=1e-5)

    dx2 = DX.compute_from(quantity=simple_dose_array, ref_vol=2)
    assert dx2.metric == "D2"
    assert np.isclose(dx2.value, 9.8, rtol=1e-5)

    # Fractional ref_vol survives in the metric id
    dx_frac = DX.compute_from(quantity=simple_dose_array, ref_vol=2.5)
    assert dx_frac.metric == "D2.5"

    dx_masked = DX.compute_from(quantity=simple_dose_array, mask=simple_mask, ref_vol=50.0)
    assert np.isclose(dx_masked.value, 4.4994994994995, rtol=1e-5)


def test_vx_metrics(simple_dose_array, simple_mask):
    """Test VX computation for various dose levels"""
    vx5 = VX.compute_from(quantity=simple_dose_array, ref_dose=5.0)
    assert vx5.metric == "V5Gy"
    assert np.isclose(vx5.value, 50.0, rtol=1e-5)
    assert str(vx5.unit) == "percent"

    vx0 = VX.compute_from(quantity=simple_dose_array, ref_dose=0.0)
    assert vx0.metric == "V0Gy"
    assert np.isclose(vx0.value, 100.0, rtol=1e-5)

    vx_high = VX.compute_from(quantity=simple_dose_array, ref_dose=15.0)
    assert vx_high.metric == "V15Gy"
    assert np.isclose(vx_high.value, 0.0, rtol=1e-5)

    # Fractional ref_dose survives in the metric id
    vx_frac = VX.compute_from(quantity=simple_dose_array, ref_dose=2.5)
    assert vx_frac.metric == "V2.5Gy"

    vx_masked = VX.compute_from(quantity=simple_dose_array, mask=simple_mask, ref_dose=5.0)
    assert np.isclose(vx_masked.value, 44.44444444444444, rtol=1e-5)

    empty_mask = np.zeros(5, dtype=bool)
    vx_empty = VX.compute_from(quantity=np.array([1, 2, 3, 4, 5]), mask=empty_mask, ref_dose=3.0)
    assert np.isnan(vx_empty.value)


def test_vx_unit_conversion():
    """Reference-dose units are converted before thresholding."""
    dose_gy = np.array([0.0, 1.0, 2.0, 3.0])
    vx_200cgy = VX.compute_from(
        quantity=dose_gy,
        ref_dose=200.0,
        ref_unit=ureg.cGy,
        quantity_unit=ureg.gray,
    )
    assert vx_200cgy.metric == "V200cGy"
    assert np.isclose(vx_200cgy.value, 50.0)

    dose_cgy = np.array([0.0, 100.0, 200.0, 300.0])
    vx_2gy = VX.compute_from(
        quantity=dose_cgy,
        ref_dose=2.0,
        ref_unit=ureg.gray,
        quantity_unit=ureg.cGy,
    )
    assert vx_2gy.metric == "V2Gy"
    assert np.isclose(vx_2gy.value, 50.0)

    with pytest.raises(ValueError, match="not compatible"):
        VX.compute_from(
            quantity=dose_gy,
            ref_dose=2.0,
            ref_unit=ureg.meter,
            quantity_unit=ureg.gray,
        )


@pytest.mark.parametrize("ref_vol", [-1, 101, float("nan")])
def test_dx_rejects_invalid_reference_volumes(simple_dose_array, ref_vol):
    """D_x reference volumes must be percentages."""
    with pytest.raises(ValueError, match="Reference volume"):
        DX.compute_from(quantity=simple_dose_array, ref_vol=ref_vol)

    with pytest.raises(ValueError, match="Reference volume"):
        DX.compute_from(quantity=np.array([]), ref_vol=ref_vol)


def test_qicollection_creation_and_metrics(cst, dose):
    """Test QICollection creation and metric computation"""
    qi_collection = QICollection.from_structure_set(
        cst=cst, dose=dose, ref_vols=[2, 50, 95], ref_doses=[0.5, 1.0]
    )

    assert isinstance(qi_collection, QICollection)
    assert len(qi_collection) == len(cst.vois)

    for voi in cst.vois:
        assert voi.name in qi_collection
        assert isinstance(qi_collection[voi.name], StructureQIs)

    structure_qis = qi_collection[cst.vois[0].name]
    for metric in ("mean", "std", "max", "min", "D2", "D50", "D95", "V0.5Gy", "V1Gy"):
        assert metric in structure_qis


def test_qicollection_custom_parameters(cst, dose):
    """Test QICollection with custom reference volumes and doses"""
    qi_collection = QICollection.from_structure_set(
        cst=cst, dose=dose, ref_vols=[10, 90], ref_doses=[0.25, 0.75]
    )

    structure_qis = qi_collection[cst.vois[0].name]
    assert "D10" in structure_qis and "D90" in structure_qis
    assert "V0.25Gy" in structure_qis and "V0.75Gy" in structure_qis


def test_qicollection_rejects_duplicate_voi_names(cst, dose):
    """Structure names are dictionary keys, so duplicates must fail explicitly."""
    from pyRadPlan.cst import StructureSet

    duplicate_voi = cst.vois[0].model_copy(deep=True)
    duplicate_voi.name = cst.vois[0].name
    modified_cst = StructureSet(vois=[cst.vois[0], duplicate_voi], ct_image=cst.ct_image)

    with pytest.raises(ValueError, match="Duplicate VOI name"):
        QICollection.from_structure_set(cst=modified_cst, dose=dose)


def test_qicollection_default_ref_doses(cst, dose):
    """Default ref_doses are derived from max dose and exclude zero."""
    qi_collection = QICollection.from_structure_set(cst=cst, dose=dose, ref_vols=[50])

    structure_qis = qi_collection[cst.vois[0].name]
    vx_keys = [k for k in structure_qis.keys() if k.startswith("V")]
    assert vx_keys, "Expected at least one VX metric"
    # No V0Gy in defaults (would always be ~100% and uninformative)
    assert "V0Gy" not in structure_qis


def test_qicollection_plotting(cst, dose, tmp_path):
    """Test QICollection plot with various filters"""
    qi_collection = QICollection.from_structure_set(
        cst=cst, dose=dose, ref_vols=[50], ref_doses=[1.0]
    )

    fig, ax = plt.subplots()
    returned_ax = qi_collection.plot(ax=ax)
    assert returned_ax is ax
    plt.savefig(str(tmp_path / "qi_basic_plot.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    qi_collection.plot(ax=ax, structures=[cst.vois[0].name])
    plt.savefig(str(tmp_path / "qi_filtered_structures.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    qi_collection.plot(ax=ax, metrics=["mean", "max", "min"])
    plt.savefig(str(tmp_path / "qi_filtered_metrics.png"))
    plt.close(fig)

    with pytest.raises(ValueError, match="None of the specified structures found"):
        qi_collection.plot(structures=["NonExistentStructure"])


def test_qi_edge_cases():
    """Test QI behavior with edge cases"""
    zeros = np.zeros(100)
    assert Mean.compute_from(quantity=zeros).value == 0.0
    assert Max.compute_from(quantity=zeros).value == 0.0
    assert VX.compute_from(quantity=zeros, ref_dose=0.0).value == 100.0

    uniform = np.ones(100) * 5.0
    assert np.isclose(Mean.compute_from(quantity=uniform).value, 5.0, rtol=1e-5)
    assert np.isclose(Std.compute_from(quantity=uniform).value, 0.0, atol=1e-10)
    assert np.isclose(DX.compute_from(quantity=uniform, ref_vol=50).value, 5.0, rtol=1e-5)
    assert np.isclose(VX.compute_from(quantity=uniform, ref_dose=5.0).value, 100.0, rtol=1e-5)


def test_qi_rejects_sitk_geometry_mismatch(dose, cst):
    """Same-shaped SimpleITK images must still align in physical space."""
    shifted_mask = sitk.Image(cst.vois[0].mask)
    origin = shifted_mask.GetOrigin()
    shifted_mask.SetOrigin((origin[0] + 1.0, origin[1], origin[2]))

    with pytest.raises(ValueError, match="geometry"):
        Mean.compute_from(quantity=dose, mask=shifted_mask)


def test_qi_rejects_multi_scenario_arrays():
    """QI computation should not silently pool robust scenarios."""
    dose = np.zeros((2, 3, 4, 5))
    mask = np.ones_like(dose, dtype=bool)

    with pytest.raises(ValueError, match="multiple scenarios"):
        Mean.compute_from(quantity=dose, mask=mask)


def test_qi_with_empty_voi(dose):
    """Empty VOIs return NaN without RuntimeWarnings."""
    empty_mask = sitk.GetImageFromArray(np.zeros(dose.GetSize()[::-1], dtype=np.uint8))
    empty_mask.CopyInformation(dose)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        assert np.isnan(Mean.compute_from(quantity=dose, mask=empty_mask).value)
        assert np.isnan(Std.compute_from(quantity=dose, mask=empty_mask).value)
        assert np.isnan(Max.compute_from(quantity=dose, mask=empty_mask).value)
        assert np.isnan(Min.compute_from(quantity=dose, mask=empty_mask).value)
        assert np.isnan(DX.compute_from(quantity=dose, mask=empty_mask, ref_vol=50).value)
        assert np.isnan(VX.compute_from(quantity=dose, mask=empty_mask, ref_dose=0.5).value)


def test_qicollection_with_empty_voi(cst, dose, tmp_path):
    """Test QICollection handles empty VOIs correctly"""
    from pyRadPlan.cst import StructureSet, VOI

    empty_mask = sitk.GetImageFromArray(np.zeros(dose.GetSize()[::-1], dtype=np.uint8))
    empty_mask.CopyInformation(dose)
    empty_voi = VOI(
        name="EmptyStructure",
        mask=empty_mask,
        voi_type="OAR",
    )

    modified_cst = StructureSet(vois=cst.vois + [empty_voi], ct_image=cst.ct_image)
    qi_collection = QICollection.from_structure_set(
        cst=modified_cst, dose=dose, ref_vols=[2, 50, 95], ref_doses=[0.5, 1.0]
    )

    empty_qis = qi_collection["EmptyStructure"]
    assert empty_qis.name == "EmptyStructure"
    for key in ("mean", "std", "max", "min", "D2", "D50", "D95"):
        assert np.isnan(empty_qis[key].value)
    assert np.isnan(empty_qis["V0.5Gy"].value)
    assert np.isnan(empty_qis["V1Gy"].value)

    fig, ax = plt.subplots()
    qi_collection.plot(ax=ax, structures=["EmptyStructure"], metrics=["mean", "max", "D50"])
    plt.savefig(str(tmp_path / "qi_empty_voi.png"))
    plt.close(fig)


def test_qi_units():
    """Test QI unit handling"""
    dose = np.ones(100)

    mean_qi = Mean.compute_from(quantity=dose)
    assert mean_qi.unit == ureg.gray

    vx_qi = VX.compute_from(quantity=dose, ref_dose=1.0)
    assert str(vx_qi.unit) == "percent"

    custom_unit = ureg.cGy
    mean_custom = Mean.compute_from(quantity=dose, unit=custom_unit)
    assert mean_custom.unit == custom_unit

    # Strings are accepted via the field validator
    mean_from_str = Mean(value=1.0, unit="gray")
    assert mean_from_str.unit == ureg.gray

    with pytest.raises(ValueError):
        Mean(value=1.0, unit="not_a_unit")


def test_structure_qis_from_voxels(simple_dose_array):
    """from_voxels builds the standard metric set from pre-extracted voxels."""
    qis = StructureQIs.from_voxels(simple_dose_array, name="test", ref_vols=[50], ref_doses=[5.0])

    assert qis.name == "test"
    assert set(qis.keys()) == {"mean", "std", "max", "min", "D50", "V5Gy"}
    assert qis["mean"].value == pytest.approx(np.mean(simple_dose_array))
    assert qis["max"].value == pytest.approx(np.max(simple_dose_array))


def test_structure_qis_from_voxels_defaults(simple_dose_array):
    """ref_vols defaults to DEFAULT_REF_VOLS; ref_doses defaults to no VX metrics."""
    qis = StructureQIs.from_voxels(simple_dose_array)

    for ref_vol in DEFAULT_REF_VOLS:
        assert f"D{ref_vol:g}" in qis
    assert not [k for k in qis.keys() if k.startswith("V")]


def test_structure_qis_compute_from_matches_direct(dose, cst):
    """compute_from agrees with the single-metric compute_from constructors."""
    voi = cst.vois[0]
    qis = StructureQIs.compute_from(dose, voi.mask, name=voi.name, ref_vols=[50], ref_doses=[0.5])

    assert qis.name == voi.name
    assert qis["mean"].value == pytest.approx(
        Mean.compute_from(quantity=dose, mask=voi.mask).value, nan_ok=True
    )
    assert qis["D50"].value == pytest.approx(
        DX.compute_from(quantity=dose, mask=voi.mask, ref_vol=50).value, nan_ok=True
    )
    assert qis["V0.5Gy"].value == pytest.approx(
        VX.compute_from(quantity=dose, mask=voi.mask, ref_dose=0.5).value, nan_ok=True
    )


def test_from_masks_matches_from_structure_set(cst, dose):
    """The mask-level entry point reproduces the structure-set one exactly."""
    from_cst = QICollection.from_structure_set(
        cst=cst, dose=dose, ref_vols=[2, 50], ref_doses=[0.5, 1.0]
    )
    masks = {voi.name: voi.mask for voi in cst.vois}
    from_masks = QICollection.from_masks(masks, dose, ref_vols=[2, 50], ref_doses=[0.5, 1.0])

    assert set(from_masks.structures) == set(from_cst.structures)
    for name, structure in from_cst.structures.items():
        for metric, qi in structure.items():
            other = from_masks[name][metric]
            assert other.value == pytest.approx(qi.value, nan_ok=True)
            assert other.unit == qi.unit


def test_from_masks_default_ref_doses(cst, dose):
    """Defaults are resolved in from_masks, so both entry points agree."""
    masks = {voi.name: voi.mask for voi in cst.vois}
    from_masks = QICollection.from_masks(masks, dose, ref_vols=[50])
    from_cst = QICollection.from_structure_set(cst=cst, dose=dose, ref_vols=[50])

    name = cst.vois[0].name
    assert list(from_masks[name].keys()) == list(from_cst[name].keys())
    assert "V0Gy" not in from_masks[name]


def test_metric_ids(cst, dose):
    """metric_ids returns the union of metric ids in first-seen order."""
    qis = QICollection.from_structure_set(cst=cst, dose=dose, ref_vols=[50], ref_doses=[1.0])

    assert qis.metric_ids() == ["mean", "std", "max", "min", "D50", "V1Gy"]
    assert qis.metric_ids([cst.vois[0].name]) == qis.metric_ids()
    # Unknown names are ignored rather than raising
    assert qis.metric_ids(["not-a-voi"]) == []


def test_format_metric_label():
    """Column labels carry the unit, except when there is nothing useful to show."""
    assert format_metric_label("mean", Mean(value=1.0, unit=ureg.gray)) == "mean [Gy]"
    assert (
        format_metric_label("V1Gy", VX(value=1.0, unit=ureg.percent, ref_dose=1.0)) == "V1Gy [%]"
    )
    # Dimensionless would render as an empty "[]"
    assert format_metric_label("mean", Mean(value=1.0, unit=ureg.dimensionless)) == "mean"
    assert format_metric_label("mean", None) == "mean"


LET_UNIT = ureg.keV / ureg.micrometer
GREEK_MU = "μ"


def test_format_unit_symbol_writes_greek_mu():
    """Micro is always Greek mu (U+03BC), whichever symbol the installed pint prefers."""
    assert format_unit_symbol(ureg.gray) == "Gy"
    assert format_unit_symbol(LET_UNIT) == f"keV / {GREEK_MU}m"
    assert "µ" not in format_unit_symbol(ureg.gray * LET_UNIT)


def test_micro_units_in_metric_ids_and_labels():
    """VX metric ids and column labels use the stable micro symbol."""
    vx = VX(value=10.0, unit=ureg.percent, ref_dose=2.0, ref_unit=LET_UNIT)
    assert vx.metric == f"V2keV / {GREEK_MU}m"
    mean = Mean(value=1.0, unit=LET_UNIT)
    assert format_metric_label("mean", mean) == f"mean [keV / {GREEK_MU}m]"
