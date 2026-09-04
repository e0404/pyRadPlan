import pytest
from pyRadPlan import load_tg119
import SimpleITK as sitk
from pyRadPlan.analysis import DVHCollection, DVH
import numpy as np
from matplotlib import pyplot as plt


@pytest.fixture
def cst():
    # Get StructureSet from TG119
    return load_tg119()[1]


@pytest.fixture
def dose():
    # Create dose image using information from TG119
    ct, _ = load_tg119()
    dose_array = np.zeros(ct.size[::-1])
    dose_array[60:80, 80:120, 80:120] = 1

    # Create dose image
    dose_image = sitk.GetImageFromArray(dose_array)
    # Copy information from CT image
    dose_image.SetSpacing((ct.resolution["x"], ct.resolution["y"], ct.resolution["z"]))
    dose_image.SetOrigin((ct.origin[0], ct.origin[1], ct.origin[2]))
    dose_image.SetDirection(
        (
            ct.direction[0],
            ct.direction[1],
            ct.direction[2],
            ct.direction[3],
            ct.direction[4],
            ct.direction[5],
            ct.direction[6],
            ct.direction[7],
            ct.direction[8],
        )
    )
    return dose_image


def test_dvhcollection(cst, dose):
    dvh = DVHCollection.from_structure_set(cst=cst, dose=dose)
    assert len(dvh.dvhs) == len(cst.vois)


def test_dvh(cst, dose):
    n_points = 500
    dvh = DVH.compute(
        mask=cst.vois[0].mask, quantity=dose, name=cst.vois[0].name, num_points=n_points
    )
    assert dvh.name == cst.vois[0].name
    assert dvh.bin_edges.shape == (n_points + 1,)
    assert dvh.bins.shape == (n_points,)
    assert np.all(dvh.bins == dvh.bin_edges[:-1])
    assert dvh.bin_centers.shape == (n_points,)
    assert np.all(dvh.bin_centers > dvh.bins)
    assert dvh.diff_volume.shape == (n_points,)
    assert dvh.cum_volume.shape == (n_points,)
    assert dvh.cumulative.shape == (2, n_points)
    assert dvh.differential.shape == (2, n_points)
    assert np.array_equal(dvh.cumulative[0], dvh.bins)
    assert np.array_equal(dvh.cumulative[1], dvh.cum_volume)
    assert np.array_equal(dvh.differential[0], dvh.bin_centers)
    assert np.array_equal(dvh.differential[1], dvh.diff_volume)

    assert isinstance(dvh.has_regular_bins, np.bool)
    assert dvh.get_vx(1.0) == np.interp(1.0, dvh.cumulative[0], dvh.cumulative[1])
    assert dvh.get_dy(50) > 0.0


# --------------------------------------------------------------------------
# Quantity dtype
# --------------------------------------------------------------------------


def test_compute_accepts_a_float32_quantity():
    """A float32 dose must work: that is what the DICOM importer produces.

    np.histogram takes the bin dtype from the quantity, so float32 input yielded
    float32 bin edges, which the model declares as float64 and rejected.
    """
    quantity = np.linspace(0.0, 3.0, 1000, dtype=np.float32).reshape(10, 10, 10)

    dvh = DVH.compute(quantity, num_points=50)

    assert dvh.bin_edges.dtype == np.float64
    assert dvh.diff_volume.dtype == np.float64


def test_compute_float32_and_float64_agree():
    """Widening the edges must not move them: the two inputs agree to float32 precision."""
    values = np.linspace(0.0, 3.0, 1000).reshape(10, 10, 10)

    as32 = DVH.compute(values.astype(np.float32), num_points=100)
    as64 = DVH.compute(values.astype(np.float64), num_points=100)

    assert np.allclose(as32.bin_edges, as64.bin_edges, atol=1e-6)
    assert np.allclose(as32.cum_volume, as64.cum_volume, atol=1e-2)


def test_compute_accepts_a_float32_sitk_image():
    image = sitk.Cast(
        sitk.GetImageFromArray(np.linspace(0.0, 3.0, 1000).reshape(10, 10, 10)),
        sitk.sitkFloat32,
    )

    dvh = DVH.compute(image, num_points=50)

    assert dvh.bin_edges.dtype == np.float64


# --------------------------------------------------------------------------
# Dy
# --------------------------------------------------------------------------


def test_get_dy_on_a_stepped_distribution():
    """Dy is the largest dose still covering at least y% of the volume.

    Equal quarters at 1, 2, 3 and 4 Gy: the hottest 25% all receive 4 Gy, the
    hottest 50% receive at least 3 Gy, and so on. A plateau like this is exactly
    what a uniformly irradiated region produces, and interpolating across the
    cumulative curve reports the dose at the wrong end of it.
    """
    quantity = np.repeat([1.0, 2.0, 3.0, 4.0], 25).reshape(10, 10, 1)

    dvh = DVH.compute(quantity, num_points=400)

    bin_width = float(dvh.bin_edges[1] - dvh.bin_edges[0])
    for volume, expected in [(25.0, 4.0), (50.0, 3.0), (75.0, 2.0), (100.0, 1.0)]:
        assert dvh.get_dy(volume) == pytest.approx(expected, abs=2 * bin_width)


def test_get_dy_on_a_uniform_distribution():
    """A flat distribution over 0..10 Gy has Dy = (100 - y)% of 10 Gy."""
    quantity = np.linspace(0.0, 10.0, 100000).reshape(-1, 1, 1)

    dvh = DVH.compute(quantity, num_points=1000)

    bin_width = float(dvh.bin_edges[1] - dvh.bin_edges[0])
    for volume in (10.0, 50.0, 90.0):
        expected = (100.0 - volume) / 100.0 * 10.0
        assert dvh.get_dy(volume) == pytest.approx(expected, abs=2 * bin_width)


def test_get_dy_is_monotonically_decreasing():
    """More volume covered can only mean a lower dose threshold."""
    rng = np.random.default_rng(0)
    dvh = DVH.compute(rng.uniform(0.0, 5.0, size=(20, 20, 20)), num_points=500)

    volumes = np.arange(5.0, 100.0, 5.0)
    doses = np.array([dvh.get_dy(float(v)) for v in volumes])
    assert np.all(np.diff(doses) <= 0.0)


def test_get_dy_accepts_an_array():
    dvh = DVH.compute(np.linspace(0.0, 10.0, 10000).reshape(-1, 1, 1), num_points=500)

    both = dvh.get_dy(np.array([25.0, 75.0]))

    assert both.shape == (2,)
    assert both[0] == pytest.approx(dvh.get_dy(25.0))
    assert both[1] == pytest.approx(dvh.get_dy(75.0))


def test_get_dy_matches_the_percentile_it_is_defined_as():
    """Dy is the (100-y)th percentile of the voxel values, which is how QI computes DX."""
    rng = np.random.default_rng(1)
    voxels = rng.gamma(shape=4.0, scale=0.5, size=(30, 30, 30))

    dvh = DVH.compute(voxels, num_points=4000)

    bin_width = float(dvh.bin_edges[1] - dvh.bin_edges[0])
    for volume in (2.0, 50.0, 95.0):
        expected = float(np.percentile(voxels, 100.0 - volume))
        assert dvh.get_dy(volume) == pytest.approx(expected, abs=2 * bin_width)


def test_get_dy_rejects_volumes_outside_the_range():
    dvh = DVH.compute(np.linspace(0.0, 1.0, 100).reshape(-1, 1, 1), num_points=10)

    with pytest.raises(ValueError):
        dvh.get_dy(-1.0)
    with pytest.raises(ValueError):
        dvh.get_dy(101.0)
