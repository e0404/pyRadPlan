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
