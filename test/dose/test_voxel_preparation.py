"""Nominal voxel preparation preserves ordering, coordinates and scenario membership."""

from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan.core.np2sitk import linear_indices_to_grid_coordinates
from pyRadPlan.core.resample import resample_array
from pyRadPlan.ct import CT, resample_ct
from pyRadPlan.dose.engines import ParticleHongPencilBeamEngine


def case(selection, resolution, scenarios=1):
    """Use overlapping non-binary masks and nontrivial physical image geometry."""
    image = sitk.GetImageFromArray(np.zeros((4, 5, 6), dtype=np.float32))
    image.SetSpacing((1.0, 1.5, 2.0))
    image.SetOrigin((10.0, -20.0, 5.0))
    angle = np.pi / 6
    image.SetDirection(
        (np.cos(angle), -np.sin(angle), 0.0, np.sin(angle), np.cos(angle), 0.0, 0.0, 0.0, 1.0)
    )
    ct = CT(cube_hu=image)
    vois = []
    for indices in ([0, 3, 8, 30, 60, 119], [3, 8, 31, 45, 119], []):
        mask = np.zeros((4, 5, 6), dtype=np.uint8)
        mask.ravel()[indices] = 2
        img = sitk.GetImageFromArray(mask)
        img.CopyInformation(image)
        vois.append(SimpleNamespace(mask=img, indices_numpy=np.flatnonzero(mask)))
    engine = object.__new__(ParticleHongPencilBeamEngine)
    engine.voxel_sub_ix = selection
    engine._ct_grid = ct.grid
    engine.dose_grid = ct.grid if resolution is None else ct.grid.resample(resolution)
    # Exercise the existing replicated-index scenario branch on a 3D image. This
    # deliberately does not claim full 4D CT resampling support in the dose engine.
    if scenarios != 1:
        ct = SimpleNamespace(num_of_ct_scen=scenarios, cube_hu=image, cube_dim=ct.cube_dim)
    return engine, ct, SimpleNamespace(vois=vois)


@pytest.mark.parametrize("resolution", [None, {"x": 1.5, "y": 2.0, "z": 2.5}])
@pytest.mark.parametrize(
    "selection", [None, [119, 3, 8, 3, 0], np.array([], dtype=np.int64), [8, 8, 8]]
)
def test_nominal_selection_matches_original_rules(selection, resolution):
    """Union and resampling preserve every original selected voxel and its coordinates."""
    engine, ct, cst = case(selection, resolution)
    original = None if selection is None else np.asarray(selection).copy()
    expected_ct = np.unique(
        np.concatenate([v.indices_numpy for v in cst.vois])
        if selection is None
        else np.asarray(selection)
    )
    expected_dose = expected_ct
    if resolution is not None:
        resampled = resample_ct(
            ct=ct, interpolator=sitk.sitkNearestNeighbor, target_grid=engine.dose_grid
        )
        mask = np.zeros(ct.cube_dim[::-1], dtype=np.float32)
        mask.ravel()[expected_ct] = 1
        values = resample_array(
            mask,
            reference_image=ct.cube_hu,
            interpolator=sitk.sitkNearestNeighbor,
            target_image=resampled.cube_hu,
        )
        expected_dose = np.unique(np.where(np.abs(np.round(values)).ravel().astype(bool)))
    engine._init_voxel_selection(ct, cst)
    np.testing.assert_array_equal(engine._vct_grid, expected_ct)
    np.testing.assert_array_equal(engine._vdose_grid, expected_dose)
    for prefix, indices, grid in (
        ("ct", expected_ct, engine._ct_grid),
        ("dose", expected_dose, engine.dose_grid),
    ):
        np.testing.assert_array_equal(
            getattr(engine, f"_v{prefix}_grid_scen_ix"), [np.ones(indices.size, dtype=bool)]
        )
        np.testing.assert_array_equal(
            np.flatnonzero(getattr(engine, f"_v{prefix}_grid_mask")), indices
        )
        coords = engine._vox_world_coords if prefix == "ct" else engine._vox_world_coords_dose_grid
        np.testing.assert_array_equal(
            coords, linear_indices_to_grid_coordinates(indices, grid, index_type="numpy")
        )
    if original is not None:
        np.testing.assert_array_equal(engine.voxel_sub_ix, original)


def test_auto_selection_does_not_materialize_voi_indices():
    """Dense masks can be unioned without generating each VOI's integer index list."""
    engine, ct, cst = case(None, None)

    class MaskOnly:
        def __init__(self, mask):
            self.mask = mask

        @property
        def indices_numpy(self):
            pytest.fail("Nominal automatic selection must use existing masks")

    expected = np.unique(np.concatenate([v.indices_numpy for v in cst.vois]))
    cst.vois = [MaskOnly(v.mask) for v in cst.vois]
    engine._init_voxel_selection(ct, cst)
    np.testing.assert_array_equal(engine._vct_grid, expected)


def test_empty_masks_keep_empty_selection():
    """An empty union remains empty instead of accidentally selecting voxel zero."""
    engine, ct, cst = case(None, None)
    cst.vois = cst.vois[-1:]
    engine._init_voxel_selection(ct, cst)
    assert engine._vct_grid.size == engine._vdose_grid.size == 0
    assert not engine._vct_grid_mask.any()


@pytest.mark.parametrize("selection", [None, [119, 3, 8, 3, 0]])
def test_replicated_multiscenario_selection_keeps_original_semantics(selection):
    """The non-nominal branch retains the existing union and membership conventions."""
    engine, ct, cst = case(selection, None, scenarios=3)
    expected = np.unique(
        np.concatenate([v.indices_numpy for v in cst.vois]) if selection is None else selection
    )
    engine._init_voxel_selection(ct, cst)
    np.testing.assert_array_equal(engine._vct_grid, expected)
    np.testing.assert_array_equal(engine._vdose_grid, expected)
    assert len(engine._vct_grid_scen_ix) == len(engine._vdose_grid_scen_ix) == 3
    for mask in engine._vct_grid_scen_ix + engine._vdose_grid_scen_ix:
        np.testing.assert_array_equal(mask, np.ones(expected.size, dtype=bool))


@pytest.mark.parametrize("selection", [[120], [1.5]])
def test_invalid_custom_indices_are_not_silently_normalized(selection):
    """Preserve NumPy's rejection of out-of-bounds and non-integer selection arrays."""
    engine, ct, cst = case(selection, None)
    with pytest.raises(IndexError):
        engine._init_voxel_selection(ct, cst)


def test_multiscenario_resampling_retains_membership(monkeypatch):
    """Keep separate memberships when the resampling fallback returns different masks."""
    engine, ct, cst = case([0, 3, 8], {"x": 1.5, "y": 2.0, "z": 2.5}, scenarios=3)
    target = sitk.Image([int(d) for d in engine.dose_grid.dimensions], sitk.sitkFloat32)
    target.SetSpacing(tuple(engine.dose_grid.resolution.values()))
    selected = ([0, 1, 2], [1, 2, 3], [0, 2, 4])
    arrays = []
    for indices in selected:
        values = np.zeros(tuple(reversed(target.GetSize())), dtype=np.float32)
        values.ravel()[indices] = 1
        arrays.append(values)
    results = iter(arrays)
    monkeypatch.setattr(
        "pyRadPlan.dose.engines._base.resample_ct",
        lambda **kwargs: SimpleNamespace(cube_hu=target),
    )
    monkeypatch.setattr(
        "pyRadPlan.dose.engines._base.resample_array", lambda *args, **kwargs: next(results)
    )
    engine._init_voxel_selection(ct, cst)
    np.testing.assert_array_equal(engine._vdose_grid, np.arange(5))
    for actual, indices in zip(engine._vdose_grid_scen_ix, selected):
        np.testing.assert_array_equal(actual, np.isin(np.arange(5), indices))
