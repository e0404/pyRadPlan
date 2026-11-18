"""Structure sets handed to a CT of another dimensionality or with other voxel values."""

import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet, create_voi, validate_cst
from pyRadPlan.optimization.robustness import voi_linear_mask


def _cst_3d(generic_input_3d) -> StructureSet:
    name, ct, mask, _, _ = generic_input_3d
    other = np.zeros_like(mask)
    other[2:5, 3:6, 4:9] = 1
    return StructureSet(
        vois=[
            create_voi(voi_type="TARGET", name=name, ct_image=ct, mask=mask),
            create_voi(voi_type="OAR", name="oar", ct_image=ct, mask=other),
        ],
        ct_image=ct,
    )


def _ct_4d(ct: CT, num_phases: int = 2) -> CT:
    cube = ct.cube_hu
    return CT(cube_hu=sitk.JoinSeries([cube + 100.0 * i for i in range(num_phases)]))


def test_3d_structures_are_repeated_on_every_phase_of_a_4d_ct(generic_input_3d):
    cst = _cst_3d(generic_input_3d)
    ct4d = _ct_4d(cst.ct_image)
    assert cst.ct_image != ct4d

    cst4d = validate_cst(cst, ct=ct4d)
    assert cst4d.ct_image == ct4d
    assert cst4d.num_of_scenarios == 2
    for voi, voi4d in zip(cst.vois, cst4d.vois):
        assert voi4d.mask.GetDimension() == 4
        per_phase = voi4d.scenario_indices(order="numpy")
        assert len(per_phase) == 2
        for indices in per_phase:
            np.testing.assert_array_equal(indices, voi.indices_numpy)
        assert voi_linear_mask(voi4d).shape == voi_linear_mask(voi).shape
        np.testing.assert_array_equal(voi_linear_mask(voi4d), voi_linear_mask(voi))


def test_4d_structures_resample_phase_by_phase(generic_input_3d):
    cst4d = validate_cst(_cst_3d(generic_input_3d), ct=_ct_4d(_cst_3d(generic_input_3d).ct_image))
    coarse = CT(
        cube_hu=sitk.JoinSeries(
            [sitk.Shrink(cst4d.ct_image.cube_hu[:, :, :, i], [2, 2, 1]) for i in range(2)]
        )
    )
    resampled = cst4d.resample_on_new_ct(coarse)
    assert resampled.ct_image == coarse
    for voi in resampled.vois:
        assert voi.mask.GetSize() == coarse.cube_hu.GetSize()
        assert len(voi.scenario_indices(order="numpy")) == 2


def test_structure_set_attaches_to_a_ct_with_the_same_grid(generic_input_3d):
    cst = _cst_3d(generic_input_3d)
    other = CT(cube_hu=cst.ct_image.cube_hu + 50.0)
    assert cst.ct_image != other
    attached = validate_cst(cst, ct=other)
    assert attached.ct_image == other
    for voi, moved in zip(cst.vois, attached.vois):
        np.testing.assert_array_equal(voi.indices_numpy, moved.indices_numpy)


def test_structure_set_rejects_a_ct_on_another_grid(generic_input_3d):
    cst = _cst_3d(generic_input_3d)
    shrunk = CT(cube_hu=sitk.Shrink(cst.ct_image.cube_hu, [2, 2, 1]))
    with pytest.raises(ValueError, match="mismatch"):
        validate_cst(cst, ct=shrunk)


@pytest.mark.parametrize("voi_phases,ct_phases", [(2, 3), (3, 2)])
def test_4d_structures_with_another_phase_count_are_refused(
    generic_input_3d, voi_phases, ct_phases
):
    cst = _cst_3d(generic_input_3d)
    cst4d = validate_cst(cst, ct=_ct_4d(cst.ct_image, voi_phases))
    with pytest.raises(ValueError, match="mismatch"):
        validate_cst(cst4d, ct=_ct_4d(cst.ct_image, ct_phases))
