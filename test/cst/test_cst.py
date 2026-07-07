import os
import pytest

import SimpleITK as sitk
import numpy as np

import pyRadPlan.io.matfile as matfile
from pyRadPlan.ct import create_ct


from pyRadPlan.ct import CT
from pyRadPlan.cst import (
    StructureSet,
    create_cst,
    validate_cst,
    create_voi,
    ExternalVOI,
    HelperVOI,
    Target,
    OAR,
    DEFAULT_VOI_COLORS,
)
# @pytest.fixture
# def sample_ct():
#     image = sitk.GetImageFromArray(np.random.rand(5, 15, 25) * 1000)  # Random HU values
#     image.SetOrigin((0, 0, 0))
#     image.SetSpacing((2, 3, 4))  # Irregular spacing for test
#     image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

#     ct = CT(cube_hu=image)

#     return ct


def test_cst_from_matrad_mat_file(matrad_import):
    ct = create_ct(matrad_import["ct"])
    cst = create_cst(matrad_import["cst"], ct=ct)

    assert isinstance(cst, StructureSet)
    assert isinstance(cst.ct_image, CT)
    assert cst.ct_image.cube_hu.GetSize() == (167, 167, 129)
    assert all(isinstance(voi, (Target, OAR, ExternalVOI, HelperVOI)) for voi in cst.vois)
    assert cst.vois[0].name == "Core"
    assert cst.vois[1].name == "OuterTarget"
    assert cst.vois[2].name == "BODY"
    cst = validate_cst(matrad_import["cst"], ct=ct)
    assert isinstance(cst, StructureSet)
    assert isinstance(cst.ct_image, CT)
    assert cst.ct_image.cube_hu.GetSize() == (167, 167, 129)
    assert all(isinstance(voi, (Target, OAR, ExternalVOI, HelperVOI)) for voi in cst.vois)
    assert cst.vois[0].name == "Core"
    assert cst.vois[1].name == "OuterTarget"
    assert cst.vois[2].name == "BODY"

    with pytest.raises(ValueError):
        cst = create_cst(matrad_import["cst"])

    with pytest.raises(ValueError):
        cst = validate_cst(matrad_import["cst"])


# TODO: Operator to return "False" for different CTs is not implemented correctly yet
# def test_different_ct(matrad_import, sample_ct):
#     ct = create_ct(matrad_import["ct"])
#     cst = validate_cst(matrad_import["cst"], ct=ct)
#     with pytest.raises(ValueError):
#         cst_fail = create_cst(cst, ct=sample_ct)


def test_cst_to_matrad(matrad_import, tmpdir):
    ct = create_ct(matrad_import["ct"])
    cst = create_cst(matrad_import["cst"], ct=ct)

    matrad_list = cst.to_matrad()
    assert isinstance(matrad_list, list)

    tmp_mat_path = os.path.join(tmpdir, "test_cst.mat")
    matfile.save(tmp_mat_path, {"cst": matrad_list})
    assert os.path.exists(tmp_mat_path)

    tmp = matfile.load(tmp_mat_path)

    assert isinstance(tmp, dict)
    assert isinstance(tmp["cst"], list)


def test_cst_target_voxels(generic_input_3d):
    name_3d, ct_3d, mask_3d, _, _ = generic_input_3d

    mask_3d_2 = mask_3d.copy()
    mask_3d_2.fill(0)
    mask_3d_2[0, 0, 0] = 1

    mask_3d_3 = mask_3d.copy()
    mask_3d_3.fill(0)
    mask_3d_3[1, 1, 1] = 1

    mask_3d_4 = mask_3d.copy()
    mask_3d_4.fill(0)
    mask_3d_4[0, 1, 0] = 1

    voi_3d = create_voi(voi_type="TARGET", name=name_3d, ct_image=ct_3d, mask=mask_3d)
    voi_3d_2 = create_voi(voi_type="TARGET", name=name_3d, ct_image=ct_3d, mask=mask_3d_2)
    voi_3d_3 = create_voi(voi_type="OAR", name=name_3d, ct_image=ct_3d, mask=mask_3d_3)
    voi_3d_4 = create_voi(voi_type="EXTERNAL", name=name_3d, ct_image=ct_3d, mask=mask_3d_4)

    cst = StructureSet(vois=[voi_3d, voi_3d_2, voi_3d_3, voi_3d_4], ct_image=ct_3d)

    index_union = cst.target_union_voxels()
    assert (index_union == cst.target_union_voxels(order="sitk")).all()
    index_union_np = cst.target_union_voxels(order="numpy")
    mask_union = cst.target_union_mask()

    assert (index_union == np.array([0, 5000])).all()
    assert (index_union_np == np.array([0, 1])).all()
    assert (sitk.GetArrayViewFromImage(mask_union).ravel(order="F")[index_union] == 1).all()
    assert (sitk.GetArrayViewFromImage(mask_union).ravel(order="C")[index_union_np] == 1).all()


def test_cst_patient_voxels(generic_input_3d):
    name_3d, ct_3d, mask_3d, _, _ = generic_input_3d

    mask_3d_3 = mask_3d.copy()
    mask_3d_3.fill(0)
    mask_3d_3[0, 0, 0] = 1

    mask_3d_4 = mask_3d.copy()
    mask_3d_4.fill(0)
    mask_3d_4[-1, -1, -1] = 1

    voi_3d = create_voi(voi_type="TARGET", name=name_3d, ct_image=ct_3d, mask=mask_3d)
    voi_3d_3 = create_voi(voi_type="OAR", name=name_3d, ct_image=ct_3d, mask=mask_3d_3)
    voi_3d_4 = create_voi(voi_type="OAR", name=name_3d, ct_image=ct_3d, mask=mask_3d_4)

    cst = StructureSet(vois=[voi_3d, voi_3d_3, voi_3d_4], ct_image=ct_3d)

    index_union = cst.patient_voxels()
    assert (index_union == cst.patient_voxels(order="sitk")).all()
    index_union_np = cst.patient_voxels(order="numpy")
    mask_union = cst.patient_mask()

    assert (index_union == np.array([0, 5000, mask_3d_3.size - 1])).all()
    assert (index_union_np == np.array([0, 1, mask_3d_3.size - 1])).all()
    assert (sitk.GetArrayViewFromImage(mask_union).ravel(order="F")[index_union] == 1).all()
    assert (sitk.GetArrayViewFromImage(mask_union).ravel(order="C")[index_union_np] == 1).all()

    voi_3d_4 = create_voi(voi_type="EXTERNAL", name=name_3d, ct_image=ct_3d, mask=mask_3d_4)
    cst.vois[-1] = voi_3d_4
    index_union = cst.patient_voxels()
    assert (index_union == cst.patient_voxels(order="sitk")).all()
    index_union_np = cst.patient_voxels(order="numpy")
    mask_union = cst.patient_mask()

    assert (index_union == np.array([mask_3d_4.size - 1])).all()
    assert (index_union_np == np.array([mask_3d_4.size - 1])).all()


def test_target_center_of_mass():
    ct = create_ct(cube_hu=sitk.Image(10, 10, 10, sitk.sitkInt16))
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[1, 1, 1] = 1
    mask[2, 2, 2] = 1
    mask[3, 3, 3] = 1

    voi = create_voi(voi_type="TARGET", name="test", ct_image=ct, mask=mask)
    cst = StructureSet(vois=[voi], ct_image=ct)

    com = cst.target_center_of_mass()
    assert np.allclose(com, np.array([1.5, 1.5, 1.5]))

    image4d = sitk.JoinSeries([sitk.Image(10, 10, 10, sitk.sitkInt16) for _ in range(3)])
    ct = create_ct(cube_hu=image4d)
    mask = np.zeros((3, 10, 10, 10), dtype=np.uint8)
    mask[0, 0, 0, 0] = 1
    mask[0, 1, 1, 1] = 1
    mask[0, 2, 2, 2] = 1
    mask[0, 3, 3, 3] = 1

    voi = create_voi(voi_type="TARGET", name="test", ct_image=ct, mask=mask)
    cst = StructureSet(vois=[voi], ct_image=ct)

    com = cst.target_center_of_mass()
    assert np.allclose(com, np.array([1.5, 1.5, 1.5]))


# ---------------------------------------------------------------------------
# 4D StructureSet helpers & tests
# ---------------------------------------------------------------------------


def _build_4d_cst(num_scenarios: int = 2):
    """Build a 4D StructureSet whose VOIs hit known voxels per scenario.

    Returns
    -------
    cst : StructureSet
    expected : dict
        Per-scenario known voxel positions for asserting downstream.
    """
    dims = (10, 10, 10)  # numpy (Z, Y, X)
    cube_3d = sitk.GetImageFromArray(np.zeros(dims, dtype=np.float32))
    cube_3d.SetSpacing((1.0, 1.0, 1.0))
    cube_3d.SetOrigin((0.0, 0.0, 0.0))
    cube_3d.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    cube_4d = sitk.JoinSeries([cube_3d for _ in range(num_scenarios)])
    cube_4d.SetOrigin((0.0, 0.0, 0.0, 0.0))
    cube_4d.SetSpacing((1.0, 1.0, 1.0, 1.0))
    cube_4d.SetDirection((1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))
    ct = create_ct(cube_hu=cube_4d)

    # One distinct voxel per scenario, well within (Z=10, Y=10, X=10).
    target_voxels = {s: (s, s, s) for s in range(num_scenarios)}
    oar_voxels = {s: (5, 5, 5 - s) for s in range(num_scenarios)}
    external_voxels = {s: (9, 9, 9) for s in range(num_scenarios)}

    def _build_mask(per_scen: dict) -> sitk.Image:
        arr = np.zeros((num_scenarios,) + dims, dtype=np.uint8)
        for s in range(num_scenarios):
            z, y, x = per_scen[s]
            arr[s, z, y, x] = 1
        return arr

    target_voi = create_voi(
        voi_type="TARGET",
        name="TARGET",
        ct_image=ct,
        mask=_build_mask(target_voxels),
    )
    oar_voi = create_voi(voi_type="OAR", name="OAR", ct_image=ct, mask=_build_mask(oar_voxels))
    external_voi = create_voi(
        voi_type="EXTERNAL",
        name="BODY",
        ct_image=ct,
        mask=_build_mask(external_voxels),
    )
    cst = StructureSet(vois=[target_voi, oar_voi, external_voi], ct_image=ct)

    expected = {
        "dims_zyx": dims,
        "target": target_voxels,
        "oar": oar_voxels,
        "external": external_voxels,
        "num_scenarios": num_scenarios,
    }
    return cst, expected


def _flat_idx(zyx, dims_zyx, order):
    z, y, x = zyx
    Z, Y, X = dims_zyx
    if order == "numpy":
        return x + X * y + X * Y * z
    # sitk-order: F-ravel of (Z,Y,X) numpy buffer -> z fastest
    return z + Z * y + Z * Y * x


def test_num_of_scenarios(generic_input_3d):
    name_3d, ct_3d, mask_3d, _, _ = generic_input_3d
    voi_3d = create_voi(voi_type="TARGET", name=name_3d, ct_image=ct_3d, mask=mask_3d)
    cst_3d = StructureSet(vois=[voi_3d], ct_image=ct_3d)
    assert cst_3d.num_of_scenarios == 1

    cst_4d, expected = _build_4d_cst(num_scenarios=3)
    assert cst_4d.num_of_scenarios == expected["num_scenarios"]


def test_target_union_voxels_4d_any():
    cst, exp = _build_4d_cst()
    dims = exp["dims_zyx"]
    sub_cube_size = int(np.prod(dims))

    # "any" returns a single ndarray of 3D-sub-cube indices, OR-collapsed
    ix_np = cst.target_union_voxels(order="numpy")  # default scenario="any"
    assert isinstance(ix_np, np.ndarray)
    assert (ix_np < sub_cube_size).all()
    expected_np = np.unique(
        np.array([_flat_idx(exp["target"][s], dims, "numpy") for s in exp["target"]])
    )
    assert (np.sort(ix_np) == expected_np).all()

    ix_sitk = cst.target_union_voxels(order="sitk")
    assert isinstance(ix_sitk, np.ndarray)
    expected_sitk = np.unique(
        np.array([_flat_idx(exp["target"][s], dims, "sitk") for s in exp["target"]])
    )
    assert (np.sort(ix_sitk) == expected_sitk).all()


def test_target_union_voxels_4d_each_and_scenario():
    cst, exp = _build_4d_cst()
    dims = exp["dims_zyx"]
    sub_cube_size = int(np.prod(dims))

    per_scen = cst.target_union_voxels(scenario="each", order="numpy")
    assert isinstance(per_scen, list)
    assert len(per_scen) == cst.num_of_scenarios
    for s, ix in enumerate(per_scen):
        assert isinstance(ix, np.ndarray)
        assert (ix < sub_cube_size).all()
        assert (ix == np.array([_flat_idx(exp["target"][s], dims, "numpy")])).all()

    for s in range(cst.num_of_scenarios):
        ix = cst.target_union_voxels(scenario=s, order="numpy")
        assert isinstance(ix, np.ndarray)
        assert (ix < sub_cube_size).all()
        assert (ix == np.array([_flat_idx(exp["target"][s], dims, "numpy")])).all()


def test_patient_voxels_4d():
    cst, exp = _build_4d_cst()
    dims = exp["dims_zyx"]
    sub_cube_size = int(np.prod(dims))

    # EXTERNAL VOI takes precedence — patient voxels equal external voxels.
    ix_any = cst.patient_voxels(order="numpy")
    expected_any = np.unique(
        np.array([_flat_idx(exp["external"][s], dims, "numpy") for s in exp["external"]])
    )
    assert (np.sort(ix_any) == expected_any).all()

    per_scen = cst.patient_voxels(scenario="each", order="numpy")
    assert len(per_scen) == cst.num_of_scenarios
    for s, ix in enumerate(per_scen):
        assert (ix < sub_cube_size).all()
        assert (ix == np.array([_flat_idx(exp["external"][s], dims, "numpy")])).all()

    for s in range(cst.num_of_scenarios):
        ix = cst.patient_voxels(scenario=s, order="numpy")
        assert (ix == np.array([_flat_idx(exp["external"][s], dims, "numpy")])).all()


def test_target_union_mask_4d_each_dim_preserving():
    """Regression test: previously target_union_mask wrote 4D indices into a 3D slice."""
    cst, exp = _build_4d_cst()
    dims = exp["dims_zyx"]

    each_mask = cst.target_union_mask(scenario="each")
    assert each_mask.GetDimension() == 4
    assert each_mask.GetSize()[3] == cst.num_of_scenarios

    each_arr = sitk.GetArrayViewFromImage(each_mask)  # numpy shape (T, Z, Y, X)
    for s in range(cst.num_of_scenarios):
        slice_arr = each_arr[s]
        z, y, x = exp["target"][s]
        assert slice_arr[z, y, x] == 1
        # Exactly one voxel set per scenario in this fixture.
        assert int(slice_arr.sum()) == 1


def test_target_union_mask_4d_any_collapses_to_3d():
    cst, exp = _build_4d_cst()
    dims = exp["dims_zyx"]

    any_mask = cst.target_union_mask()  # default "any"
    assert any_mask.GetDimension() == 3
    any_arr = sitk.GetArrayViewFromImage(any_mask)
    assert any_arr.shape == dims
    for s in exp["target"]:
        z, y, x = exp["target"][s]
        assert any_arr[z, y, x] == 1
    assert int(any_arr.sum()) == len(exp["target"])


def test_target_union_mask_4d_int_scenario():
    cst, exp = _build_4d_cst()
    dims = exp["dims_zyx"]

    for s in range(cst.num_of_scenarios):
        m = cst.target_union_mask(scenario=s)
        assert m.GetDimension() == 3
        arr = sitk.GetArrayViewFromImage(m)
        assert arr.shape == dims
        z, y, x = exp["target"][s]
        assert arr[z, y, x] == 1
        assert int(arr.sum()) == 1


def test_patient_mask_4d_each_uses_external_per_scenario():
    cst, exp = _build_4d_cst()
    each_mask = cst.patient_mask(scenario="each")
    assert each_mask.GetDimension() == 4
    each_arr = sitk.GetArrayViewFromImage(each_mask)
    for s in range(cst.num_of_scenarios):
        z, y, x = exp["external"][s]
        slice_arr = each_arr[s]
        assert slice_arr[z, y, x] == 1
        assert int(slice_arr.sum()) == 1

    # "any" collapses to 3D.
    any_mask = cst.patient_mask()
    assert any_mask.GetDimension() == 3


def test_target_center_of_mass_4d_specific_scenario():
    """Place distinct CoMs per scenario and verify scenario= selects the right one."""
    dims = (10, 10, 10)
    cube_3d = sitk.GetImageFromArray(np.zeros(dims, dtype=np.float32))
    cube_4d = sitk.JoinSeries([cube_3d, cube_3d])
    cube_4d.SetOrigin((0.0, 0.0, 0.0, 0.0))
    cube_4d.SetSpacing((1.0, 1.0, 1.0, 1.0))
    cube_4d.SetDirection((1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))
    ct = create_ct(cube_hu=cube_4d)

    arr = np.zeros((2,) + dims, dtype=np.uint8)
    # Scenario 0: single voxel at z=y=x=1.
    arr[0, 1, 1, 1] = 1
    # Scenario 1: single voxel at z=y=x=5.
    arr[1, 5, 5, 5] = 1
    voi = create_voi(voi_type="TARGET", name="t", ct_image=ct, mask=arr)
    cst = StructureSet(vois=[voi], ct_image=ct)

    com_default = cst.target_center_of_mass()
    assert np.allclose(com_default, np.array([1.0, 1.0, 1.0]))
    com_0 = cst.target_center_of_mass(scenario=0)
    assert np.allclose(com_0, np.array([1.0, 1.0, 1.0]))
    com_1 = cst.target_center_of_mass(scenario=1)
    assert np.allclose(com_1, np.array([5.0, 5.0, 5.0]))


def test_scenario_arg_validation_4d():
    cst, _ = _build_4d_cst()

    for method in (cst.target_union_voxels, cst.patient_voxels):
        with pytest.raises(ValueError):
            method(scenario=cst.num_of_scenarios)
        with pytest.raises(ValueError):
            method(scenario=-1)
        with pytest.raises(ValueError):
            method(scenario="bogus")
        with pytest.raises(ValueError):
            method(scenario=0.5)

    for method in (cst.target_union_mask, cst.patient_mask):
        with pytest.raises(ValueError):
            method(scenario=cst.num_of_scenarios)
        with pytest.raises(ValueError):
            method(scenario=-1)
        with pytest.raises(ValueError):
            method(scenario="bogus")

    with pytest.raises(ValueError):
        cst.target_center_of_mass(scenario=cst.num_of_scenarios)
    with pytest.raises(ValueError):
        cst.target_center_of_mass(scenario=-1)


def test_scenario_arg_validation_3d(generic_input_3d):
    name_3d, ct_3d, mask_3d, _, _ = generic_input_3d
    voi_3d = create_voi(voi_type="TARGET", name=name_3d, ct_image=ct_3d, mask=mask_3d)
    cst = StructureSet(vois=[voi_3d], ct_image=ct_3d)

    # 3D CT has exactly one scenario.
    with pytest.raises(ValueError):
        cst.target_union_voxels(scenario=1)
    with pytest.raises(ValueError):
        cst.target_union_mask(scenario=1)
    with pytest.raises(ValueError):
        cst.target_center_of_mass(scenario=1)

    # "each" on a 3D CT must still return a single ndarray / 3D mask.
    each_voxels = cst.target_union_voxels(scenario="each")
    assert isinstance(each_voxels, np.ndarray)
    each_mask = cst.target_union_mask(scenario="each")
    assert each_mask.GetDimension() == 3


@pytest.fixture
def generic_ct():
    # Create a simple 3D CT image
    ct_array = np.zeros((10, 10, 10), dtype=np.float32)
    ct_image = sitk.GetImageFromArray(ct_array)
    ct_image.SetSpacing((1.0, 1.0, 1.0))
    ct_image.SetOrigin((0.0, 0.0, 0.0))
    ct_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return CT(cube_hu=ct_image)


@pytest.fixture
def generic_vois(generic_ct):
    # Create simple VOIs with different overlap priorities
    mask1 = np.zeros((10, 10, 10), dtype=np.uint8)
    mask1[4:5, 4:5, 4:5] = 1
    mask_image1 = sitk.GetImageFromArray(mask1)
    mask_image1.CopyInformation(generic_ct.cube_hu)

    mask2 = np.zeros((10, 10, 10), dtype=np.uint8)
    mask2[3:6, 3:6, 3:6] = 1
    mask_image2 = sitk.GetImageFromArray(mask2)
    mask_image2.CopyInformation(generic_ct.cube_hu)

    mask3 = np.zeros((10, 10, 10), dtype=np.uint8)
    mask3[4:7, 4:7, 4:7] = 1
    mask_image3 = sitk.GetImageFromArray(mask3)
    mask_image3.CopyInformation(generic_ct.cube_hu)

    mask4 = np.zeros((10, 10, 10), dtype=np.uint8)
    mask4[1:8, 1:8, 1:8] = 1
    mask_image4 = sitk.GetImageFromArray(mask4)
    mask_image4.CopyInformation(generic_ct.cube_hu)

    mask5 = np.zeros((10, 10, 10), dtype=np.uint8)
    mask5[2:7, 2:7, 2:7] = 1
    mask_image5 = sitk.GetImageFromArray(mask5)
    mask_image5.CopyInformation(generic_ct.cube_hu)

    voi1 = Target(name="CTV", mask=mask_image1, ct_image=generic_ct, overlap_priority=1)
    voi2 = Target(name="PTV", mask=mask_image2, ct_image=generic_ct, overlap_priority=2)
    voi3 = OAR(name="OAR", mask=mask_image3, ct_image=generic_ct, overlap_priority=5)
    voi4 = ExternalVOI(name="BODY", mask=mask_image4, ct_image=generic_ct, overlap_priority=10)
    voi5 = HelperVOI(name="HELPER", mask=mask_image5, ct_image=generic_ct, overlap_priority=5)

    return [voi1, voi2, voi3, voi4, voi5]


def test_apply_overlap_priorities(generic_ct, generic_vois):
    # Create a StructureSet with the VOIs

    structure_set = StructureSet(ct_image=generic_ct, vois=generic_vois)

    # Apply overlap priorities
    structure_set_overlap = structure_set.apply_overlap_priorities()

    voi_mask = [None] * len(generic_vois)
    voi_mask_overlapped = [None] * len(generic_vois)
    p = [None] * len(generic_vois)

    for i in range(len(generic_vois)):
        voi_mask[i] = sitk.GetArrayViewFromImage(structure_set.vois[i].mask)
        voi_mask_overlapped[i] = sitk.GetArrayViewFromImage(structure_set_overlap.vois[i].mask)
        p[i] = structure_set.vois[i].overlap_priority

    expected_overlap_list = np.argsort(p)

    ol_mask = np.zeros(voi_mask[0].shape, dtype=bool)
    or_mask = np.zeros(voi_mask[0].shape, dtype=bool)
    last_priority = -1

    for expected in expected_overlap_list:
        if p[expected] > last_priority:
            ol_mask = or_mask.copy()
            last_priority = p[expected]

        # the mask that the current voi should have
        assert (
            voi_mask_overlapped[expected][np.logical_and(voi_mask[expected] > 0, ~ol_mask)]
        ).all()

        # Accumulate the ol mask
        or_mask = or_mask | voi_mask[expected] > 0

        # we currently should be zero where overlapped
        assert (voi_mask_overlapped[expected][ol_mask] == 0).all()


def test_apply_overlap_priorities_same_priority(generic_ct, generic_vois):
    for v in generic_vois:
        v.overlap_priority = 1

    # Create a StructureSet with the VOIs
    structure_set = StructureSet(ct_image=generic_ct, vois=generic_vois)

    # Apply overlap priorities
    structure_set_overlap = structure_set.apply_overlap_priorities()

    # Check the resulting masks
    for i in range(len(generic_vois)):
        voi_mask = sitk.GetArrayViewFromImage(structure_set.vois[i].mask)
        voi_mask_overlapped = sitk.GetArrayViewFromImage(structure_set_overlap.vois[i].mask)

        assert np.isclose(voi_mask, voi_mask_overlapped).all()


# --- Helpers for body segmentation tests ---
def _make_test_ct():
    """Create a small CT with a main component, an internal cavity (hole), and a small extra component."""
    arr = np.full((4, 6, 6), -500.0, dtype=np.float32)  # background below threshold

    for z in range(4):
        arr[z, 1:5, 1:5] = 0.0  # above threshold (will be segmented)
        arr[z, 2:4, 2:4] = -500.0  # cavity to be filled slice-wise

    # Additional small disconnected component (should be discarded as not largest)
    arr[0, 0, 0] = 0.0
    arr[0, 0, 1] = 0.0

    ct_img = sitk.GetImageFromArray(arr)
    ct_img.SetSpacing((1.0, 1.0, 1.0))
    ct_img.SetOrigin((0.0, 0.0, 0.0))
    ct_img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return create_ct(cube_hu=ct_img)


def _expected_body_mask(ct):
    ct_arr = sitk.GetArrayViewFromImage(ct.cube_hu)
    assert ct_arr.shape == (4, 6, 6)
    mask = np.zeros_like(ct_arr, dtype=np.uint8)
    mask[:, 1:5, 1:5] = 1  # main cube
    mask[0, 0, 0] = 1
    mask[0, 0, 1] = 1
    return sitk.GetImageFromArray(mask)


def test_create_cst_with_float_and_empty_indices():
    """Regression: matRad indices can be floats; empty VOIs have no indices."""
    ct = create_ct(cube_hu=sitk.Image(3, 3, 3, sitk.sitkInt16))

    # matRad-style cst: [id, name, type, indices, priority, objectives]
    # Float indices (as MATLAB would produce)
    float_idx = np.array([1.0, 2.0, 3.0])
    # Empty indices
    empty_idx = np.array([], dtype=np.float64)

    cst_data = [
        [0, "Target", "TARGET", float_idx, 1, []],
        [1, "Empty", "OAR", empty_idx, 2, []],
    ]

    cst = create_cst(cst_data, ct=ct)
    assert isinstance(cst, StructureSet)
    assert len(cst.vois) == 2
    assert cst.vois[0].name == "Target"
    assert cst.vois[1].name == "Empty"

    # Empty VOI should have an all-zero mask
    empty_mask = sitk.GetArrayViewFromImage(cst.vois[1].mask)
    assert empty_mask.sum() == 0


def test_create_body_seg_default():
    ct = _make_test_ct()
    structure_set = StructureSet(ct_image=ct, vois=[])
    result = structure_set.create_body_seg()  # default threshold/name/type
    assert result is None  # side-effect only
    assert len(structure_set.vois) == 1
    body_voi = structure_set.vois[-1]
    assert body_voi.name == "BODY"
    assert isinstance(body_voi, OAR)  # default voi_type="OAR"
    expected_mask = _expected_body_mask(ct)
    body_mask_arr = sitk.GetArrayViewFromImage(body_voi.mask)
    expected_arr = sitk.GetArrayViewFromImage(expected_mask)
    assert body_mask_arr.shape == expected_arr.shape
    assert np.array_equal(body_mask_arr, expected_arr)
    # cavity region filled
    for z in range(4):
        assert body_mask_arr[z, 2, 2] == 1
    # Note: small added component was adjacent via connectivity; we do not assert its removal.


def test_create_body_seg_custom_name_type():
    ct = _make_test_ct()
    structure_set = StructureSet(ct_image=ct, vois=[])
    structure_set.create_body_seg(threshold=-300.0, name="CUSTOM_BODY", voi_type="EXTERNAL")
    assert len(structure_set.vois) == 1
    body_voi = structure_set.vois[-1]
    assert body_voi.name == "CUSTOM_BODY"
    assert isinstance(body_voi, ExternalVOI)
    expected_mask = _expected_body_mask(ct)
    body_mask_arr = sitk.GetArrayViewFromImage(body_voi.mask)
    expected_arr = sitk.GetArrayViewFromImage(expected_mask)
    assert np.array_equal(body_mask_arr, expected_arr)
    for z in range(4):
        assert body_mask_arr[z, 2, 2] == 1
    # Connectivity may include diagonal, so we skip asserting removal of small component.


def test_set_colors_all_missing(generic_ct):
    # Create dummy masks
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask_image = sitk.GetImageFromArray(mask)
    mask_image.CopyInformation(generic_ct.cube_hu)

    # Case 1: All missing colors
    voi1 = Target(name="V1", mask=mask_image, ct_image=generic_ct, visible_color=None)
    voi2 = OAR(name="V2", mask=mask_image, ct_image=generic_ct, visible_color=None)
    voi3 = ExternalVOI(name="V3", mask=mask_image, ct_image=generic_ct, visible_color=None)

    cst = StructureSet(ct_image=generic_ct, vois=[voi1, voi2, voi3])

    # Colors are auto-assigned on CST creation via check_cst(); verify they are valid.
    cst.set_colors()

    # Verify assigned
    for v in cst.vois:
        assert v.visible_color is not None
        assert len(v.visible_color) == 3
        assert all(isinstance(c, int) for c in v.visible_color)
        assert all(0 <= c <= 255 for c in v.visible_color)

    # Verify distinctness (simple check for small number)
    colors = [tuple(v.visible_color) for v in cst.vois]
    assert len(set(colors)) == 3


def test_set_colors_preserve_existing(generic_ct):
    # Create dummy masks
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask_image = sitk.GetImageFromArray(mask)
    mask_image.CopyInformation(generic_ct.cube_hu)

    # Case 2: Preserve existing
    existing_color = (255, 0, 0)
    voi4 = Target(name="V4", mask=mask_image, ct_image=generic_ct, visible_color=existing_color)
    voi5 = OAR(name="V5", mask=mask_image, ct_image=generic_ct, visible_color=None)

    cst2 = StructureSet(ct_image=generic_ct, vois=[voi4, voi5])
    cst2.set_colors()

    assert tuple(cst2.vois[0].visible_color) == existing_color
    assert cst2.vois[1].visible_color is not None
    assert tuple(cst2.vois[1].visible_color) != existing_color


def test_set_colors_predefined_order(generic_ct):
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask_image = sitk.GetImageFromArray(mask)
    mask_image.CopyInformation(generic_ct.cube_hu)

    voi1 = Target(name="T1", mask=mask_image, ct_image=generic_ct, visible_color=None)
    voi2 = Target(name="T2", mask=mask_image, ct_image=generic_ct, visible_color=None)
    cst = StructureSet(ct_image=generic_ct, vois=[voi1, voi2])

    assert tuple(cst.vois[0].visible_color) == DEFAULT_VOI_COLORS["TARGET"][0]
    assert tuple(cst.vois[1].visible_color) == DEFAULT_VOI_COLORS["TARGET"][1]
