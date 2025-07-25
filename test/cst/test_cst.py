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
    voi3 = OAR(name="OAR", mask=mask_image3, ct_image=generic_ct)
    voi4 = ExternalVOI(name="BODY", mask=mask_image4, ct_image=generic_ct)
    voi5 = HelperVOI(name="HELPER", mask=mask_image5, ct_image=generic_ct)

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

    for expected in expected_overlap_list:
        # the mask that the current voi should have
        assert (
            voi_mask_overlapped[expected][np.logical_and(voi_mask[expected] > 0, ~ol_mask)]
        ).all()

        # Accumulate the ol mask
        ol_mask = ol_mask | voi_mask[expected] > 0

        # we currently should be zero where overlapped
        assert not (voi_mask_overlapped[expected][ol_mask] == 0).all()
        # the accumulated ol mask


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
