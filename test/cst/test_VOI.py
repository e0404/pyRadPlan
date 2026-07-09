import pytest
import numpy as np
import SimpleITK as sitk
from pyRadPlan.cst import OAR, Target, HelperVOI, ExternalVOI, create_voi


def test_create_voi_no_args():
    with pytest.raises(ValueError):
        create_voi()


def test_target_constructor_empty_args():
    with pytest.raises(ValueError):
        Target()


def test_oar_constructor_empty_args():
    with pytest.raises(ValueError):
        OAR()


def test_helper_voi_constructor_empty_args():
    with pytest.raises(ValueError):
        HelperVOI()


def test_target_constructor_3d(generic_input_3d):
    name, ct, mask, alpha_x, beta_x = generic_input_3d
    target = Target(name=name, ct_image=ct, mask=mask)
    assert tuple(target.grid.origin) == ct.cube_hu.GetOrigin()
    assert tuple(target.grid.resolution_vector) == ct.cube_hu.GetSpacing()
    assert tuple(target.grid.direction_vector) == ct.cube_hu.GetDirection()
    assert isinstance(target.mask, sitk.Image)
    assert target.voi_type == "TARGET"
    assert target.mask.GetOrigin() == tuple(target.grid.origin)
    assert target.mask.GetSpacing() == tuple(target.grid.resolution_vector)
    assert target.mask.GetDirection() == tuple(target.grid.direction_vector)

    # test non default alpha_x and beta_x
    target_2 = Target(name=name, ct_image=ct, mask=mask, alpha_x=alpha_x, beta_x=beta_x)
    assert target_2.alpha_x == alpha_x
    assert target_2.beta_x == beta_x


def test_target_constructor_4d(generic_input_4d):
    name, ct, mask, alpha_x, beta_x = generic_input_4d
    a = ct.cube_hu.GetOrigin()
    b = ct.cube_hu.GetSpacing()
    c = ct.cube_hu.GetDirection()
    d = ct.cube_hu.GetDimension()
    e = ct.cube_hu.GetSize()

    target = Target(name=name, ct_image=ct, mask=mask)

    assert tuple(target.grid.origin) == ct.cube_hu.GetOrigin()
    assert tuple(target.grid.resolution_vector) == ct.cube_hu.GetSpacing()
    assert tuple(target.grid.direction_vector) == ct.cube_hu.GetDirection()
    assert isinstance(target.mask, sitk.Image)
    assert target.voi_type == "TARGET"

    # test non default alpha_x and beta_x
    target_2 = Target(name=name, ct_image=ct, mask=mask, alpha_x=alpha_x, beta_x=beta_x)
    assert target_2.alpha_x == alpha_x
    assert target_2.beta_x == beta_x


def test_target_constructor_4d_np(generic_input_4d):
    name, ct, mask, alpha_x, beta_x = generic_input_4d
    target = Target(name=name, ct_image=ct, mask=sitk.GetArrayFromImage(mask))
    assert tuple(target.grid.origin) == ct.cube_hu.GetOrigin()
    assert tuple(target.grid.resolution_vector) == ct.cube_hu.GetSpacing()
    assert tuple(target.grid.direction_vector) == ct.cube_hu.GetDirection()
    assert isinstance(target.mask, sitk.Image)
    assert target.voi_type == "TARGET"

    # test non default alpha_x and beta_x
    target_2 = Target(name=name, ct_image=ct, mask=mask, alpha_x=alpha_x, beta_x=beta_x)
    assert target_2.alpha_x == alpha_x
    assert target_2.beta_x == beta_x


def test_mixed_constructors(generic_input_3d, generic_input_4d):
    name_3d, ct_3d, mask_3d, _, _ = generic_input_3d
    name_4d, ct_4d, mask_4d, _, _ = generic_input_4d
    with pytest.raises(ValueError):
        target = Target(name=name_3d, ct_image=ct_3d, mask=mask_4d)

    with pytest.raises(ValueError):
        target = Target(name=name_4d, ct_image=ct_4d, mask=mask_3d)


def test_oar_constructor(generic_input_3d):
    name, ct, mask, alpha_x, beta_x = generic_input_3d
    oar = OAR(name=name, ct_image=ct, mask=mask)
    assert tuple(oar.grid.origin) == ct.cube_hu.GetOrigin()
    assert tuple(oar.grid.resolution_vector) == ct.cube_hu.GetSpacing()
    assert tuple(oar.grid.direction_vector) == ct.cube_hu.GetDirection()
    assert isinstance(oar.mask, sitk.Image)
    assert oar.voi_type == "OAR"

    # test non default alpha_x and beta_x
    oar_2 = OAR(name=name, ct_image=ct, mask=mask, alpha_x=alpha_x, beta_x=beta_x)
    assert oar_2.alpha_x == alpha_x
    assert oar_2.beta_x == beta_x


def test_helper_voi_constructor(generic_input_3d):
    name, ct, mask, alpha_x, beta_x = generic_input_3d
    helper_voi = HelperVOI(name=name, ct_image=ct, mask=mask)
    assert tuple(helper_voi.grid.origin) == ct.cube_hu.GetOrigin()
    assert tuple(helper_voi.grid.resolution_vector) == ct.cube_hu.GetSpacing()
    assert tuple(helper_voi.grid.direction_vector) == ct.cube_hu.GetDirection()
    assert isinstance(helper_voi.mask, sitk.Image)
    assert helper_voi.voi_type == "HELPER"

    # test non default alpha_x and beta_x
    helper_voi_2 = HelperVOI(name=name, ct_image=ct, mask=mask, alpha_x=alpha_x, beta_x=beta_x)
    assert helper_voi_2.alpha_x == alpha_x
    assert helper_voi_2.beta_x == beta_x


def test_external_voi_constructor(generic_input_3d):
    name, ct, mask, alpha_x, beta_x = generic_input_3d
    external_voi = ExternalVOI(name=name, ct_image=ct, mask=mask)
    assert tuple(external_voi.grid.origin) == ct.cube_hu.GetOrigin()
    assert tuple(external_voi.grid.resolution_vector) == ct.cube_hu.GetSpacing()
    assert tuple(external_voi.grid.direction_vector) == ct.cube_hu.GetDirection()
    assert isinstance(external_voi.mask, sitk.Image)
    assert external_voi.voi_type == "EXTERNAL"

    # test non default alpha_x and beta_x
    external_voi_2 = ExternalVOI(name=name, ct_image=ct, mask=mask, alpha_x=alpha_x, beta_x=beta_x)
    assert external_voi_2.alpha_x == alpha_x
    assert external_voi_2.beta_x == beta_x


def test_voi_idx_wrong_shape(generic_input_3d):
    name, ct, _, _, _ = generic_input_3d
    with pytest.raises(ValueError):
        Target(name=name, ct_image=ct, mask=np.array([[1], [1], [1]]))


def test_voi_idx_wrong_dim(generic_input_3d):
    name, ct, _, _, _ = generic_input_3d
    with pytest.raises(ValueError):
        Target(name=name, ct_image=ct, mask=np.array([1]))

    with pytest.raises(ValueError):
        Target(name=name, ct_image=ct, mask=np.array([1] * 5)[:, None])


def test_voi_idx_non_int_idx(generic_input_3d):
    name, ct, _, _, _ = generic_input_3d
    with pytest.raises(ValueError):
        Target(name=name, ct_image=ct, mask=[[1.2, 3, 4]])


def test_voi_idx_wrong_dtype(generic_input_3d):
    name, ct, mask, _, _ = generic_input_3d
    mask_float = mask.astype("float32")
    mask_float = sitk.GetImageFromArray(mask_float)
    with pytest.raises(ValueError):
        Target(name=name, ct_image=ct, mask=mask_float)

    mask = np.zeros_like(mask) * 3.0
    with pytest.raises(ValueError):
        Target(name=name, ct_image=ct, mask=mask)


def test_voi_idx_wrong_input(generic_input_3d):
    name, ct, _, _, _ = generic_input_3d
    with pytest.raises(ValueError):
        Target(name=name, ct_image=ct, mask=[[2], [3], [4]])


def test_create_voi_target_from_dict(generic_input_3d):
    name, ct, mask, _, _ = generic_input_3d
    voi = create_voi(data={"voi_type": "TARGET", "name": name, "ct_image": ct, "mask": mask})
    assert isinstance(voi, Target)
    assert tuple(voi.grid.origin) == ct.cube_hu.GetOrigin()
    assert tuple(voi.grid.resolution_vector) == ct.cube_hu.GetSpacing()
    assert tuple(voi.grid.direction_vector) == ct.cube_hu.GetDirection()
    assert isinstance(voi.mask, sitk.Image)
    assert voi.voi_type == "TARGET"


def test_create_voi_oar_from_dict(generic_input_3d):
    name, ct, mask, _, _ = generic_input_3d
    voi = create_voi(data={"voi_type": "OAR", "name": name, "ct_image": ct, "mask": mask})
    assert isinstance(voi, OAR)
    assert tuple(voi.grid.origin) == ct.cube_hu.GetOrigin()
    assert tuple(voi.grid.resolution_vector) == ct.cube_hu.GetSpacing()
    assert tuple(voi.grid.direction_vector) == ct.cube_hu.GetDirection()
    assert isinstance(voi.mask, sitk.Image)
    assert voi.voi_type == "OAR"


def test_create_voi_helper_from_dict(generic_input_3d):
    name, ct, mask, _, _ = generic_input_3d
    voi = create_voi(data={"voi_type": "HELPER", "name": name, "ct_image": ct, "mask": mask})
    assert isinstance(voi, HelperVOI)
    assert tuple(voi.grid.origin) == ct.cube_hu.GetOrigin()
    assert tuple(voi.grid.resolution_vector) == ct.cube_hu.GetSpacing()
    assert tuple(voi.grid.direction_vector) == ct.cube_hu.GetDirection()
    assert isinstance(voi.mask, sitk.Image)
    assert voi.voi_type == "HELPER"


def test_create_voi_invalid_voi_type(generic_input_3d):
    name, ct, mask, _, _ = generic_input_3d
    with pytest.raises(ValueError):
        create_voi(data={"voi_type": "invalid", "name": name, "ct_image": ct, "mask": mask})


def test_create_voi_no_voi_type(generic_input_3d):
    name, ct, mask, _, _ = generic_input_3d
    with pytest.raises(ValueError):
        create_voi(data={"name": name, "ct_image": ct, "mask": mask})


def test_create_voi_no_data():
    with pytest.raises(ValueError):
        create_voi()


def test_create_voi_from_VOI(generic_input_3d):
    name, ct, mask, _, _ = generic_input_3d
    voi = create_voi(data={"voi_type": "HELPER", "name": name, "ct_image": ct, "mask": mask})
    voi_2 = create_voi(voi)
    assert voi == voi_2


def test_create_target_from_kwargs(generic_input_3d):
    name, ct, mask, _, _ = generic_input_3d
    voi = create_voi(voi_type="TARGET", name=name, ct_image=ct, mask=mask)
    assert isinstance(voi, Target)
    assert tuple(voi.grid.origin) == ct.cube_hu.GetOrigin()
    assert tuple(voi.grid.resolution_vector) == ct.cube_hu.GetSpacing()
    assert tuple(voi.grid.direction_vector) == ct.cube_hu.GetDirection()
    assert isinstance(voi.mask, sitk.Image)
    assert voi.voi_type == "TARGET"


def test_create_oar_from_kwargs(generic_input_3d):
    name, ct, mask, _, _ = generic_input_3d
    voi = create_voi(voi_type="OAR", name=name, ct_image=ct, mask=mask)
    assert isinstance(voi, OAR)
    assert tuple(voi.grid.origin) == ct.cube_hu.GetOrigin()
    assert tuple(voi.grid.resolution_vector) == ct.cube_hu.GetSpacing()
    assert tuple(voi.grid.direction_vector) == ct.cube_hu.GetDirection()
    assert isinstance(voi.mask, sitk.Image)
    assert voi.voi_type == "OAR"


def test_create_helper_from_kwargs(generic_input_3d):
    name, ct, mask, _, _ = generic_input_3d
    voi = create_voi(voi_type="HELPER", name=name, ct_image=ct, mask=mask)
    assert isinstance(voi, HelperVOI)
    assert tuple(voi.grid.origin) == ct.cube_hu.GetOrigin()
    assert tuple(voi.grid.resolution_vector) == ct.cube_hu.GetSpacing()
    assert tuple(voi.grid.direction_vector) == ct.cube_hu.GetDirection()
    assert isinstance(voi.mask, sitk.Image)
    assert voi.voi_type == "HELPER"


def test_create_voi_invalid_voi_type_kwargs(generic_input_3d):
    name, ct, mask, _, _ = generic_input_3d
    with pytest.raises(ValueError):
        create_voi(voi_type="invalid", name=name, ct_image=ct, mask=mask)


def test_create_voi_no_voi_type_kwargs(generic_input_3d):
    name, ct, mask, _, _ = generic_input_3d
    with pytest.raises(ValueError):
        create_voi(name=name, ct_image=ct, mask=mask)


def test_voi_indices(generic_input_3d, generic_input_4d):
    name_3d, ct_3d, mask_3d, _, _ = generic_input_3d
    name_4d, ct_4d, mask_4d, _, _ = generic_input_4d
    voi_3d = create_voi(voi_type="TARGET", name=name_3d, ct_image=ct_3d, mask=mask_3d)
    voi_4d = create_voi(voi_type="TARGET", name=name_4d, ct_image=ct_4d, mask=mask_4d)
    assert (voi_3d.indices == np.array([5000])).all()
    assert (sitk.GetArrayViewFromImage(voi_3d.mask).ravel(order="F")[voi_3d.indices] == 1).all()
    assert (voi_4d.indices == np.array([103, 10000])).all()
    assert (sitk.GetArrayViewFromImage(voi_4d.mask).ravel(order="F")[voi_4d.indices] == 1).all()


def test_voi_indices_np(generic_input_3d, generic_input_4d):
    name_3d, ct_3d, mask_3d, _, _ = generic_input_3d
    name_4d, ct_4d, mask_4d, _, _ = generic_input_4d
    voi_3d = create_voi(voi_type="TARGET", name=name_3d, ct_image=ct_3d, mask=mask_3d)
    voi_4d = create_voi(voi_type="TARGET", name=name_4d, ct_image=ct_4d, mask=mask_4d)
    assert (voi_3d.indices_numpy == np.array([1])).all()
    assert (
        sitk.GetArrayViewFromImage(voi_3d.mask).ravel(order="C")[voi_3d.indices_numpy] == 1
    ).all()
    assert (voi_4d.indices_numpy == np.array([1, 510100])).all()
    assert (
        sitk.GetArrayViewFromImage(voi_4d.mask).ravel(order="C")[voi_4d.indices_numpy] == 1
    ).all()


def test_voi_get_indices_by_order(generic_input_3d, generic_input_4d):
    name_3d, ct_3d, mask_3d, _, _ = generic_input_3d
    name_4d, ct_4d, mask_4d, _, _ = generic_input_4d
    voi_3d = create_voi(voi_type="TARGET", name=name_3d, ct_image=ct_3d, mask=mask_3d)
    voi_4d = create_voi(voi_type="TARGET", name=name_4d, ct_image=ct_4d, mask=mask_4d)
    assert (voi_3d.get_indices(order="numpy") == voi_3d.indices_numpy).all()
    assert (voi_4d.get_indices(order="numpy") == voi_4d.indices_numpy).all()
    assert (voi_3d.get_indices(order="sitk") == voi_3d.indices).all()
    assert (voi_4d.get_indices(order="sitk") == voi_4d.indices).all()


def test_voi_numpy_mask(generic_input_3d, generic_input_4d):
    name_3d, ct_3d, mask_3d, _, _ = generic_input_3d
    name_4d, ct_4d, mask_4d, _, _ = generic_input_4d
    voi_3d = create_voi(voi_type="TARGET", name=name_3d, ct_image=ct_3d, mask=mask_3d)
    voi_4d = create_voi(voi_type="TARGET", name=name_4d, ct_image=ct_4d, mask=mask_4d)
    assert (voi_3d._numpy_mask == mask_3d).all()
    assert (voi_4d._numpy_mask == sitk.GetArrayViewFromImage(mask_4d)).all()


def test_scenario_indices(generic_input_3d, generic_input_4d):
    name_3d, ct_3d, mask_3d, _, _ = generic_input_3d
    name_4d, ct_4d, mask_4d, _, _ = generic_input_4d
    voi_3d = create_voi(voi_type="TARGET", name=name_3d, ct_image=ct_3d, mask=mask_3d)
    voi_4d = create_voi(voi_type="TARGET", name=name_4d, ct_image=ct_4d, mask=mask_4d)

    # Default (scenario=None): always return a list of per-scenario index arrays.
    scen_3d = voi_3d.scenario_indices()
    assert isinstance(scen_3d, list)
    assert len(scen_3d) == 1
    assert (scen_3d[0] == np.array([1])).all()

    scen_3d_sitk = voi_3d.scenario_indices(order="sitk")
    assert isinstance(scen_3d_sitk, list)
    assert len(scen_3d_sitk) == 1
    assert (scen_3d_sitk[0] == np.array([5000])).all()

    scen_4d = voi_4d.scenario_indices()
    assert isinstance(scen_4d, list)
    assert len(scen_4d) == 2
    assert (scen_4d[0] == np.array([1])).all()
    assert (scen_4d[1] == np.array([10100])).all()

    scen_4d_sitk = voi_4d.scenario_indices(order="sitk")
    assert isinstance(scen_4d_sitk, list)
    assert len(scen_4d_sitk) == 2
    assert (scen_4d_sitk[0] == np.array([5000])).all()
    assert (scen_4d_sitk[1] == np.array([51])).all()

    with pytest.raises(ValueError):
        voi_3d.scenario_indices(order="invalid")
    with pytest.raises(ValueError):
        voi_4d.scenario_indices(order="invalid")


def test_scenario_indices_specific_scenario(generic_input_3d, generic_input_4d):
    """A specific scenario index returns the 3D sub-cube indices, not a list."""
    name_3d, ct_3d, mask_3d, _, _ = generic_input_3d
    name_4d, ct_4d, mask_4d, _, _ = generic_input_4d
    voi_3d = create_voi(voi_type="TARGET", name=name_3d, ct_image=ct_3d, mask=mask_3d)
    voi_4d = create_voi(voi_type="TARGET", name=name_4d, ct_image=ct_4d, mask=mask_4d)

    # 3D mask: only scenario 0 is valid.
    one_3d = voi_3d.scenario_indices(scenario=0)
    assert isinstance(one_3d, np.ndarray)
    assert (one_3d == np.array([1])).all()
    assert (voi_3d.scenario_indices(scenario=0, order="sitk") == np.array([5000])).all()

    # 4D mask: each scenario returns indices into a 3D sub-cube.
    one_4d_0 = voi_4d.scenario_indices(scenario=0)
    one_4d_1 = voi_4d.scenario_indices(scenario=1)
    assert isinstance(one_4d_0, np.ndarray)
    assert isinstance(one_4d_1, np.ndarray)
    assert (one_4d_0 == np.array([1])).all()
    assert (one_4d_1 == np.array([10100])).all()
    assert (voi_4d.scenario_indices(scenario=0, order="sitk") == np.array([5000])).all()
    assert (voi_4d.scenario_indices(scenario=1, order="sitk") == np.array([51])).all()

    # Per-scenario indices must reference a 3D sub-cube, not the full 4D cube.
    sub_cube_size = int(np.prod(voi_4d.mask.GetSize()[:3]))
    for s in range(voi_4d.num_of_scenarios):
        ix_np = voi_4d.scenario_indices(scenario=s, order="numpy")
        ix_sitk = voi_4d.scenario_indices(scenario=s, order="sitk")
        assert (ix_np < sub_cube_size).all()
        assert (ix_sitk < sub_cube_size).all()


def test_scenario_indices_out_of_range(generic_input_3d, generic_input_4d):
    name_3d, ct_3d, mask_3d, _, _ = generic_input_3d
    name_4d, ct_4d, mask_4d, _, _ = generic_input_4d
    voi_3d = create_voi(voi_type="TARGET", name=name_3d, ct_image=ct_3d, mask=mask_3d)
    voi_4d = create_voi(voi_type="TARGET", name=name_4d, ct_image=ct_4d, mask=mask_4d)

    # 3D mask has only one scenario.
    with pytest.raises(ValueError):
        voi_3d.scenario_indices(scenario=1)
    with pytest.raises(ValueError):
        voi_3d.scenario_indices(scenario=-1)

    # 4D mask: scenario beyond num_of_scenarios.
    with pytest.raises(ValueError):
        voi_4d.scenario_indices(scenario=voi_4d.num_of_scenarios)
    with pytest.raises(ValueError):
        voi_4d.scenario_indices(scenario=-1)

    # Non-integer scenario index.
    with pytest.raises(ValueError):
        voi_4d.scenario_indices(scenario="0")
    with pytest.raises(ValueError):
        voi_4d.scenario_indices(scenario=0.0)


def test_indices_full_cube_for_4d(generic_input_4d):
    """``indices`` / ``indices_numpy`` must reference the FULL 4D cube."""
    name_4d, ct_4d, mask_4d, _, _ = generic_input_4d
    voi_4d = create_voi(voi_type="TARGET", name=name_4d, ct_image=ct_4d, mask=mask_4d)

    full_cube_size = voi_4d.mask.GetNumberOfPixels()
    sub_cube_size = int(np.prod(voi_4d.mask.GetSize()[:3]))
    # Sanity check: 4D cube is strictly larger than a single 3D sub-cube.
    assert full_cube_size == sub_cube_size * voi_4d.num_of_scenarios

    # All full-cube indices must lie within the full cube.
    assert (voi_4d.indices < full_cube_size).all()
    assert (voi_4d.indices_numpy < full_cube_size).all()

    # The mask values at the full-cube indices are 1 in the raveled cube.
    arr = sitk.GetArrayViewFromImage(voi_4d.mask)
    assert (arr.ravel(order="F")[voi_4d.indices] == 1).all()
    assert (arr.ravel(order="C")[voi_4d.indices_numpy] == 1).all()

    # Numpy-order indices for the 4D cube put scenarios in contiguous blocks
    # of size sub_cube_size. The voxel set in scenario s sits in
    # [s * sub_cube_size, (s + 1) * sub_cube_size).
    per_scen = voi_4d.scenario_indices(order="numpy")
    expected_full_numpy = np.sort(
        np.concatenate([per_scen[s] + s * sub_cube_size for s in range(voi_4d.num_of_scenarios)])
    )
    assert (np.sort(voi_4d.indices_numpy) == expected_full_numpy).all()


def test_num_of_scenarios(generic_input_3d, generic_input_4d):
    name_3d, ct_3d, mask_3d, _, _ = generic_input_3d
    name_4d, ct_4d, mask_4d, _, _ = generic_input_4d
    voi_3d = create_voi(voi_type="TARGET", name=name_3d, ct_image=ct_3d, mask=mask_3d)
    voi_4d = create_voi(voi_type="TARGET", name=name_4d, ct_image=ct_4d, mask=mask_4d)
    assert voi_3d.num_of_scenarios == 1
    assert voi_4d.num_of_scenarios == voi_4d.mask.GetSize()[3]
    assert len(voi_3d.scenario_indices()) == voi_3d.num_of_scenarios
    assert len(voi_4d.scenario_indices()) == voi_4d.num_of_scenarios


def test_masked_ct(generic_input_3d, generic_input_4d):
    name_3d, ct_3d, mask_3d, _, _ = generic_input_3d
    name_4d, ct_4d, mask_4d, _, _ = generic_input_4d
    voi_3d = create_voi(voi_type="TARGET", name=name_3d, ct_image=ct_3d, mask=mask_3d)
    voi_4d = create_voi(voi_type="TARGET", name=name_4d, ct_image=ct_4d, mask=mask_4d)
    with pytest.raises(ValueError):
        voi_3d.mask_image(ct_3d, "invalid")
    with pytest.raises(ValueError):
        voi_4d.mask_image(ct_4d, "invalid")

    masked_ct_3d_sitk = voi_3d.mask_image(ct_3d, "sitk")
    assert isinstance(masked_ct_3d_sitk, sitk.Image)
    assert (sitk.GetArrayFromImage(masked_ct_3d_sitk) == mask_3d.astype(np.float32) * 1000).all()

    masked_ct_3d_np = voi_3d.mask_image(ct_3d, "numpy")
    assert (masked_ct_3d_np == mask_3d.astype(np.float32) * 1000).all()

    masked_ct_4d_sitk = voi_4d.mask_image(ct_4d, "sitk")
    assert isinstance(masked_ct_4d_sitk, sitk.Image)
    assert (
        sitk.GetArrayFromImage(masked_ct_4d_sitk)
        == sitk.GetArrayViewFromImage(mask_4d).astype(np.float32) * 1000
    ).all()

    masked_ct_4d_np = voi_4d.mask_image(ct_4d, "numpy")
    assert (masked_ct_4d_np == sitk.GetArrayViewFromImage(mask_4d).astype(np.float32) * 1000).all()


def test_scenario_ct_data(generic_input_3d, generic_input_4d):
    name_3d, ct_3d, mask_3d, _, _ = generic_input_3d
    name_4d, ct_4d, mask_4d, _, _ = generic_input_4d
    voi_3d = create_voi(voi_type="TARGET", name=name_3d, ct_image=ct_3d, mask=mask_3d)
    voi_4d = create_voi(voi_type="TARGET", name=name_4d, ct_image=ct_4d, mask=mask_4d)

    # Default (scenario=None): always return a list of per-scenario arrays.
    data_3d = voi_3d.scenario_ct_data(ct_3d)
    assert isinstance(data_3d, list)
    assert len(data_3d) == 1
    assert (data_3d[0] == np.array([1000])).all()

    data_4d = voi_4d.scenario_ct_data(ct_4d)
    assert isinstance(data_4d, list)
    assert len(data_4d) == 2
    assert (data_4d[0] == np.array([1000])).all()
    assert (data_4d[1] == np.array([1000])).all()

    # Specific scenario: returns a single ndarray.
    one_3d = voi_3d.scenario_ct_data(ct_3d, scenario=0)
    assert isinstance(one_3d, np.ndarray)
    assert (one_3d == np.array([1000])).all()
    for s in range(voi_4d.num_of_scenarios):
        one_4d = voi_4d.scenario_ct_data(ct_4d, scenario=s)
        assert isinstance(one_4d, np.ndarray)
        assert (one_4d == np.array([1000])).all()

    # Out-of-range scenarios raise.
    with pytest.raises(ValueError):
        voi_3d.scenario_ct_data(ct_3d, scenario=1)
    with pytest.raises(ValueError):
        voi_4d.scenario_ct_data(ct_4d, scenario=voi_4d.num_of_scenarios)
    with pytest.raises(ValueError):
        voi_4d.scenario_ct_data(ct_4d, scenario=-1)


def test_create_target_camel_case(generic_input_3d):
    name, ct, mask, alpha_x, beta_x = generic_input_3d

    voi = create_voi(
        data={
            "voiType": "TARGET",
            "name": name,
            "ctImage": ct,
            "mask": mask,
            "alphaX": alpha_x,
            "betaX": beta_x,
        }
    )
    assert isinstance(voi, Target)
    assert tuple(voi.grid.origin) == ct.cube_hu.GetOrigin()
    assert tuple(voi.grid.resolution_vector) == ct.cube_hu.GetSpacing()
    assert tuple(voi.grid.direction_vector) == ct.cube_hu.GetDirection()
    assert isinstance(voi.mask, sitk.Image)
    assert voi.voi_type == "TARGET"
    assert voi.alpha_x == alpha_x
    assert voi.beta_x == beta_x


def test_create_oar_camel_case(generic_input_3d):
    name, ct, mask, alpha_x, beta_x = generic_input_3d
    voi = create_voi(
        data={
            "voiType": "OAR",
            "name": name,
            "ctImage": ct,
            "mask": mask,
            "alphaX": alpha_x,
            "betaX": beta_x,
        }
    )
    assert isinstance(voi, OAR)
    assert tuple(voi.grid.origin) == ct.cube_hu.GetOrigin()
    assert tuple(voi.grid.resolution_vector) == ct.cube_hu.GetSpacing()
    assert tuple(voi.grid.direction_vector) == ct.cube_hu.GetDirection()
    assert isinstance(voi.mask, sitk.Image)
    assert voi.voi_type == "OAR"
    assert voi.alpha_x == alpha_x
    assert voi.beta_x == beta_x


def test_create_helper_camel_case(generic_input_3d):
    name, ct, mask, alpha_x, beta_x = generic_input_3d
    voi = create_voi(
        data={
            "voiType": "HELPER",
            "name": name,
            "ctImage": ct,
            "mask": mask,
            "alphaX": alpha_x,
            "betaX": beta_x,
        }
    )
    assert isinstance(voi, HelperVOI)
    assert tuple(voi.grid.origin) == ct.cube_hu.GetOrigin()
    assert tuple(voi.grid.resolution_vector) == ct.cube_hu.GetSpacing()
    assert tuple(voi.grid.direction_vector) == ct.cube_hu.GetDirection()
    assert isinstance(voi.mask, sitk.Image)
    assert voi.voi_type == "HELPER"
    assert voi.alpha_x == alpha_x
    assert voi.beta_x == beta_x


def test_resample_on_new_ct_binary_mask_3d():
    """Resampling with linear interpolation eroded mask boundaries. Making sure that resampling is done right"""
    from pyRadPlan.ct import create_ct

    orig_ct = create_ct(cube_hu=sitk.Image(10, 10, 10, sitk.sitkInt16))
    orig_ct.cube_hu.SetSpacing((2.0, 2.0, 2.0))

    mask_arr = np.zeros((10, 10, 10), dtype=np.uint8)
    mask_arr[4:7, 4:7, 4:7] = 1
    voi = create_voi(voi_type="TARGET", name="block", ct_image=orig_ct, mask=mask_arr)

    # round-trip must preserve the block
    new_img = sitk.Image(7, 7, 7, sitk.sitkInt16)
    new_img.SetSpacing((3.0, 3.0, 3.0))
    new_ct = create_ct(cube_hu=new_img)

    back_arr = sitk.GetArrayFromImage(
        voi._resample_on_new_ct(new_ct)._resample_on_new_ct(orig_ct).mask
    )
    assert back_arr[4:7, 4:7, 4:7].sum() == 27
    assert back_arr.sum() == 27


def test_resample_on_new_ct_binary_mask_4d():
    """Same boundary-erosion bug for 4D masks."""
    from pyRadPlan.ct import create_ct

    phase = sitk.Image(10, 10, 10, sitk.sitkInt16)
    phase.SetSpacing((2.0, 2.0, 2.0))
    orig_ct = create_ct(cube_hu=sitk.JoinSeries([phase, phase]))

    mask_arr = np.zeros((10, 10, 10), dtype=np.uint8)
    mask_arr[4:7, 4:7, 4:7] = 1
    mask_4d = sitk.JoinSeries([sitk.GetImageFromArray(mask_arr)] * 2)
    voi = create_voi(voi_type="TARGET", name="block", ct_image=orig_ct, mask=mask_4d)

    # Correspondence is a 3d CT as before
    new_img = sitk.Image(7, 7, 7, sitk.sitkInt16)
    new_img.SetSpacing((3.0, 3.0, 3.0))
    new_ct = create_ct(cube_hu=new_img)

    # Round-trip: resample to different grid and back, block must survive
    orig_3d_ct = create_ct(cube_hu=phase)
    back_arr = sitk.GetArrayFromImage(
        voi._resample_on_new_ct(new_ct)._resample_on_new_ct(orig_3d_ct).mask
    )
    assert back_arr[0, 4:7, 4:7, 4:7].sum() == 27
    assert back_arr[1, 4:7, 4:7, 4:7].sum() == 27
    assert back_arr.sum() == 54


def test_create_voi_invalid_voi_type_camel_case(generic_input_3d):
    name, ct, mask, _, _ = generic_input_3d
    with pytest.raises(ValueError):
        create_voi(voi_type="invalid", name=name, ctImage=ct, mask=mask)


# Grid source validation tests


class TestGridSourceValidation:
    """Test class for validating different grid source inputs."""

    def test_ct_only_input(self, generic_input_3d):
        """Test VOI creation with only CT input."""
        name, ct, mask, _, _ = generic_input_3d
        target = Target(name=name, ct_image=ct, mask=mask)
        assert hasattr(target, "grid")
        assert tuple(target.grid.origin) == ct.cube_hu.GetOrigin()
        assert tuple(target.grid.resolution_vector) == ct.cube_hu.GetSpacing()

    def test_grid_only_input(self, generic_input_3d):
        """Test VOI creation with only Grid input."""
        name, ct, mask, _, _ = generic_input_3d
        from pyRadPlan.core import Grid

        grid = Grid.from_sitk_image(ct.cube_hu)
        target = Target(name=name, grid=grid, mask=mask)
        assert target.grid is grid
        assert tuple(target.grid.origin) == ct.cube_hu.GetOrigin()

    def test_sitk_image_as_grid_input(self, generic_input_3d):
        """Test VOI creation with SimpleITK image as 'grid' parameter."""
        name, ct, mask, _, _ = generic_input_3d
        target = Target(name=name, grid=ct.cube_hu, mask=mask)
        assert hasattr(target, "grid")
        assert tuple(target.grid.origin) == ct.cube_hu.GetOrigin()
        assert tuple(target.grid.resolution_vector) == ct.cube_hu.GetSpacing()

    def test_sitk_image_as_image_input(self, generic_input_3d):
        """Test VOI creation with SimpleITK image as 'image' parameter."""
        name, ct, mask, _, _ = generic_input_3d
        target = Target(name=name, image=ct.cube_hu, mask=mask)
        assert hasattr(target, "grid")
        assert tuple(target.grid.origin) == ct.cube_hu.GetOrigin()
        assert tuple(target.grid.resolution_vector) == ct.cube_hu.GetSpacing()

    def test_sitk_image_as_ct_image_input(self, generic_input_3d):
        """Test VOI creation with SimpleITK image as 'ct_image' parameter."""
        name, ct, mask, _, _ = generic_input_3d
        target = Target(name=name, ct_image=ct.cube_hu, mask=mask)
        assert hasattr(target, "grid")
        assert tuple(target.grid.origin) == ct.cube_hu.GetOrigin()
        assert tuple(target.grid.resolution_vector) == ct.cube_hu.GetSpacing()

    def test_multiple_sources_grid_priority(self, generic_input_3d):
        """Test that Grid has priority when multiple sources are provided."""
        name, ct, mask, _, _ = generic_input_3d
        from pyRadPlan.core import Grid

        grid = Grid.from_sitk_image(ct.cube_hu)

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            target = Target(name=name, grid=grid, ct_image=ct, mask=mask)
            assert len(w) == 1
            assert "Multiple grid sources provided" in str(w[0].message)

        assert target.grid is grid

    def test_multiple_sources_ct_over_sitk(self, generic_input_3d):
        """Test that CT has priority over SimpleITK image when both are provided."""
        name, ct, mask, _, _ = generic_input_3d

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            target = Target(name=name, ct_image=ct, image=ct.cube_hu, mask=mask)
            assert len(w) == 1
            assert "Both CT and SimpleITK image provided" in str(w[0].message)

        # Should use CT (converted to Grid)
        assert hasattr(target, "grid")
        assert tuple(target.grid.origin) == ct.cube_hu.GetOrigin()

    def test_multiple_sources_all_three(self, generic_input_3d):
        """Test behavior when Grid, CT, and SimpleITK image are all provided."""
        name, ct, mask, _, _ = generic_input_3d
        from pyRadPlan.core import Grid

        grid = Grid.from_sitk_image(ct.cube_hu)

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            target = Target(name=name, grid=grid, ct_image=ct, image=ct.cube_hu, mask=mask)
            assert len(w) == 1
            assert "Multiple grid sources provided" in str(w[0].message)

        # Should use Grid (highest priority)
        assert target.grid is grid

    def test_dict_input_ct_only(self, generic_input_3d):
        """Test dictionary input with only CT."""
        name, ct, mask, _, _ = generic_input_3d
        target_dict = {"voi_type": "TARGET", "name": name, "ct_image": ct, "mask": mask}
        target = create_voi(data=target_dict)
        assert isinstance(target, Target)
        assert hasattr(target, "grid")

    def test_dict_input_grid_only(self, generic_input_3d):
        """Test dictionary input with only Grid."""
        name, ct, mask, _, _ = generic_input_3d
        from pyRadPlan.core import Grid

        grid = Grid.from_sitk_image(ct.cube_hu)
        target_dict = {"voi_type": "TARGET", "name": name, "grid": grid, "mask": mask}
        target = create_voi(data=target_dict)
        assert isinstance(target, Target)
        assert target.grid is grid

    def test_dict_input_sitk_as_grid(self, generic_input_3d):
        """Test dictionary input with SimpleITK image as grid."""
        name, ct, mask, _, _ = generic_input_3d
        target_dict = {"voi_type": "TARGET", "name": name, "grid": ct.cube_hu, "mask": mask}
        target = create_voi(data=target_dict)
        assert isinstance(target, Target)
        assert hasattr(target, "grid")

    def test_dict_input_sitk_as_image(self, generic_input_3d):
        """Test dictionary input with SimpleITK image as image."""
        name, ct, mask, _, _ = generic_input_3d
        target_dict = {"voi_type": "TARGET", "name": name, "image": ct.cube_hu, "mask": mask}
        target = create_voi(data=target_dict)
        assert isinstance(target, Target)
        assert hasattr(target, "grid")

    def test_dict_input_multiple_sources(self, generic_input_3d):
        """Test dictionary input with multiple grid sources."""
        name, ct, mask, _, _ = generic_input_3d
        from pyRadPlan.core import Grid

        grid = Grid.from_sitk_image(ct.cube_hu)

        target_dict = {
            "voi_type": "TARGET",
            "name": name,
            "grid": grid,
            "ct_image": ct,
            "mask": mask,
        }

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            target = create_voi(data=target_dict)
            assert len(w) == 1
            assert "Multiple grid sources provided" in str(w[0].message)

        assert isinstance(target, Target)
        assert target.grid is grid

    def test_sitk_mask_key_ignored(self, generic_input_3d):
        """Test that SimpleITK images with 'mask' key are not treated as grid sources."""
        name, ct, mask, _, _ = generic_input_3d
        # This should work normally - mask should not be treated as a grid source
        target = Target(name=name, ct_image=ct, mask=mask)
        assert hasattr(target, "grid")
        assert isinstance(target.mask, sitk.Image)

    def test_non_dict_input_passthrough(self):
        """Test that non-dictionary inputs are passed through unchanged."""
        from pyRadPlan.cst._voi import VOI

        result = VOI.validate_inputs("not_a_dict")
        assert result == "not_a_dict"

    def test_all_voi_types_with_different_sources(self, generic_input_3d):
        """Test all VOI types work with different grid source types."""
        name, ct, mask, _, _ = generic_input_3d
        from pyRadPlan.core import Grid

        # Test each VOI type with different source types
        voi_types = ["TARGET", "OAR", "HELPER", "EXTERNAL"]
        source_configs = [
            {"ct_image": ct},
            {"grid": Grid.from_sitk_image(ct.cube_hu)},
            {"image": ct.cube_hu},
            {"ct_image": ct.cube_hu},  # SimpleITK as ct_image
        ]

        for voi_type in voi_types:
            for source_config in source_configs:
                voi_data = {"voi_type": voi_type, "name": name, "mask": mask, **source_config}
                voi = create_voi(data=voi_data)
                assert voi.voi_type == voi_type
                assert hasattr(voi, "grid")


class TestGridSourceEdgeCases:
    """Test edge cases and error conditions for grid sources."""

    def test_no_grid_source_uses_default(self, generic_input_3d):
        """Test that VOI works with default Grid when no grid source is provided."""
        name, _, mask, _, _ = generic_input_3d
        # Create a basic mask that matches default grid dimensions
        simple_mask = np.zeros((1, 1, 1), dtype=np.uint8)
        simple_mask[0, 0, 0] = 1

        target = Target(name=name, mask=simple_mask)
        assert hasattr(target, "grid")
        # Should use the default Grid factory

    def test_invalid_sitk_key_ignored(self, generic_input_3d):
        """Test that SimpleITK images with invalid keys are ignored."""
        name, ct, mask, _, _ = generic_input_3d

        # 'invalid_key' should not be recognized as a grid source
        target_dict = {
            "voi_type": "TARGET",
            "name": name,
            "ct_image": ct,
            "invalid_key": ct.cube_hu,
            "mask": mask,
        }
        target = create_voi(data=target_dict)
        assert isinstance(target, Target)
        # Should use CT, not the invalid_key SimpleITK image


class TestCreateGridFromMask:
    """Test cases for the _create_grid_from_mask() static method."""

    def test_create_grid_from_numpy_3d_mask(self):
        """Test creating Grid from 3D numpy array mask."""
        from pyRadPlan.cst._voi import VOI
        from pyRadPlan.core import Grid

        # Mask in (z, y, x) format
        mask_3d = np.ones((10, 20, 30), dtype=np.uint8)

        grid = VOI._create_grid_from_mask(mask_3d)

        assert isinstance(grid, Grid)
        assert grid.dimensions == (30, 20, 10)  # Converted from (z,y,x) to (x,y,z)
        assert grid.resolution == {"x": 1.0, "y": 1.0, "z": 1.0}

        np.testing.assert_array_equal(grid.origin, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(grid.direction, np.eye(3))

    def test_create_grid_from_numpy_4d_mask(self):
        """Test creating Grid from 4D numpy array mask."""
        from pyRadPlan.cst._voi import VOI
        from pyRadPlan.core import Grid

        # Create a 4D numpy mask in (t, z, y, x) format
        mask_4d = np.ones((2, 10, 20, 30), dtype=np.uint8)

        grid = VOI._create_grid_from_mask(mask_4d)

        assert isinstance(grid, Grid)
        assert grid.dimensions == (30, 20, 10, 2)  # converted from (t,z,y,x) to (x,y,z,t)
        assert grid.resolution == {"x": 1.0, "y": 1.0, "z": 1.0, "t": 1.0}

        np.testing.assert_array_equal(grid.origin, np.array([0.0, 0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(grid.direction, np.eye(4))

    def test_create_grid_from_sitk_3d_mask(self):
        """Test creating Grid from 3D SimpleITK image mask."""
        from pyRadPlan.cst._voi import VOI
        from pyRadPlan.core import Grid

        # SimpleITK mask (already in x, y, z format)
        mask_array = np.ones((10, 20, 30), dtype=np.uint8)  # (z, y, x) for numpy
        sitk_mask = sitk.GetImageFromArray(mask_array)  # Will be (x, y, z) in SimpleITK
        sitk_mask.SetSpacing((1.5, 2.0, 2.5))
        sitk_mask.SetOrigin((10.0, 20.0, 30.0))

        grid = VOI._create_grid_from_mask(sitk_mask)

        assert isinstance(grid, Grid)
        assert grid.dimensions == sitk_mask.GetSize()

        expected_resolution = {"x": 1.5, "y": 2.0, "z": 2.5}
        assert grid.resolution == expected_resolution

        np.testing.assert_array_equal(grid.origin, np.array([10.0, 20.0, 30.0]))

    def test_create_grid_from_sitk_4d_mask(self):
        """Test creating Grid from 4D SimpleITK image mask."""
        from pyRadPlan.cst._voi import VOI
        from pyRadPlan.core import Grid

        # 4D SimpleITK mask
        mask_3d = np.ones((10, 20, 30), dtype=np.uint8)
        sitk_mask_3d = sitk.GetImageFromArray(mask_3d)
        sitk_mask_4d = sitk.JoinSeries([sitk_mask_3d, sitk_mask_3d])
        sitk_mask_4d.SetSpacing((1.5, 2.0, 2.5, 1.0))
        sitk_mask_4d.SetOrigin((10.0, 20.0, 30.0, 0.0))

        grid = VOI._create_grid_from_mask(sitk_mask_4d)

        assert isinstance(grid, Grid)

        # Match SimpleITK image (x, y, z, t)
        assert grid.dimensions == sitk_mask_4d.GetSize()

        expected_resolution = {"x": 1.5, "y": 2.0, "z": 2.5, "t": 1.0}
        assert grid.resolution == expected_resolution

        np.testing.assert_array_equal(grid.origin, np.array([10.0, 20.0, 30.0, 0.0]))

    def test_create_grid_from_mask_invalid_type(self):
        """Test error handling for invalid mask types."""
        from pyRadPlan.cst._voi import VOI

        # Invalid mask
        invalid_mask = "not_a_mask"

        with pytest.raises(ValueError, match="Unsupported mask type"):
            VOI._create_grid_from_mask(invalid_mask)

    def test_create_grid_from_numpy_2d_mask_error(self):
        """Test error handling for unsupported 2D numpy arrays."""
        from pyRadPlan.cst._voi import VOI

        # 2D numpy mask (should fail)
        mask_2d = np.ones((20, 30), dtype=np.uint8)

        with pytest.raises(ValueError, match="Unsupported array dimensionality"):
            VOI._create_grid_from_mask(mask_2d)

    def test_create_grid_from_numpy_5d_mask_error(self):
        """Test error handling for unsupported 5D numpy arrays."""
        from pyRadPlan.cst._voi import VOI

        # 5D numpy mask (should fail)
        mask_5d = np.ones((2, 3, 10, 20, 30), dtype=np.uint8)

        with pytest.raises(ValueError, match="Unsupported array dimensionality"):
            VOI._create_grid_from_mask(mask_5d)

    def test_create_grid_from_different_numpy_dtypes(self):
        """Test creating Grid from numpy arrays with different dtypes."""
        from pyRadPlan.cst._voi import VOI
        from pyRadPlan.core import Grid

        # Test with different numpy dtypes
        dtypes = [np.uint8, np.int32, np.float32, np.bool_]

        for dtype in dtypes:
            mask_3d = np.ones((5, 10, 15), dtype=dtype)
            grid = VOI._create_grid_from_mask(mask_3d)

            assert isinstance(grid, Grid)
            assert grid.dimensions == (15, 10, 5)  # (z,y,x) -> (x,y,z)
            assert grid.resolution == {"x": 1.0, "y": 1.0, "z": 1.0}

    def test_create_grid_preserves_sitk_direction_matrix(self):
        """Test that SimpleITK direction matrix is preserved."""
        from pyRadPlan.cst._voi import VOI
        from pyRadPlan.core import Grid

        # SimpleITK mask with custom direction matrix
        mask_array = np.ones((5, 10, 15), dtype=np.uint8)
        sitk_mask = sitk.GetImageFromArray(mask_array)

        # Custom direction matrix (rotation)
        custom_direction = (0, 1, 0, -1, 0, 0, 0, 0, 1)  # 90° rotation around z
        sitk_mask.SetDirection(custom_direction)

        grid = VOI._create_grid_from_mask(sitk_mask)

        # Verify direction matrix is preserved
        expected_direction = np.array(custom_direction).reshape(3, 3)
        np.testing.assert_array_equal(grid.direction, expected_direction)


def test_voi_geometry_computed_fields():
    from pyRadPlan.ct import create_ct

    img = sitk.Image(20, 20, 20, sitk.sitkInt16)
    img.SetSpacing((2.0, 2.0, 2.0))
    img.SetOrigin((-20.0, -20.0, -20.0))
    ct = create_ct(cube_hu=img)

    # 2x2x10 voxel rod elongated along z (numpy order is (z, y, x))
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[5:15, 9:11, 9:11] = 1
    voi = create_voi(voi_type="TARGET", name="rod", ct_image=ct, mask=mask)

    # voxel centers: x/y in {-2, 0}, z in {-10, ..., 8} -> mean -1 on every axis
    assert voi.center_of_mass == pytest.approx((-1.0, -1.0, -1.0), abs=1e-6)

    axes = voi.principal_axes
    assert len(axes) == 3
    # first (dominant) axis points along the rod, i.e. z
    assert abs(axes[0][2]) == pytest.approx(1.0, abs=1e-6)
    for axis in axes:
        assert np.linalg.norm(axis) == pytest.approx(1.0, abs=1e-6)

    shape = voi.shape_parameters
    assert shape["volume"] == pytest.approx(2 * 2 * 10 * 8.0)
    assert shape["bounding_box_size"] == pytest.approx((4.0, 4.0, 20.0))
    diameters = shape["equivalent_ellipsoid_diameters"]
    assert diameters[0] >= diameters[1] >= diameters[2]
    assert shape["elongation"] >= 1.0
    assert shape["flatness"] >= 1.0


def test_voi_geometry_empty_mask():
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    voi = create_voi(voi_type="TARGET", name="empty", mask=mask)
    assert voi.center_of_mass is None
    assert voi.principal_axes is None
    assert voi.shape_parameters is None


def test_voi_geometry_4d_uses_nominal_scenario(generic_input_4d):
    name, ct, mask, _, _ = generic_input_4d
    voi = create_voi(voi_type="TARGET", name=name, ct_image=ct, mask=mask)
    assert voi.center_of_mass is not None
    assert len(voi.center_of_mass) == 3
    assert voi.shape_parameters is not None


def test_voi_geometry_in_model_dump(generic_input_3d):
    name, ct, mask, _, _ = generic_input_3d
    voi = create_voi(voi_type="TARGET", name=name, ct_image=ct, mask=mask)
    dumped = voi.model_dump()
    assert dumped["center_of_mass"] == voi.center_of_mass
    assert dumped["principal_axes"] == voi.principal_axes
    assert dumped["shape_parameters"] == voi.shape_parameters


@pytest.mark.parametrize(
    "color_in, color_expected",
    [
        ("red", (255, 0, 0)),
        ((0.5, 0.5, 0.5), (128, 128, 128)),
        ((10, 20, 30), (10, 20, 30)),
        (np.array([40, 50, 60]), (40, 50, 60)),
        (np.array([0.5, 0.25, 0.0]), (128, 64, 0)),
        (None, None),
    ],
)
def test_voi_visible_color_validator(generic_input_3d, color_in, color_expected):
    name, ct, mask, _, _ = generic_input_3d
    kwargs = {"voi_type": "TARGET", "name": name, "ct_image": ct, "mask": mask}
    voi = create_voi(**kwargs, visible_color=color_in)
    assert voi.visible_color == color_expected
