import numpy as np
import pytest
from pyRadPlan.machines import BeamLimitingDevice, MLC, Jaw, create_bld


def sample_MLC(device_type=None, orientation=None):
    leaf_width = 5
    positions = [
        [0, 5],
        [-10, 10],
        [-15, 10],
        [-18, 12],
        [-25, 20],
        [-20, 25],
        [-10, 28],
        [0, 30],
        [5, 15],
        [10, 13],
    ]
    number_of_elements = len(positions)
    boundaries = np.arange(
        -int(number_of_elements / 2) * leaf_width,
        int(number_of_elements / 2) * leaf_width,
        leaf_width,
    )

    mlc_information = {
        "device_type": device_type if device_type is not None else "MLC",
        "leaf_position_boundaries": boundaries,
        "leaf_positions": positions,
        "leaf_width": leaf_width,
    }
    if orientation is not None:
        mlc_information["device_orientation"] = orientation

    return mlc_information


def test_MLC_wrong_args():
    with pytest.raises(ValueError):
        MLC(device_type="MLC")


def test_jaw_wrong_args():
    with pytest.raises(ValueError):
        Jaw(device_type="JAW")
    # missing field_width or positions
    with pytest.raises(ValueError):
        Jaw(device_type="JAWX", positions=[-30, 40])
    with pytest.raises(ValueError):
        Jaw(device_type="JAWX", field_width=60)
    # wrong number of positions
    with pytest.raises(ValueError):
        Jaw(device_type="JAWX", positions=[-30, 40, 0], field_width=60)
    with pytest.raises(ValueError):
        Jaw(device_type="JAWX", positions=[-30], field_width=60)
    # field_width samller than maximum position
    with pytest.raises(ValueError):
        Jaw(device_type="JAWX", positions=[-30, 40], field_width=20)


def test_create_bld_no_args():
    with pytest.raises(ValueError):
        create_bld()


def test_create_bld_from_MLC():
    mlc_X1 = MLC(device_orientation="X")
    mlc_Y1 = MLC(device_orientation="Y")
    mlc_X2 = MLC(device_type="MLCX")
    mlc_Y2 = MLC(device_type="MLCY")
    assert mlc_X1.device_type == "MLCX"
    assert mlc_Y1.device_type == "MLCY"
    assert mlc_X2.device_orientation == "X"
    assert mlc_Y2.device_orientation == "Y"

    new_mlc_X1 = create_bld(mlc_X1)
    new_mlc_Y1 = create_bld(mlc_Y1)
    new_mlc_X2 = create_bld(mlc_X2)
    new_mlc_Y2 = create_bld(mlc_Y2)
    assert isinstance(new_mlc_X1, BeamLimitingDevice)
    assert isinstance(new_mlc_Y1, BeamLimitingDevice)
    assert isinstance(new_mlc_X2, BeamLimitingDevice)
    assert isinstance(new_mlc_Y2, BeamLimitingDevice)


def test_create_bld_from_jaw():
    positions = [-30, 40]
    field_width = 100
    jaw_X1 = Jaw(device_orientation="X", positions=positions, field_width=field_width)
    jaw_Y1 = Jaw(device_orientation="Y", positions=positions, field_width=field_width)
    jaw_X2 = Jaw(device_type="JAWX", positions=positions, field_width=field_width)
    jaw_Y2 = Jaw(device_type="JAWY", positions=positions, field_width=field_width)
    assert jaw_X1.device_type == "JAWX"
    assert jaw_Y1.device_type == "JAWY"
    assert jaw_X2.device_orientation == "X"
    assert jaw_Y2.device_orientation == "Y"

    new_jaw_X1 = create_bld(jaw_X1)
    new_jaw_Y1 = create_bld(jaw_Y1)
    new_jaw_X2 = create_bld(jaw_X2)
    new_jaw_Y2 = create_bld(jaw_Y2)
    assert isinstance(new_jaw_X1, BeamLimitingDevice)
    assert isinstance(new_jaw_Y1, BeamLimitingDevice)
    assert isinstance(new_jaw_X2, BeamLimitingDevice)
    assert isinstance(new_jaw_Y2, BeamLimitingDevice)


def test_create_bld_orientation():
    mlc_X1 = create_bld(device_type="MLC", device_orientation="X")
    mlc_Y1 = create_bld(device_type="MLC", device_orientation="Y")
    mlc_X2 = create_bld(device_type="MLCX")
    mlc_Y2 = create_bld(device_type="MLCY")
    assert mlc_X1.device_type == "MLCX"
    assert mlc_Y1.device_type == "MLCY"
    assert mlc_X2.device_orientation == "X"
    assert mlc_Y2.device_orientation == "Y"

    with pytest.raises(ValueError):
        create_bld(device_type="MLCX", device_orientation="Y")
    with pytest.raises(ValueError):
        create_bld(device_type="MLCY", device_orientation="X")

    positions = [-30, 40]
    field_width = 100
    jaw_X1 = create_bld(
        device_type="JAW", device_orientation="X", positions=positions, field_width=field_width
    )
    jaw_Y1 = create_bld(
        device_type="JAW", device_orientation="Y", positions=positions, field_width=field_width
    )
    jaw_X2 = create_bld(device_type="JAWX", positions=positions, field_width=field_width)
    jaw_Y2 = create_bld(device_type="JAWY", positions=positions, field_width=field_width)
    assert jaw_X1.device_type == "JAWX"
    assert jaw_Y1.device_type == "JAWY"
    assert jaw_X2.device_orientation == "X"
    assert jaw_Y2.device_orientation == "Y"

    with pytest.raises(ValueError):
        create_bld(device_type="JAWX", device_orientation="Y")
    with pytest.raises(ValueError):
        create_bld(device_type="JAWY", device_orientation="X")


def test_create_bld_lower_case():
    mlc_X1 = create_bld(device_type="mlc", device_orientation="x")
    mlc_Y1 = create_bld(device_type="mlc", device_orientation="y")
    mlc_X2 = create_bld(device_type="mlcx")
    mlc_Y2 = create_bld(device_type="mlcy")
    assert mlc_X1.device_type == "MLCX"
    assert mlc_Y1.device_type == "MLCY"
    assert mlc_X2.device_orientation == "X"
    assert mlc_Y2.device_orientation == "Y"

    positions = [-30, 40]
    field_width = 100
    jaw_X1 = create_bld(
        device_type="jaw", device_orientation="X", positions=positions, field_width=field_width
    )
    jaw_Y1 = create_bld(
        device_type="jaw", device_orientation="Y", positions=positions, field_width=field_width
    )
    jaw_X2 = create_bld(device_type="jawx", positions=positions, field_width=field_width)
    jaw_Y2 = create_bld(device_type="jawy", positions=positions, field_width=field_width)
    assert jaw_X1.device_type == "JAWX"
    assert jaw_Y1.device_type == "JAWY"
    assert jaw_X2.device_orientation == "X"
    assert jaw_Y2.device_orientation == "Y"


def test_create_bld_from_dict_MLC():
    # MLCX
    sample_mlc_X1 = sample_MLC("MLCX", "X")
    sample_mlc_X2 = sample_MLC("MLC", "X")
    sample_mlc_X3 = sample_MLC(device_type="MLCX")
    sample_mlc_X_boundaries_and_positions_mismatch = sample_MLC("MLCX", "X")
    sample_mlc_X_unsorted = sample_MLC("MLCX", "X")
    sample_mlc_X_no_leaf_width = sample_MLC("MLCX", "X")
    sample_mlc_X_wrong_leaf_width = sample_MLC("MLCX", "X")
    sample_mlc_X_boundaries_and_positions_mismatch["leaf_position_boundaries"] = [0, 1, 2]
    sample_mlc_X_unsorted["leaf_position_boundaries"] = sample_mlc_X_unsorted[
        "leaf_position_boundaries"
    ][::-1]
    sample_mlc_X_no_leaf_width.pop("leaf_width")
    sample_mlc_X_wrong_leaf_width["leaf_width"] = 1
    mlc_X1 = create_bld(sample_mlc_X1)
    mlc_X2 = create_bld(sample_mlc_X2)
    mlc_X3 = create_bld(sample_mlc_X3)
    mlc_X_no_leaf_width = create_bld(sample_mlc_X_no_leaf_width)
    assert isinstance(mlc_X1, MLC)
    assert isinstance(mlc_X2, MLC)
    assert isinstance(mlc_X3, MLC)
    assert mlc_X1.device_type == "MLCX"
    assert mlc_X2.device_type == "MLCX"
    assert mlc_X3.device_type == "MLCX"
    assert mlc_X1.device_orientation == "X"
    assert mlc_X2.device_orientation == "X"
    assert mlc_X3.device_orientation == "X"
    assert mlc_X_no_leaf_width.leaf_width == 5
    with pytest.raises(ValueError):
        create_bld(sample_mlc_X_boundaries_and_positions_mismatch)
    with pytest.raises(ValueError):
        create_bld(sample_mlc_X_unsorted)
    with pytest.raises(ValueError):
        create_bld(sample_mlc_X_wrong_leaf_width)

    # MLCY
    sample_mlc_Y1 = sample_MLC("MLCY", "Y")
    sample_mlc_Y2 = sample_MLC("MLC", "Y")
    sample_mlc_Y3 = sample_MLC("MLCY")
    sample_mlc_Y_boundaries_and_positions_mismatch = sample_MLC("MLCY", "Y")
    sample_mlc_Y_unsorted = sample_MLC("MLCY", "Y")
    sample_mlc_Y_no_leaf_width = sample_MLC("MLCY", "Y")
    sample_mlc_Y_wrong_leaf_width = sample_MLC("MLCY", "Y")
    sample_mlc_Y_boundaries_and_positions_mismatch["leaf_position_boundaries"] = [0, 1, 2]
    sample_mlc_Y_unsorted["leaf_position_boundaries"] = sample_mlc_Y_unsorted[
        "leaf_position_boundaries"
    ][::-1]
    sample_mlc_Y_no_leaf_width.pop("leaf_width")
    sample_mlc_Y_wrong_leaf_width["leaf_width"] = 1
    mlc_Y1 = create_bld(sample_mlc_Y1)
    mlc_Y2 = create_bld(sample_mlc_Y2)
    mlc_Y3 = create_bld(sample_mlc_Y3)
    mlc_Y_no_leaf_width = create_bld(sample_mlc_Y_no_leaf_width)
    assert isinstance(mlc_Y1, MLC)
    assert isinstance(mlc_Y2, MLC)
    assert isinstance(mlc_Y3, MLC)
    assert mlc_Y1.device_type == "MLCY"
    assert mlc_Y2.device_type == "MLCY"
    assert mlc_Y3.device_type == "MLCY"
    assert mlc_Y1.device_orientation == "Y"
    assert mlc_Y2.device_orientation == "Y"
    assert mlc_Y3.device_orientation == "Y"
    assert mlc_Y_no_leaf_width.leaf_width == 5
    with pytest.raises(ValueError):
        create_bld(sample_mlc_Y_boundaries_and_positions_mismatch)
    with pytest.raises(ValueError):
        create_bld(sample_mlc_Y_unsorted)
    with pytest.raises(ValueError):
        create_bld(sample_mlc_Y_wrong_leaf_width)


def test_create_bld_from_dict_jaw():
    jaw_info = {
        "device_type": "JAW",
        "device_orientation": "X",
        "positions": [-30, 40],
        "field_width": 100,
    }
    jaw_X = create_bld(jaw_info)
    assert isinstance(jaw_X, Jaw)
    assert jaw_X.device_type == "JAWX"
    assert jaw_X.device_orientation == "X"

    jaw_info["device_orientation"] = "Y"
    jaw_Y = create_bld(jaw_info)
    assert isinstance(jaw_Y, Jaw)
    assert jaw_Y.device_type == "JAWY"
    assert jaw_Y.device_orientation == "Y"


def test_create_MLC_transmission_mask():
    resolution = 0.5
    # MLCX
    sample_mlc_X = sample_MLC("MLCX", "X")
    mlc_X = create_bld(sample_mlc_X)
    mask_X = mlc_X.calculate_transmission_mask(resolution)
    open_leaf_positions_X = 0
    for p in mlc_X.leaf_positions:
        open_width = (p[1] - p[0]) / resolution  # in pixels
        leaf_width = mlc_X.leaf_width / resolution  # in pixels
        open_leaf_positions_X += open_width * leaf_width
    assert np.isclose(mask_X.sum(), open_leaf_positions_X, rtol=1e-3)

    # MLCY
    sample_mlc_Y = sample_MLC("MLCY", "Y")
    mlc_Y = create_bld(sample_mlc_Y)
    mask_Y = mlc_Y.calculate_transmission_mask(resolution)
    open_leaf_positions_Y = 0
    for p in mlc_Y.leaf_positions:
        open_width = (p[1] - p[0]) / resolution  # in pixels
        leaf_width = mlc_Y.leaf_width / resolution  # in pixels
        open_leaf_positions_Y += open_width * leaf_width
    assert np.isclose(mask_Y.sum(), open_leaf_positions_Y, rtol=1e-3)


def test_create_Jaw_transmission_mask():
    resolution = 0.5
    jaw_info = {
        "device_type": "JAW",
        "device_orientation": "X",
        "positions": [-30, 40],
        "field_width": 100,
    }
    jaw_X = create_bld(jaw_info)
    mask_X = jaw_X.calculate_transmission_mask(resolution)

    jaw_info["device_orientation"] = "Y"
    jaw_Y = create_bld(jaw_info)
    mask_Y = jaw_Y.calculate_transmission_mask(resolution)

    np.testing.assert_array_equal(mask_X, np.rot90(mask_Y))
