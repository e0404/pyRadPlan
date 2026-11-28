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
    # field_width smaller than maximum position
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


def _mlc_with_rows(
    j_start: int,
    j_end: int,
    leaf_width: float = 10.0,
    number_of_elements: int = 40,
    orientation: str = "X",
) -> MLC:
    """An MLC whose shape spans global (isocenter-relative) leaf rows j_start..j_end."""
    leaf_row_pos = np.arange(j_start, j_end + 1) * leaf_width
    n = len(leaf_row_pos)
    return MLC(
        device_type=f"MLC{orientation}",
        number_of_elements=number_of_elements,
        leaf_width=leaf_width,
        leaf_position_boundaries=leaf_row_pos,
        leaf_positions=np.tile([-5.0, 5.0], (n, 1)),
    )


def test_full_leaf_positions():
    """A field's leaf rows must fall within the fixed window implied by
    central_leaf_pair to be placeable -- same row count alone isn't enough.
    """
    # Exactly fills the device's [-central, central-1] window: every leaf
    # pair placed, none left over.
    mlc_1 = _mlc_with_rows(-20, 19)
    # Same row count as mlc_1, but shifted by one leaf width off that window
    # -- cannot be placed on every one of its own rows.
    mlc_2 = _mlc_with_rows(-19, 20)
    # Entirely to one side of isocenter (e.g. a beam whose native field only
    # needs positive-z leaf rows) -- previously produced an all-inactive mask
    # (silent, wrong) instead of an error.
    mlc_3 = _mlc_with_rows(1, 20)

    # Boundaries at multiples of the leaf width: a leaf boundary on isocenter,
    # between pairs 20 and 21 (1-based).
    central_leaf_pair = 20.5

    full = mlc_1.full_leaf_positions(central_leaf_pair)
    assert full.shape == (40, 2)
    np.testing.assert_array_equal(full, mlc_1.leaf_positions)

    with pytest.raises(ValueError, match="does not fit within the physical collimator"):
        mlc_2.full_leaf_positions(central_leaf_pair)

    with pytest.raises(ValueError, match="does not fit within the physical collimator"):
        mlc_3.full_leaf_positions(central_leaf_pair)


@pytest.mark.parametrize("spacing", [0.5, 0.7, 1.0])
def test_transmission_mask_grid_has_exact_spacing(spacing):
    """FieldShapeAsBLD labels the mask with ``spacing * arange(-n//2, n//2 + 1)``, so
    the mask must be sampled at exactly that pitch, or it is silently rescaled.
    """
    mlc = MLC(
        device_type="MLCX",
        leaf_width=10.0,
        leaf_position_boundaries=np.array([-20.0, -10.0, 0.0, 10.0]),
        leaf_positions=np.tile([-13.3, 8.1], (4, 1)),
    )
    mask = mlc.calculate_transmission_mask(spacing)

    # A one-cell-wide edge ramp sampled at the cell pitch sums to the exact open
    # length, so any mismatch between sampling and labelled pitch shows up here.
    row = mask[mask.shape[0] // 2]
    np.testing.assert_allclose(row.sum() * spacing, 8.1 + 13.3, atol=1e-5)


@pytest.mark.parametrize("orientation", ["X", "Y"])
@pytest.mark.parametrize(
    ("j_start", "j_end"),
    [
        pytest.param(-3, -1, id="below_isocenter"),
        pytest.param(-2, 1, id="straddling_isocenter"),
        pytest.param(1, 3, id="above_isocenter"),
    ],
)
def test_transmission_mask_grid_covers_every_leaf_row(j_start, j_end, orientation):
    """The mask grid must hold the whole device, wherever the field sits.

    ``leaf_position_boundaries`` are each row's *lower* edge, so a row reaches one
    leaf width above its boundary. The transmission mask grid must span from the
    lowest row's lower edge to the highest row's upper edge.
    """
    resolution = 0.5
    open_width = 10.0  # _mlc_with_rows opens every row from -5 to +5 mm
    mlc = _mlc_with_rows(j_start, j_end, orientation=orientation)
    boundaries = np.asarray(mlc.leaf_position_boundaries)
    n_rows = len(boundaries)

    mask = mlc.calculate_transmission_mask(resolution)

    travel_axis = 1 if orientation == "X" else 0
    covered_rows = int((mask.sum(axis=travel_axis) > 0).sum())
    # One cell of slack: the edge ramp (see MLC.calculate_transmission_mask())
    # reaches half a cell past each outer edge.
    assert abs(covered_rows - n_rows * mlc.leaf_width / resolution) <= 1, (
        f"rows span [{boundaries.min():g}, {(boundaries + mlc.leaf_width).max():g}] mm, "
        f"but only {covered_rows * resolution:g} of {n_rows * mlc.leaf_width:g} mm "
        "of leaf rows appear in the mask"
    )

    # leaf_leakage is 0 and the edge ramps (see MLC.calculate_transmission_mask())
    # integrate exactly, so the mask sums to the open area in pixels.
    expected_area = n_rows * (mlc.leaf_width / resolution) * (open_width / resolution)
    assert mask.sum() == pytest.approx(expected_area, rel=1e-3)


def test_full_leaf_positions_follow_the_device_leaf_order():
    """The device counts its leaf pairs towards negative stack coordinates."""
    openings = np.array([[-5.0, 1.0], [-4.0, 2.0], [-3.0, 3.0]])
    mlc = MLC(
        device_type="MLCX",
        number_of_elements=6,
        leaf_width=10.0,
        leaf_position_boundaries=np.array([-10.0, 0.0, 10.0]),
        leaf_positions=openings,
    )

    full = mlc.full_leaf_positions(3.5)

    # Rows at -10, 0 and 10 mm are the device's leaf pairs 4, 3 and 2 (1-based)
    np.testing.assert_array_equal(np.flatnonzero(mlc.is_active_leaf_pair(3.5)), [1, 2, 3])
    np.testing.assert_array_equal(full[[3, 2, 1]], openings)
    np.testing.assert_array_equal(full[[0, 4, 5]], np.zeros((3, 2)))


def test_central_leaf_pair_needs_leaf_rows():
    mlc = MLC(device_orientation="X")
    with pytest.raises(ValueError, match="leaf rows"):
        _ = mlc.central_leaf_pair


def test_leaf_rows_from_positions_with_rounding_errors_are_consistent():
    centers = -12.3 + np.arange(6) * 4.1  # 4.1 mm rows, spacings differ in the last bit
    assert len(np.unique(np.diff(centers))) > 1
    assert MLC.validate_leaf_width_consistency(centers, None) == pytest.approx(4.1)
    with pytest.raises(ValueError, match="inconsistent leaf widths"):
        MLC.validate_leaf_width_consistency(np.array([0.0, 5.0, 11.0]), None)
