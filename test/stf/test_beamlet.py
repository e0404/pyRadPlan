from pyRadPlan.stf import IonSpot, PhotonBixel
from pyRadPlan.stf import RangeShifter


def test_proton_beamlet_only_energy():
    proton_beamlet = IonSpot(energy=100)
    assert proton_beamlet.energy == 100
    assert proton_beamlet.num_particles_per_mu == 1.0e6
    assert proton_beamlet.min_mu == 0.0
    assert proton_beamlet.max_mu == float("inf")
    assert isinstance(proton_beamlet.range_shifter, RangeShifter)
    assert proton_beamlet.focus_ix == 0


def test_proton_beamlet_all():
    proton_beamlet = IonSpot(
        energy=100,
        num_particles_per_mu=1.0e7,
        min_mu=1.0,
        max_mu=10.0,
        range_shifter=RangeShifter(),
        focus_ix=1,
    )
    assert proton_beamlet.energy == 100
    assert proton_beamlet.num_particles_per_mu == 1.0e7
    assert proton_beamlet.min_mu == 1.0
    assert proton_beamlet.max_mu == 10.0
    assert isinstance(proton_beamlet.range_shifter, RangeShifter)
    assert proton_beamlet.focus_ix == 1


def test_photon_beamlet_only_energy():
    photon_beamlet = PhotonBixel(energy=6)
    assert photon_beamlet.energy == 6
    assert photon_beamlet.num_particles_per_mu == 1.0e6
    assert photon_beamlet.min_mu == 0.0
    assert photon_beamlet.max_mu == float("inf")
    assert photon_beamlet.relative_fluence == 1.0


def test_photon_beamlet_all():
    photon_beamlet = PhotonBixel(
        energy=6,
        num_particles_per_mu=1.0e7,
        min_mu=1.0,
        max_mu=10.0,
        relative_fluence=0.5,
    )
    assert photon_beamlet.energy == 6
    assert photon_beamlet.num_particles_per_mu == 1.0e7
    assert photon_beamlet.min_mu == 1.0
    assert photon_beamlet.max_mu == 10.0
    assert photon_beamlet.relative_fluence == 0.5


########################## FieldShapes #############################
import numpy as np
import pytest
from pyRadPlan.machines import create_bld
from pyRadPlan.stf import FieldShape, FieldShapeAsMask, FieldShapeAsBLD, FieldShapeComposite


def sample_MLC(orientation):
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
        "device_type": "MLC",
        "device_orientation": orientation,
        "leaf_position_boundaries": boundaries,
        "leaf_positions": positions,
        "leaf_width": leaf_width,
    }
    mlc = create_bld(mlc_information)
    return mlc


def make_grid(resolution, field_width):
    n = int(np.ceil(field_width / resolution))
    if n % 2 == 0:
        n += 1
    half_size = n // 2
    return resolution * np.arange(-half_size, half_size + 1)


def assert_spatial_consistencies(field_shape, resolution, field_width):
    assert np.isclose(field_shape.resolution, resolution)
    assert np.isclose(field_shape.field_width, field_width)
    assert np.isclose(np.unique(np.diff(field_shape.grid))[0], resolution)
    assert np.isclose((len(field_shape.grid) - 1) * resolution, field_width)


def test_shape():
    energy = 6
    resolution = 0.5
    mlc_X = sample_MLC("X")
    mask_mlc_X = np.fliplr(
        mlc_X.calculate_transmission_mask(resolution)
    )  # IEC X --> BEV-LPS X (flip)
    half_size = mask_mlc_X.shape[0] // 2
    field_grid = resolution * np.arange(-half_size, half_size + 1)

    # missing resolution or grid
    with pytest.raises(ValueError):
        FieldShapeAsMask(energy=energy, mask=mask_mlc_X)
    with pytest.raises(ValueError):
        FieldShapeAsBLD(energy=energy, bld=mlc_X)

    # create grid automatically from resolution or grid
    shape_mask_resolution = FieldShapeAsMask(energy=energy, mask=mask_mlc_X, resolution=resolution)
    shape_mask_grid = FieldShapeAsMask(energy=energy, mask=mask_mlc_X, grid=field_grid)
    shape_bld_resolution = FieldShapeAsBLD(energy=energy, bld=mlc_X, resolution=resolution)
    shape_bld_grid = FieldShapeAsBLD(energy=energy, bld=mlc_X, grid=field_grid)
    assert isinstance(shape_mask_resolution, FieldShape)
    assert isinstance(shape_mask_grid, FieldShape)
    assert isinstance(shape_bld_resolution, FieldShape)
    assert isinstance(shape_bld_grid, FieldShape)
    np.testing.assert_array_equal(shape_mask_resolution.grid, field_grid)
    assert shape_mask_grid.resolution == resolution
    np.testing.assert_array_equal(shape_bld_resolution.grid, field_grid)
    assert shape_bld_grid.resolution == resolution
    np.testing.assert_array_equal(shape_mask_resolution.grid, shape_bld_resolution.grid)
    np.testing.assert_array_equal(shape_mask_resolution.mask, shape_bld_resolution.mask)

    # TODO: test field_width


def test_shape_resampling():
    energy = 6
    resolution = 0.5
    resolution_resampled = 1.2
    padding = 10.0
    mlc_X = sample_MLC("X")
    mask_mlc_X = np.fliplr(mlc_X.calculate_transmission_mask(resolution))

    field_shape_mask = FieldShapeAsMask(energy=energy, mask=mask_mlc_X, resolution=resolution)
    field_shape_bld = FieldShapeAsBLD(energy=energy, bld=mlc_X, resolution=resolution)

    field_width = field_shape_mask.field_width
    field_width_padded = field_width + padding

    grid = make_grid(resolution, field_width)
    grid_resampled = make_grid(resolution_resampled, field_width)
    grid_padded = make_grid(resolution, field_width_padded)
    grid_resampled_padded = make_grid(resolution_resampled, field_width_padded)

    field_width_resampled_padded = (len(grid_resampled_padded) - 1) * resolution_resampled

    # RESAMPLING
    resampled_mask_new_resolution = field_shape_mask.resample(new_resolution=resolution_resampled)
    resampled_bld_new_resolution = field_shape_bld.resample(new_resolution=resolution_resampled)
    resampled_mask_new_grid = field_shape_mask.resample(new_grid=grid_resampled)
    resampled_bld_new_grid = field_shape_bld.resample(new_grid=grid_resampled)
    assert_spatial_consistencies(resampled_mask_new_resolution, resolution_resampled, field_width)
    assert_spatial_consistencies(resampled_bld_new_resolution, resolution_resampled, field_width)
    assert_spatial_consistencies(resampled_mask_new_grid, resolution_resampled, field_width)
    assert_spatial_consistencies(resampled_bld_new_grid, resolution_resampled, field_width)

    # PADDING
    padded_mask_new_field_width = field_shape_mask.resample(new_field_width=field_width_padded)
    padded_bld_new_field_width = field_shape_bld.resample(new_field_width=field_width_padded)
    padded_mask_new_grid = field_shape_mask.resample(new_grid=grid_padded)
    padded_bld_new_grid = field_shape_bld.resample(new_grid=grid_padded)
    assert_spatial_consistencies(padded_mask_new_field_width, resolution, field_width_padded)
    assert_spatial_consistencies(padded_bld_new_field_width, resolution, field_width_padded)
    assert_spatial_consistencies(padded_mask_new_grid, resolution, field_width_padded)
    assert_spatial_consistencies(padded_bld_new_grid, resolution, field_width_padded)

    # RESAMPLING & PADDING
    resampled_padded_mask_new_resolution = field_shape_mask.resample(
        new_resolution=resolution_resampled, new_field_width=field_width_padded
    )
    resampled_padded_bld_new_resolution = field_shape_bld.resample(
        new_resolution=resolution_resampled, new_field_width=field_width_padded
    )
    resampled_padded_mask_new_grid = field_shape_mask.resample(new_grid=grid_resampled_padded)
    resampled_padded_bld_new_grid = field_shape_bld.resample(new_grid=grid_resampled_padded)
    assert_spatial_consistencies(
        resampled_padded_mask_new_resolution, resolution_resampled, field_width_resampled_padded
    )
    assert_spatial_consistencies(
        resampled_padded_bld_new_resolution, resolution_resampled, field_width_resampled_padded
    )
    assert_spatial_consistencies(
        resampled_padded_mask_new_grid, resolution_resampled, field_width_resampled_padded
    )
    assert_spatial_consistencies(
        resampled_padded_bld_new_grid, resolution_resampled, field_width_resampled_padded
    )

    with pytest.raises(ValueError):
        field_shape_mask.resample()
    with pytest.raises(ValueError):
        field_shape_bld.resample()
    with pytest.raises(ValueError):
        field_shape_mask.resample(new_resolution=resolution_resampled, new_grid=grid)
    with pytest.raises(ValueError):
        field_shape_bld.resample(new_resolution=resolution_resampled, new_grid=grid)
    with pytest.raises(ValueError):
        field_shape_mask.resample(new_resolution=resolution, new_grid=grid_resampled)
    with pytest.raises(ValueError):
        field_shape_bld.resample(new_resolution=resolution, new_grid=grid_resampled)
    with pytest.raises(ValueError):
        field_shape_mask.resample(new_grid=grid_padded, new_field_width=field_width)
    with pytest.raises(ValueError):
        field_shape_bld.resample(new_grid=grid_padded, new_field_width=field_width)
    with pytest.raises(ValueError):
        field_shape_mask.resample(new_grid=grid, new_field_width=field_width_padded)
    with pytest.raises(ValueError):
        field_shape_bld.resample(new_grid=grid, new_field_width=field_width_padded)

    # TODO: test created masks!


def test_shape_composite():
    energy = 6
    resolution = 0.5
    mlc_X = sample_MLC("X")
    mlc_Y = sample_MLC("Y")

    shape_x = FieldShapeAsBLD(energy=energy, bld=mlc_X, resolution=resolution)
    shape_y = FieldShapeAsBLD(energy=energy, bld=mlc_Y, resolution=resolution)

    # composite derives spatial params from children
    composite = FieldShapeComposite(energy=energy, shapes=[shape_x, shape_y])
    assert isinstance(composite, FieldShape)
    assert np.isclose(composite.resolution, resolution)
    assert np.isclose(composite.field_width, max(shape_x.field_width, shape_y.field_width))

    # mask is element-wise product of children resampled to composite grid
    expected = (
        shape_x.resample(new_grid=composite.grid).mask
        * shape_y.resample(new_grid=composite.grid).mask
    )
    np.testing.assert_array_almost_equal(composite.mask, expected)

    # single-shape composite behaves like the child
    composite_single = FieldShapeComposite(energy=energy, shapes=[shape_x])
    np.testing.assert_array_almost_equal(composite_single.mask, shape_x.mask)

    # resampling preserves composite type and adjusts spatial params
    resampled = composite.resample(new_resolution=1.0)
    assert isinstance(resampled, FieldShapeComposite)
    assert_spatial_consistencies(resampled, 1.0, composite.field_width)

    # explicit grid overrides child-derived params
    composite_with_grid = FieldShapeComposite(
        energy=energy, shapes=[shape_x, shape_y], grid=composite.grid
    )
    np.testing.assert_array_equal(composite_with_grid.grid, composite.grid)


# def test_shape_spatial_inconsistancies():

# def test_shape_from_irregular_mask():
#     # TODO:
#     # - non-square mask
#     # - test padding for even mask widths
