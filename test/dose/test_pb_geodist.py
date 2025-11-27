import pytest
import array_api_strict as xp
from pyRadPlan.dose.engines._base_pencilbeam import PencilBeamEngineAbstract


class DummyEngine(PencilBeamEngineAbstract):
    def _compute_bixel(self, curr_ray, k):
        pass


@pytest.fixture
def engine():
    # Provide minimal init args, override abstract methods
    class TestEngine(DummyEngine):
        def __init__(self):
            # Set required attributes for calc_geo_dists
            self.dose_grid = {"resolution": [5.0, 5.0, 5.0]}
            self.mult_scen = "nominal"
            super().__init__()

    return TestEngine()


def test_calc_geo_dists_identity_rotation(engine):
    # a == b, so no rotation
    sad = 1000.0
    rot_coords_bev = xp.asarray(
        [[1.0, sad + 2.0, 3.0], [4.0, sad + 5.0, 6.0], [7.0, sad + 8.0, 9.0]]
    )
    source_point_bev = xp.asarray([0.0, -sad, 0.0])
    target_point_bev = xp.asarray([0.0, sad, 0.0])
    rad_depth_ix = xp.asarray([False, True, True])
    lateral_cutoff = 50.0

    ix, rad_distances_sq, lat_dists, iso_lat_dists = engine.calc_geo_dists(
        rot_coords_bev, source_point_bev, target_point_bev, sad, rad_depth_ix, lateral_cutoff
    )

    # Should return indices, squared distances, etc.
    assert ix.shape == rad_depth_ix.shape
    assert xp.all(ix == rad_depth_ix)
    assert rad_distances_sq.shape[0] == lat_dists.shape[0] == iso_lat_dists.shape[0]
    assert lat_dists.shape[1] == 2
    assert iso_lat_dists.shape[1] == 2


def test_calc_geo_dists_with_rotation(engine):
    # a != b, so rotation is applied
    sad = 1000.0
    rot_coords_bev = xp.asarray(
        [[1.0, sad + 2.0, 3.0], [4.0, sad + 5.0, 6.0], [7.0, sad + 8.0, 9.0]]
    )
    source_point_bev = xp.asarray([0.0, -sad, 0.0])
    # Rotate target point slightly
    target_point_bev = xp.asarray([1.0, sad + 1.0, 0.5])
    rad_depth_ix = xp.asarray([False, True, True])
    lateral_cutoff = 50.0

    ix, rad_distances_sq, lat_dists, iso_lat_dists = engine.calc_geo_dists(
        rot_coords_bev, source_point_bev, target_point_bev, sad, rad_depth_ix, lateral_cutoff
    )

    assert ix.shape == rad_depth_ix.shape
    assert rad_distances_sq.shape[0] == lat_dists.shape[0] == iso_lat_dists.shape[0]
    assert lat_dists.shape[1] == 2
    assert iso_lat_dists.shape[1] == 2


def test_calc_geo_dists_lateral_cutoff(engine):
    # Test that lateral cutoff excludes points far from the ray
    sad = 1000.0
    rot_coords_bev = xp.asarray(
        [[1000.0, sad + 1000.0, 1000.0], [4.0, sad + 5.0, 6.0], [7.0, sad + 8.0, 9.0]]
    )
    source_point_bev = xp.asarray([0.0, -sad, 0.0])
    target_point_bev = xp.asarray([0.0, sad, 0.0])
    rad_depth_ix = xp.asarray([True, True, True])
    lateral_cutoff = 1.0  # Very small cutoff

    ix, rad_distances_sq, lat_dists, iso_lat_dists = engine.calc_geo_dists(
        rot_coords_bev, source_point_bev, target_point_bev, sad, rad_depth_ix, lateral_cutoff
    )

    # Only points close to the ray should be within cutoff
    assert xp.sum(xp.astype(ix, xp.int64)) <= 3
    assert lat_dists.shape == (xp.sum(xp.astype(ix, xp.int64)), 2)
    assert iso_lat_dists.shape == (xp.sum(xp.astype(ix, xp.int64)), 2)


def test_calc_geo_dists_types(engine):
    # Test with float32 input
    sad = 1000.0
    rot_coords_bev = xp.asarray([[1.0, sad + 2.0, 3.0]], dtype=xp.float32)
    source_point_bev = xp.asarray([0.0, -sad, 0.0], dtype=xp.float32)
    target_point_bev = xp.asarray([0.0, sad, 0.0], dtype=xp.float32)
    rad_depth_ix = xp.asarray([True])
    lateral_cutoff = 50.0

    ix, rad_distances_sq, lat_dists, iso_lat_dists = engine.calc_geo_dists(
        rot_coords_bev, source_point_bev, target_point_bev, sad, rad_depth_ix, lateral_cutoff
    )

    assert ix.dtype == xp.bool or ix.dtype == bool
    assert lat_dists.dtype == xp.float32
    assert iso_lat_dists.dtype == xp.float32
