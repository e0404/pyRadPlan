"""ScenarioSet: construction from points, queries, grouping, layout view and set algebra."""

import numpy as np
import pytest
from pydantic import ValidationError

from pyRadPlan.scenarios import (
    MATRAD_AXES,
    TAG_NOMINAL,
    CTPhaseUncertainty,
    IndependentGaussian,
    RangeUncertainty,
    Scenario,
    ScenarioBasis,
    ScenarioSet,
    SetupUncertainty,
)


def _dims(phases=(0,), probs=None):
    probs = [1.0 / len(phases)] * len(phases) if probs is None else probs
    return (
        CTPhaseUncertainty(phases=list(phases), probabilities=list(probs)),
        SetupUncertainty(),
        RangeUncertainty(),
    )


@pytest.fixture
def wc_like_set():
    """Two CT phases x (nominal, +x shift, range shift) - a matRad-like non-combined grid."""
    points = {
        "ct_phase": [0, 0, 0, 1, 1, 1],
        "setup": [[0, 0, 0], [2.25, 0, 0], [0, 0, 0], [0, 0, 0], [2.25, 0, 0], [0, 0, 0]],
        "range": [[0, 0], [0, 0], [1, 0.035], [0, 0], [0, 0], [1, 0.035]],
    }
    return ScenarioSet.from_points(_dims((0, 1)), points, axes=MATRAD_AXES)


def test_from_points_fills_missing_dimensions_with_nominal():
    s = ScenarioSet.from_points(_dims(), {"setup": [[0, 0, 0], [1, 2, 3]]})
    assert len(s) == 2
    assert np.array_equal(s.values("range"), np.zeros((2, 2)))
    assert np.array_equal(s.values("ct_phase"), np.zeros((2, 1)))
    assert s.dimension_names == ("ct_phase", "setup", "range")
    assert s.keys == ("ct_phase.index", "setup.x", "setup.y", "setup.z", "range.abs", "range.rel")


def test_from_points_probabilities_match_independent_gaussian_times_phase_mass():
    dims = _dims((0, 1), (0.25, 0.75))
    points = {"ct_phase": [0, 1, 1], "setup": [[0, 0, 0], [0, 0, 0], [2.25, 0, 0]]}
    s = ScenarioSet.from_points(dims, points)
    dist = IndependentGaussian(dims)
    cont = np.hstack([s.values("setup"), s.values("range")])
    expected = dist.pdf(cont) * np.array([0.25, 0.75, 0.75])
    assert np.allclose(s.probabilities, expected)
    assert np.isclose(s.weights.sum(), 1.0)
    assert np.allclose(s.weights, expected / expected.sum())
    assert s.distribution == "independent_gaussian"


def test_from_points_explicit_probabilities_and_weights():
    s = ScenarioSet.from_points(
        _dims(), {"setup": np.zeros((3, 3))}, probabilities=[1, 2, 3], weights=[0.5, 0.25, 0.25]
    )
    assert np.array_equal(s.probabilities, [1, 2, 3])
    assert np.array_equal(s.weights, [0.5, 0.25, 0.25])
    assert s.distribution is None


def test_from_points_rejects_values_outside_discrete_support():
    with pytest.raises(ValueError, match="support"):
        ScenarioSet.from_points(_dims((0,)), {"ct_phase": [0, 1]})


def test_from_points_rejects_unknown_dimension_and_bad_shapes():
    with pytest.raises(ValueError, match="unknown dimensions"):
        ScenarioSet.from_points(_dims(), {"bio": [[1.0]]})
    with pytest.raises(ValueError, match="shape"):
        ScenarioSet.from_points(_dims(), {"setup": [[0, 0], [1, 1]]})


def test_scenario_objects(wc_like_set):
    s = wc_like_set
    scen = s[1]
    assert isinstance(scen, Scenario)
    assert scen.index == 1
    assert scen.ct_phase == 0
    assert np.array_equal(scen["setup"], [2.25, 0, 0])
    assert scen.value("setup.x") == 2.25
    assert scen.value("ct_phase") == 0
    assert scen.is_computed and not scen.is_nominal
    assert s[-1].index == 5
    assert [sc.index for sc in s] == list(range(6))
    assert len(s.scenarios) == 6
    with pytest.raises(IndexError):
        s[6]
    with pytest.raises(TypeError):
        s[[0, 1]]
    assert "index=1" in repr(scen)


def test_values_extent_and_ct_phases(wc_like_set):
    s = wc_like_set
    assert s.values("setup").shape == (6, 3)
    assert np.array_equal(s.component_values("range.rel"), [0, 0, 0.035, 0, 0, 0.035])
    assert np.array_equal(s.extent("setup"), [2.25, 0, 0])
    assert np.allclose(s.extent("range"), [1, 0.035])
    with pytest.raises(TypeError):
        s.extent("ct_phase")
    assert np.array_equal(s.ct_phases(), [0, 0, 0, 1, 1, 1])
    assert s.ct_phase(4) == 1
    with pytest.raises(KeyError):
        s.values("bio")


def test_nominal_detection():
    s = ScenarioSet.from_points(_dims(), {"setup": [[1, 0, 0], [0, 0, 0]]})
    assert s.nominal_index() == 1  # found by value
    tagged = ScenarioSet.from_points(
        _dims(), {"setup": [[1, 0, 0], [0, 0, 0]]}, tags=[{TAG_NOMINAL}, {}]
    )
    assert tagged.nominal_index() == 0  # tag wins
    assert tagged[0].is_nominal
    assert np.array_equal(tagged.tagged(TAG_NOMINAL), [0])
    none = ScenarioSet.from_points(_dims(), {"setup": [[1, 0, 0]]})
    assert not none.has_nominal()
    with pytest.raises(ValueError):
        none.nominal_index()


def test_group_by_dimensions_and_components(wc_like_set):
    s = wc_like_set
    groups = s.group_by("ct_phase", "setup")
    assert [ix for _, ix in groups] == [[0, 2], [1], [3, 5], [4]]
    assert np.array_equal(groups[1][0]["setup"], [2.25, 0, 0])
    assert groups[1][0]["ct_phase"] == 0
    by_comp = s.group_by("setup.x")
    assert [ix for _, ix in by_comp] == [[0, 2, 3, 5], [1, 4]]
    assert by_comp[1][0] == {"setup.x": 2.25}
    assert s.group_by() == [({}, list(range(6)))]
    assert s.dimensions_affecting("geometry") == ("ct_phase", "setup")
    assert s.dimensions_affecting("range") == ("range",)
    assert s.dimensions_affecting("voxel_params") == ("ct_phase",)


def test_voxel_param_sets_follow_ct_phase(wc_like_set):
    s = wc_like_set
    assert s.num_voxel_param_sets == 2
    assert np.array_equal(s.voxel_param_set_indices(), [0, 0, 0, 1, 1, 1])
    assert s.voxel_param_set(5) == 1
    # a set using phases (0, 2) still has two columns, numbered by first appearance
    s2 = ScenarioSet.from_points(_dims((0, 2)), {"ct_phase": [2, 0, 2]})
    assert np.array_equal(s2.voxel_param_set_indices(), [0, 1, 0])
    # without any voxel-parameter dimension everything shares one column
    s3 = ScenarioSet.from_points((SetupUncertainty(),), {"setup": np.zeros((3, 3))})
    assert s3.num_voxel_param_sets == 1
    assert np.array_equal(s3.voxel_param_set_indices(), [0, 0, 0])
    assert np.array_equal(s3.ct_phases(), [0, 0, 0])


def test_require_supported(wc_like_set):
    wc_like_set.require_supported(("ct_phase", "setup", "range"))
    with pytest.raises(ValueError, match="range"):
        wc_like_set.require_supported(("ct_phase", "setup"))


def test_layout_matrad_axes(wc_like_set):
    layout = wc_like_set.layout
    assert layout.axes == (
        ("ct_phase.index",),
        ("setup.x", "setup.y", "setup.z"),
        ("range.abs", "range.rel"),
    )
    assert layout.axis_names == ("ct_phase", "setup", "range")
    assert layout.shape == (2, 2, 2)
    expected_pos = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1]])
    assert np.array_equal(layout.position, expected_pos)
    mask = layout.mask
    assert mask.shape == (2, 2, 2)
    assert mask.sum() == 6
    assert not mask[0, 1, 1] and not mask[1, 1, 1]
    assert not layout.is_dense
    assert np.array_equal(layout.coords[1][0], [0, 0, 0])  # nominal first
    assert np.array_equal(layout.coords[2][1], [1, 0.035])
    grid = layout.index_grid
    assert grid[1, 0, 1] == 5 and grid[0, 1, 1] == -1


def test_layout_nominal_first_even_if_not_first_in_points():
    s = ScenarioSet.from_points(
        _dims(), {"setup": [[1, 0, 0], [0, 0, 0], [-1, 0, 0]]}, axes=MATRAD_AXES
    )
    assert np.array_equal(s.layout.coords[1], [[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    assert np.array_equal(s.layout.position[:, 1], [1, 0, 2])


def test_with_axes_retie_and_per_component_axes(wc_like_set):
    s = wc_like_set
    tied = s.with_axes("ct_phase", ("setup", "range"))
    assert tied.layout.axis_names == ("ct_phase", "setup.x+setup.y+setup.z+range.abs+range.rel")
    assert tied.layout.shape == (2, 3)
    assert tied.layout.is_dense
    split = s.with_axes("setup.x", "setup.y", "setup.z")
    assert split.layout.axis_names == ("setup_x", "setup_y", "setup_z", "ct_phase", "range")
    assert split.layout.shape == (2, 1, 1, 2, 2)
    with pytest.raises(ValueError):
        s.with_axes("setup", "setup.x")
    with pytest.raises(ValueError):
        s.with_axes("bio")
    # layout_for does not change the set's default
    assert s.layout_for("ct_phase", ("setup", "range")).shape == (2, 3)
    assert s.layout.shape == (2, 2, 2)


def test_isel_sel_linear_index(wc_like_set):
    s = wc_like_set
    assert [sc.index for sc in s.isel(setup=1)] == [1, 4]
    assert [sc.index for sc in s.isel({"ct_phase": 1, "range": 1})] == [5]
    assert [sc.index for sc in s.sel(setup=(2.25, 0, 0))] == [1, 4]
    assert [sc.index for sc in s.sel(setup_x=2.25, ct_phase=1)] == [4]
    assert [sc.index for sc in s.sel({"range.rel": 0.035})] == [2, 5]
    assert s.sel(setup_x=2.2500001, atol=1e-6)[0].index == 1
    assert s.linear_index(ct_phase=1, setup=0, range=1) == 5
    with pytest.raises(KeyError):
        s.linear_index(ct_phase=0, setup=1, range=1)  # empty grid cell
    with pytest.raises(KeyError):
        s.linear_index(setup=1)  # partial
    with pytest.raises(IndexError):
        s.isel(setup=5)


def test_grid_view_places_objects_without_copy(wc_like_set):
    s = wc_like_set
    container = np.empty((6,), dtype=object)
    for i in range(6):
        container[i] = np.full((2, 2), i)
    view = s.grid_view(container)
    assert view.shape == (2, 2, 2)
    assert view[1, 0, 1] is container[5]
    assert view[0, 1, 1] is None
    with pytest.raises(ValueError):
        s.grid_view(container[:3])


def test_subset_reindexes_and_keeps_order(wc_like_set):
    sub = wc_like_set.subset([4, 1])
    assert len(sub) == 2
    assert [sc.index for sc in sub] == [0, 1]
    assert np.array_equal(sub.ct_phases(), [1, 0])
    assert np.array_equal(sub.values("setup")[:, 0], [2.25, 2.25])
    assert sub.axes == wc_like_set.axes
    with pytest.raises(IndexError):
        wc_like_set.subset([7])


def test_extend_appends_with_shifted_indices(wc_like_set):
    more = ScenarioSet.from_points(_dims((0, 1)), {"setup": [[0, 1, 0]]})
    ext = wc_like_set.extend(more)
    assert len(ext) == 7
    assert ext[6].index == 6
    assert np.array_equal(ext[6]["setup"], [0, 1, 0])
    assert ext.layout.shape == (2, 3, 2)  # new coordinate appended on the setup axis
    assert np.array_equal(ext.layout.position[:6], wc_like_set.layout.position)
    with pytest.raises(ValueError):
        wc_like_set.extend(ScenarioSet.from_points(_dims((0,)), {"setup": [[0, 0, 0]]}))


def test_derived_scenarios_bases():
    s = ScenarioSet.from_points(
        _dims(),
        {"setup": [[0, 0, 0], [1, 0, 0], [0, 0, 0]]},
        bases=[None, None, {"coefficients": {0: 1.0}}],
    )
    assert np.array_equal(s.computed_indices(), [0, 1])
    assert np.array_equal(s.derived_indices(), [2])
    assert s.num_computed == 2
    assert not s[2].is_computed
    assert s[2].basis.is_alias
    assert "derived" in repr(s[2])
    comp = s.computed()
    assert len(comp) == 2 and all(sc.is_computed for sc in comp)
    # extent ignores derived scenarios
    assert np.array_equal(s.extent("setup"), [1, 0, 0])
    # subset must keep the referenced scenario
    with pytest.raises(ValueError, match="not part of the subset"):
        s.subset([2])
    sub = s.subset([2, 0])
    assert sub[0].basis.coefficients == {1: 1.0}
    # extend shifts referenced indices
    ext = s.extend(s)
    assert ext[5].basis.coefficients == {3: 1.0}
    # invalid bases
    with pytest.raises(ValidationError, match="invalid scenario"):
        ScenarioSet.from_points(
            _dims(), {"setup": [[0, 0, 0]]}, bases=[{"coefficients": {0: 1.0}}]
        )
    with pytest.raises(ValidationError, match="derived scenario"):
        ScenarioSet.from_points(
            _dims(),
            {"setup": np.zeros((3, 3))},
            bases=[None, {"coefficients": {0: 1.0}}, {"coefficients": {1: 1.0}}],
        )
    with pytest.raises(ValidationError):
        ScenarioBasis(coefficients={})


def test_weights_helpers(wc_like_set):
    w = wc_like_set.with_weights(np.ones(6))
    assert np.array_equal(w.weights, np.ones(6))
    assert np.allclose(w.normalized().weights, np.full(6, 1 / 6))
    with pytest.raises(ValueError):
        wc_like_set.with_weights([-1, 1, 1, 1, 1, 1])


def test_serialization_round_trip(wc_like_set):
    s = wc_like_set
    as_dict = s.model_dump()
    assert as_dict["dimensions"][0]["dimension"] == "ct_phase"
    assert as_dict["dimensions"][0]["phases"] == [0, 1]
    r = ScenarioSet.model_validate(as_dict)
    assert r == s
    rj = ScenarioSet.model_validate_json(s.model_dump_json())
    assert rj == s
    assert np.array_equal(rj.layout.position, s.layout.position)
    assert rj[1].value("setup.x") == 2.25


def test_table_and_str(wc_like_set):
    rows = wc_like_set.to_table()
    assert len(rows) == 6 and rows[1][2] == 2.25
    text = str(wc_like_set)
    assert text.splitlines()[0].startswith("#\tct_phase.index\tsetup.x")
    assert "ScenarioSet(num_scenarios=6" in repr(wc_like_set)


def test_points_are_read_only(wc_like_set):
    with pytest.raises(ValueError):
        wc_like_set.values("setup")[0, 0] = 5.0
