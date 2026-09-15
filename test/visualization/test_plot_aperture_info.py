import numpy as np
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
import pytest

from pyRadPlan import CT, plot_aperture_info
from pyRadPlan.cst import validate_cst
from pyRadPlan.ct import validate_ct
from pyRadPlan.machines import MLC
from pyRadPlan.result import PlanResult, Sequence
from pyRadPlan.result._beam_sequence import BeamSequence
from pyRadPlan.stf import Beam, Ray, SteeringInformation
from pyRadPlan.stf._beamlet import PhotonBixel
from pyRadPlan.visualization._plot_aperture_info import (
    _draw_mlc_borders,
    _draw_target_outline,
    _target_bev_points,
)

matplotlib.use("Agg")


def _make_mlc(leaf_width: float, opening: float) -> MLC:
    boundaries = np.array([-10.0, 0.0, 10.0])
    positions = np.array([[-opening, opening], [-opening, opening], [-opening, opening]])
    return MLC(
        device_type="MLCX",
        leaf_position_boundaries=boundaries,
        leaf_positions=positions,
        leaf_width=leaf_width,
    )


def _make_beam_sequence(n_shapes: int = 2) -> BeamSequence:
    mlc_sequence = [_make_mlc(leaf_width=10.0, opening=5.0 + s) for s in range(n_shapes)]
    return BeamSequence(
        shapes=np.ones((3, 4, n_shapes), dtype=np.uint8),
        shapes_weight=np.array([1.0 + s for s in range(n_shapes)]),
        bixel_ix=list(range(4)),
        fluence=np.ones(4),
        mlc_sequence=mlc_sequence,
        mlc_window=[-20.0, 20.0, -15.0, 15.0],
    )


@pytest.fixture
def sample_sequence():
    return Sequence(beam=[_make_beam_sequence(2), _make_beam_sequence(3)])


def test_plot_aperture_info_from_sequence(sample_sequence, tmp_path):
    figures = plot_aperture_info(
        sample_sequence, show_plot=False, save_filename=str(tmp_path / "apertures")
    )
    assert len(figures) == 2
    assert (tmp_path / "apertures_beam0.png").exists()
    assert (tmp_path / "apertures_beam1.png").exists()


def test_plot_aperture_info_from_result(sample_sequence):
    result = PlanResult(sequence=sample_sequence)
    figures = plot_aperture_info(result, show_plot=False)
    assert len(figures) == 2


def test_plot_aperture_info_beam_subset(sample_sequence):
    figures = plot_aperture_info(sample_sequence, beam_indices=1, show_plot=False)
    assert len(figures) == 1


def test_plot_aperture_info_physical_coordinates(sample_sequence):
    figures = plot_aperture_info(sample_sequence, leaf_coordinate="physical", show_plot=False)
    ax = figures[0].axes[0]
    assert ax.get_ylabel() == "vert. pos. [mm]"


def test_plot_aperture_info_invalid_leaf_coordinate(sample_sequence):
    with pytest.raises(ValueError):
        plot_aperture_info(sample_sequence, leaf_coordinate="bogus", show_plot=False)


def test_plot_aperture_info_no_sequence_raises():
    with pytest.raises(ValueError):
        plot_aperture_info(PlanResult(), show_plot=False)


def test_plot_aperture_info_unsequenced_result_requires_stf(sample_sequence):
    with pytest.raises(ValueError):
        plot_aperture_info(sample_sequence, unsequenced_result=PlanResult(), show_plot=False)


def _make_ray(x: float, z: float) -> Ray:
    return Ray(ray_pos_bev=[x, 1000.0, z], ray_pos=[x, 0.0, z], beamlets=[PhotonBixel(energy=6.0)])


@pytest.fixture
def fluence_comparison_setup():
    """A single 2x2-ray beam, consistent between stf and both fluence vectors."""
    xs = [-2.5, 2.5, -2.5, 2.5]
    zs = [-2.5, -2.5, 2.5, 2.5]
    rays = [_make_ray(x, z) for x, z in zip(xs, zs)]
    beam = Beam(iso_center=[0.0, 0.0, 0.0], bixel_width=5.0, rays=rays)
    stf = SteeringInformation(beams=[beam])

    mlc = _make_mlc(leaf_width=10.0, opening=2.0)
    beam_sequence = BeamSequence(
        shapes=np.ones((3, 4, 1), dtype=np.uint8),
        shapes_weight=np.array([1.0]),
        bixel_ix=list(range(4)),
        fluence=np.ones(4),
        mlc_sequence=[mlc],
        mlc_window=[-20.0, 20.0, -15.0, 15.0],
    )
    sequence = Sequence(beam=[beam_sequence], fluence=np.array([1.1, 2.1, 2.9, 3.9]))
    unsequenced_result = PlanResult(fluence=np.array([1.0, 2.0, 3.0, 4.0]))
    return stf, sequence, unsequenced_result


def test_plot_aperture_info_fluence_comparison(fluence_comparison_setup):
    stf, sequence, unsequenced_result = fluence_comparison_setup
    baseline_figures = plot_aperture_info(sequence, stf=stf, show_plot=False)

    figures = plot_aperture_info(
        sequence, stf=stf, unsequenced_result=unsequenced_result, show_plot=False
    )
    # fluence comparison is a separate figure from the shape grid.
    assert len(baseline_figures) == 1
    assert len(figures) == 2
    assert len(figures[0].axes) == len(baseline_figures[0].axes)
    assert len(figures[1].axes) > 0


def test_plot_aperture_info_fluence_comparison_keeps_true_aspect_ratio(fluence_comparison_setup):
    stf, sequence, unsequenced_result = fluence_comparison_setup
    figures = plot_aperture_info(
        sequence, stf=stf, unsequenced_result=unsequenced_result, show_plot=False
    )
    fluence_axes = [ax for ax in figures[1].axes if ax.get_images()]
    assert len(fluence_axes) == 3
    for ax in fluence_axes:
        assert ax.get_aspect() == 1.0  # "equal", not "auto" (which reports "auto")
        assert ax.get_xlabel() == "x (BEV) [mm]"
        assert ax.get_ylabel() == "z (BEV) [mm]"


def test_plot_aperture_info_fluence_difference_panel(fluence_comparison_setup):
    stf, sequence, unsequenced_result = fluence_comparison_setup
    figures = plot_aperture_info(
        sequence, stf=stf, unsequenced_result=unsequenced_result, show_plot=False
    )
    axes_by_title = {ax.get_title(): ax for ax in figures[1].axes}
    diff_ax = axes_by_title["Difference (sequencing − optimization)"]

    im = diff_ax.get_images()[0]
    assert im.get_cmap().name == "RdBu_r"
    vmin, vmax = im.get_clim()
    assert vmin == pytest.approx(-vmax)

    delivered = np.nan_to_num(sequence.fluence, nan=0.0)
    target = np.nan_to_num(unsequenced_result.fluence, nan=0.0)
    expected_max_abs = np.max(np.abs(delivered - target))
    assert vmax == pytest.approx(expected_max_abs)


def test_plot_aperture_info_fluence_comparison_saves_files(fluence_comparison_setup, tmp_path):
    stf, sequence, unsequenced_result = fluence_comparison_setup
    plot_aperture_info(
        sequence,
        stf=stf,
        unsequenced_result=unsequenced_result,
        show_plot=False,
        save_filename=str(tmp_path / "mlc"),
    )
    assert (tmp_path / "mlc_beam0.png").exists()
    assert (tmp_path / "mlc_beam0_comparison.png").exists()


@pytest.fixture
def dose_comparison_setup(fluence_comparison_setup):
    """Extends fluence_comparison_setup with per-beam dose on both results."""
    stf, sequence, unsequenced_result = fluence_comparison_setup
    ct = CT(cube_hu=sitk.Image(10, 10, 5, sitk.sitkFloat32))

    unsequenced_result = PlanResult(
        fluence=unsequenced_result.fluence,
        beam_quantities={"physical_dose": [np.random.rand(5, 10, 10).astype(np.float32)]},
    )
    sequenced_result = PlanResult(
        sequence=sequence,
        beam_quantities={
            "physical_dose_sequenced": [np.random.rand(5, 10, 10).astype(np.float32)]
        },
    )
    return stf, sequenced_result, unsequenced_result, ct


def test_plot_aperture_info_dose_comparison_row(dose_comparison_setup):
    stf, sequenced_result, unsequenced_result, ct = dose_comparison_setup
    baseline_figures = plot_aperture_info(
        sequenced_result, stf=stf, unsequenced_result=unsequenced_result, show_plot=False
    )
    figures = plot_aperture_info(
        sequenced_result,
        stf=stf,
        unsequenced_result=unsequenced_result,
        image_volume=ct,
        show_plot=False,
    )
    # both cases produce a separate (shape grid, comparison) figure pair;
    # the dose row adds axes to the comparison figure only.
    assert len(figures) == len(baseline_figures) == 2
    assert len(figures[0].axes) == len(baseline_figures[0].axes)
    assert len(figures[1].axes) > len(baseline_figures[1].axes)


def test_plot_aperture_info_dose_comparison_columns_align_with_fluence(dose_comparison_setup):
    stf, sequenced_result, unsequenced_result, ct = dose_comparison_setup
    figures = plot_aperture_info(
        sequenced_result,
        stf=stf,
        unsequenced_result=unsequenced_result,
        image_volume=ct,
        show_plot=False,
    )
    axes_by_title = {ax.get_title(): ax for ax in figures[1].axes}
    fluence_target = axes_by_title["Target fluence (optimization)"]
    fluence_delivered = axes_by_title["Delivered fluence (sequencing)"]
    fluence_diff = axes_by_title["Difference (sequencing − optimization)"]
    dose_optimized = axes_by_title["Target dose (optimization)"]
    dose_delivered = axes_by_title["Delivered dose (sequencing)"]
    dose_diff = axes_by_title["Difference (sequencing − optimization)"]

    # Compare GridSpec column spans directly (not the rendered bbox, which
    # `aspect="equal"` shrinks/centers independently for each panel based on
    # its own data's aspect ratio).
    def colspan(ax):
        s = ax.get_subplotspec()
        return s.colspan

    assert colspan(fluence_target) == colspan(dose_optimized)
    assert colspan(fluence_delivered) == colspan(dose_delivered)
    assert colspan(fluence_diff) == colspan(dose_diff)


def test_plot_aperture_info_dose_requires_unsequenced_result(dose_comparison_setup):
    stf, sequenced_result, _unsequenced_result, ct = dose_comparison_setup
    with pytest.raises(ValueError):
        plot_aperture_info(sequenced_result, stf=stf, image_volume=ct, show_plot=False)


def test_plot_aperture_info_dose_requires_plan_result(dose_comparison_setup):
    stf, sequenced_result, unsequenced_result, ct = dose_comparison_setup
    with pytest.raises(ValueError):
        plot_aperture_info(
            sequenced_result.sequence,
            stf=stf,
            unsequenced_result=unsequenced_result,
            image_volume=ct,
            show_plot=False,
        )


def test_target_bev_points_projects_target_onto_its_own_isocenter(test_data_photons_raw):
    ct = validate_ct(test_data_photons_raw["ct"])
    cst = validate_cst(test_data_photons_raw["cst"], ct=ct)
    center_of_mass = np.asarray(cst.target_center_of_mass()).ravel()

    beam = Beam(iso_center=center_of_mass.tolist(), bixel_width=5.0, rays=[])
    points = _target_bev_points(ct, cst, beam, "X")

    assert points is not None
    assert points.ndim == 2 and points.shape[1] == 2
    np.testing.assert_allclose(points.mean(axis=0), [0.0, 0.0], atol=1e-6)
    # Symmetric about the isocenter it is centered on, and bounded by the target.
    np.testing.assert_allclose(points.min(axis=0), -points.max(axis=0), rtol=1e-6)


def test_target_bev_points_returns_none_without_target(test_data_photons_raw):
    class _NoTargetCst:
        def target_union_mask(self):
            return None

    ct = CT(cube_hu=sitk.Image(4, 4, 4, sitk.sitkFloat32))
    beam = Beam(iso_center=[0.0, 0.0, 0.0], bixel_width=5.0, rays=[])

    assert _target_bev_points(ct, _NoTargetCst(), beam, "X") is None


def test_dose_comparison_names_available_quantities_when_delivered_dose_missing(
    dose_comparison_setup,
):
    stf, sequenced_result, unsequenced_result, ct = dose_comparison_setup
    without_delivered = PlanResult(
        sequence=sequenced_result.sequence,
        beam_quantities={
            "physical_dose": sequenced_result.beam_quantities["physical_dose_sequenced"]
        },
    )

    with pytest.raises(ValueError, match="physical_dose_sequenced"):
        plot_aperture_info(
            without_delivered,
            stf=stf,
            unsequenced_result=unsequenced_result,
            image_volume=ct,
            show_plot=False,
        )
