"""Range shifter handling in the particle pencil-beam engines."""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.ndimage as ndi
import SimpleITK as sitk

from pyRadPlan.dose import calc_dose_forward
from pyRadPlan.dose.engines import ParticleHongPencilBeamEngine as HongEngine
from pyRadPlan.dose.engines._base_pencilbeam_particle import (
    ParticlePencilBeamEngineAbstract as ParticleEngine,
)

RASHI_THICKNESS = 20.0


def _engine_stub(radiation_mode):
    return SimpleNamespace(
        _machine=SimpleNamespace(radiation_mode=radiation_mode),
        rashi_scatter_ions=ParticleEngine.rashi_scatter_ions,
        _calc_sigma_rashi=lambda *args: ParticleEngine._calc_sigma_rashi(
            _engine_stub(radiation_mode), *args
        ),
    )


def _add_range_shifter(stf):
    for beam in stf.beams:
        for ray in beam.rays:
            for beamlet in ray.beamlets:
                beamlet.range_shifter.eq_thickness = RASHI_THICKNESS
                beamlet.range_shifter.source_rashi_distance = beam.sad - 400.0


def _physical_dose(ct, cst, stf, pln):
    return sitk.GetArrayFromImage(calc_dose_forward(ct, cst, stf, pln)["physical_dose"].values)


# Reference values from matRad_calcSigmaRashi (matRad dev, 3ebbc20a)
@pytest.mark.parametrize(
    ("radiation_mode", "bd_entry", "ssd", "expected"),
    [
        ("protons", {"range": 150.0, "energy": None}, 900.0, 3.094377306715),
        ("protons", {"range": None, "energy": 120.0}, 950.0, 4.769545465262),
        ("carbon", {"range": 150.0, "energy": None}, 900.0, 0.831805266743),
        ("carbon", 250.0, 900.0, 0.913096063298),
        ("helium", {"range": 100.0, "energy": None}, 920.0, 2.177983175068),
    ],
)
def test_sigma_rashi_matches_matrad(radiation_mode, bd_entry, ssd, expected):
    """Range shifter sigma for protons and heavier ions, from base data range or energy."""
    range_shifter = {"eq_thickness": 20.0, "source_rashi_distance": 700.0}
    sigma = ParticleEngine._calc_sigma_rashi(
        _engine_stub(radiation_mode), bd_entry, range_shifter, ssd
    )
    assert sigma == pytest.approx(expected, rel=1e-9)


def test_sigma_rashi_rejects_too_thick_range_shifter():
    """A range shifter close to the particle range cannot be modeled."""
    range_shifter = {"eq_thickness": 145.0, "source_rashi_distance": 700.0}
    with pytest.raises(ValueError, match="too thick"):
        ParticleEngine._calc_sigma_rashi(
            _engine_stub("protons"), {"range": 150.0}, range_shifter, 900.0
        )


@pytest.mark.parametrize("base_data_range", [0.0, -1.0])
def test_sigma_rashi_rejects_range_shifter_without_particle_range(base_data_range):
    """Base data without a positive range leaves no range to shift."""
    range_shifter = {"eq_thickness": 20.0, "source_rashi_distance": 700.0}
    with pytest.raises(ValueError, match="too thick"):
        ParticleEngine._calc_sigma_rashi(
            _engine_stub("protons"), {"range": base_data_range}, range_shifter, 900.0
        )


def test_sigma_rashi_takes_energy_when_range_is_not_finite():
    """A NaN range in the base data counts as missing."""
    range_shifter = {"eq_thickness": 20.0, "source_rashi_distance": 700.0}
    sigmas = [
        ParticleEngine._calc_sigma_rashi(
            _engine_stub("protons"),
            {"range": base_data_range, "energy": 120.0},
            range_shifter,
            950.0,
        )
        for base_data_range in (np.nan, None)
    ]
    assert sigmas[0] == sigmas[1]


def test_range_shifter_without_scatter_model_only_shifts_depth():
    """Radiation modes without a scattering model keep the depth offset and warn."""
    bixel = {
        "range_shifter": {"eq_thickness": 20.0, "source_rashi_distance": 700.0},
        "kernel": {"range": 150.0, "energy": 200.0},
        "SSD": 900.0,
        "rad_depth_offset": 1.5,
    }
    with pytest.warns(UserWarning, match="not implemented for radiation mode 'VHEE'"):
        ParticleEngine._get_beam_modifiers(_engine_stub("VHEE"), bixel)
    assert bixel["add_sigma_sq"] == 0
    assert bixel["rad_depth_offset"] == pytest.approx(21.5)


def test_protons_range_shifter_broadens_dose(test_data_protons):
    """A range shifter on every spot runs through the dose calculation and widens the spots.

    Regression test: the range shifter keys were read in camelCase (KeyError) and the
    scattering sigma never reached the lateral profile.
    """
    pln, ct, cst, stf, _, _ = test_data_protons
    dose_open = _physical_dose(ct, cst, stf, pln)
    _add_range_shifter(stf)
    dose_rashi = _physical_dose(ct, cst, stf, pln)

    def lateral_std(dose, axis):
        coords = np.arange(dose.shape[axis]).reshape([-1 if i == axis else 1 for i in range(3)])
        centre = ndi.center_of_mass(dose)[axis]
        return np.sqrt(((coords - centre) ** 2 * dose).sum() / dose.sum())

    # beams at gantry 0 and 180 run along y, the lateral axes are z and x
    for axis in (0, 2):
        assert lateral_std(dose_rashi, axis) > 1.1 * lateral_std(dose_open, axis)


def test_range_shifter_sigma_enters_every_lateral_profile_once(test_data_protons, monkeypatch):
    """The range shifter sigma is added once to the spots and once to the lateral cut-off."""
    pln, ct, cst, stf, _, _ = test_data_protons
    monkeypatch.setattr(ParticleEngine, "_calc_sigma_rashi", lambda *args: 3.0)
    sigmas_sq = []
    lateral_profile = HongEngine._lateral_profile

    def spy(self, radial_dist_sq, lat_dists, sigma_ini_sq, kernels):
        sigmas_sq.append(float(sigma_ini_sq))
        return lateral_profile(self, radial_dist_sq, lat_dists, sigma_ini_sq, kernels)

    monkeypatch.setattr(HongEngine, "_lateral_profile", spy)

    calc_dose_forward(ct, cst, stf, pln)
    sigmas_sq_open = np.unique(np.round(sigmas_sq, 6))
    sigmas_sq.clear()
    _add_range_shifter(stf)
    calc_dose_forward(ct, cst, stf, pln)

    np.testing.assert_allclose(np.unique(np.round(sigmas_sq, 6)), sigmas_sq_open + 3.0**2)


def test_range_shifter_widens_lateral_cut_off(test_data_protons, monkeypatch):
    """The lateral cut-off of every used energy accounts for the range shifter scattering."""
    pln, ct, cst, stf, _, _ = test_data_protons
    cut_offs = []
    calc_cut_off = ParticleEngine._calc_lateral_particle_cut_off

    def spy(self, cut_off_level, stf_element):
        calc_cut_off(self, cut_off_level, stf_element)
        energies = {b["energy"] for ray in stf_element["beam"]["rays"] for b in ray["beamlets"]}
        cut_offs.append(
            {
                energy: self._machine.get_kernel_by_energy(energy).lateral_cut_off.cut_off.copy()
                for energy in energies
            }
        )

    monkeypatch.setattr(ParticleEngine, "_calc_lateral_particle_cut_off", spy)

    calc_dose_forward(ct, cst, stf, pln)
    cut_offs_open = list(cut_offs)
    cut_offs.clear()
    _add_range_shifter(stf)
    calc_dose_forward(ct, cst, stf, pln)

    assert len(cut_offs) == len(cut_offs_open) > 0
    for beam_open, beam_rashi in zip(cut_offs_open, cut_offs):
        assert beam_open.keys() == beam_rashi.keys()
        for energy, cut_off_open in beam_open.items():
            assert np.all(beam_rashi[energy] > cut_off_open)


def test_lateral_cut_off_covers_range_shifter_on_part_of_the_spots(test_data_protons, monkeypatch):
    """Spots that differ only in their range shifter each count for the lateral cut-off.

    Regression test: of spots with the same energy, focus and SSD only the first one was
    considered, so a range shifter on a later one did not widen the cut-off.
    """
    pln, ct, cst, stf, _, _ = test_data_protons
    cut_offs = []
    calc_cut_off = ParticleEngine._calc_lateral_particle_cut_off

    def spy(self, cut_off_level, stf_element):
        calc_cut_off(self, cut_off_level, stf_element)
        cut_offs.append(
            {
                energy: self._machine.get_kernel_by_energy(energy).lateral_cut_off.cut_off.copy()
                for ray in stf_element["beam"]["rays"]
                for energy in {beamlet["energy"] for beamlet in ray["beamlets"]}
            }
        )

    monkeypatch.setattr(ParticleEngine, "_calc_lateral_particle_cut_off", spy)

    # every spot twice: first without, then with a range shifter
    for beam in stf.beams:
        for ray in beam.rays:
            ray.beamlets = [*ray.beamlets, *(b.model_copy(deep=True) for b in ray.beamlets)]
            for beamlet in ray.beamlets[len(ray.beamlets) // 2 :]:
                beamlet.range_shifter.eq_thickness = RASHI_THICKNESS
                beamlet.range_shifter.source_rashi_distance = beam.sad - 400.0
    calc_dose_forward(ct, cst, stf, pln)
    cut_offs_mixed = list(cut_offs)
    cut_offs.clear()
    _add_range_shifter(stf)
    calc_dose_forward(ct, cst, stf, pln)

    assert len(cut_offs) == len(cut_offs_mixed) > 0
    for beam_mixed, beam_rashi in zip(cut_offs_mixed, cut_offs):
        assert beam_mixed.keys() == beam_rashi.keys()
        for energy, cut_off_rashi in beam_rashi.items():
            np.testing.assert_array_equal(beam_mixed[energy], cut_off_rashi)


@pytest.mark.parametrize("test_data", ["test_data_protons", "test_data_carbon"])
def test_range_shifter_pulls_dose_back_by_its_thickness(test_data, request):
    """The depth dose of a beam moves towards the source by the range shifter thickness."""
    pln, ct, cst, stf, _, _ = request.getfixturevalue(test_data)
    stf.beams = stf.beams[:1]

    depth_dose_open = _physical_dose(ct, cst, stf, pln).sum(axis=(0, 2))
    _add_range_shifter(stf)
    depth_dose_rashi = _physical_dose(ct, cst, stf, pln).sum(axis=(0, 2))

    # the beam at gantry 0 runs along +y; the voxels at the surface and the far edge of the
    # dose grid have no counterpart in the other profile
    shift = round(RASHI_THICKNESS / ct.resolution["y"])
    assert shift > 0
    np.testing.assert_allclose(
        depth_dose_rashi[1 : -shift - 1],
        depth_dose_open[shift + 1 : -1],
        rtol=0.05,
        atol=1e-3 * depth_dose_open.max(),
    )
