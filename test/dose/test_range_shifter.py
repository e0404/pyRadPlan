"""Range shifter handling in the particle pencil-beam engines."""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.ndimage as ndi
import SimpleITK as sitk

from pyRadPlan.dose import calc_dose_forward
from pyRadPlan.dose.engines._base_pencilbeam_particle import (
    ParticlePencilBeamEngineAbstract as ParticleEngine,
)


def _engine_stub(radiation_mode):
    return SimpleNamespace(
        _machine=SimpleNamespace(radiation_mode=radiation_mode),
        rashi_scatter_ions=ParticleEngine.rashi_scatter_ions,
        _calc_sigma_rashi=lambda *args: ParticleEngine._calc_sigma_rashi(
            _engine_stub(radiation_mode), *args
        ),
    )


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
    dose_open = sitk.GetArrayFromImage(calc_dose_forward(ct, cst, stf, pln)["physical_dose"])

    for beam in stf.beams:
        for ray in beam.rays:
            for beamlet in ray.beamlets:
                beamlet.range_shifter.eq_thickness = 20.0
                beamlet.range_shifter.source_rashi_distance = beam.sad - 400.0
    dose_rashi = sitk.GetArrayFromImage(calc_dose_forward(ct, cst, stf, pln)["physical_dose"])

    def lateral_std(dose, axis):
        coords = np.arange(dose.shape[axis]).reshape([-1 if i == axis else 1 for i in range(3)])
        centre = ndi.center_of_mass(dose)[axis]
        return np.sqrt(((coords - centre) ** 2 * dose).sum() / dose.sum())

    # beams at gantry 0 and 180 run along y, the lateral axes are z and x
    for axis in (0, 2):
        assert lateral_std(dose_rashi, axis) > 1.1 * lateral_std(dose_open, axis)
