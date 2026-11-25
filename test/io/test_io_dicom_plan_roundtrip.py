"""Round trip of RT Ion Plans through the independently written exporter and importer."""

import numpy as np
import pytest

from pyRadPlan import IonPlan, generate_stf, load_tg119
from pyRadPlan.io import save_data
from pyRadPlan.io.dicom import DicomImporter


def _spots(stf, weights) -> list:
    """(energy, world position on the isocenter plane, weight) of every weighted spot."""
    spots = []
    index = 0
    for beam in stf.beams:
        for ray in beam.rays:
            position = np.asarray(beam.iso_center) + np.asarray(ray.ray_pos)
            for beamlet in ray.beamlets:
                weight = float(weights[index])
                index += 1
                if weight > 0:
                    spots.append((round(beamlet.energy, 3), *np.round(position, 3), weight))
    return sorted(spots)


@pytest.fixture(scope="module")
def tg119():
    return load_tg119()


@pytest.mark.parametrize(
    ("gantry", "couch"),
    [([0.0], [0.0]), ([90.0, 225.0], [30.0, 330.0]), ([270.0], [300.0])],
)
def test_ion_plan_round_trip(tg119, tmp_path, gantry, couch):
    ct, cst = tg119
    pln = IonPlan(radiation_mode="protons", machine="Generic", num_of_fractions=5)
    pln.prop_stf = {"gantry_angles": gantry, "couch_angles": couch, "bixel_width": 10}
    stf = generate_stf(ct, cst, pln)
    weights = np.random.default_rng(0).uniform(0.2, 1.0, stf.total_number_of_bixels)
    weights[::5] = 0.0

    save_data(
        file_name=tmp_path / "plan",
        format="dcm",
        ct=ct,
        cst=cst,
        pln=pln,
        stf=stf,
        weights=weights,
    )
    plan = DicomImporter(tmp_path / "plan").load_plan(ct=ct)

    assert plan["pln"].num_of_fractions == 5
    original = _spots(stf, weights)
    imported = _spots(plan["stf"], plan["result"].fluence)
    assert len(imported) == len(original)
    np.testing.assert_allclose(np.array(imported), np.array(original), rtol=1e-6, atol=1e-3)


def test_weights_round_trip_with_other_particles_per_mu(tg119, tmp_path):
    ct, cst = tg119
    pln = IonPlan(radiation_mode="protons", machine="Generic", num_of_fractions=1)
    pln.prop_stf = {"gantry_angles": [0.0], "couch_angles": [0.0], "bixel_width": 10}
    stf = generate_stf(ct, cst, pln)
    for beam in stf.beams:
        for ray in beam.rays:
            for beamlet in ray.beamlets:
                beamlet.num_particles_per_mu = 3.0e6
    weights = np.random.default_rng(0).uniform(0.2, 1.0, stf.total_number_of_bixels)

    save_data(
        file_name=tmp_path / "plan",
        format="dcm",
        ct=ct,
        cst=cst,
        pln=pln,
        stf=stf,
        weights=weights,
    )
    plan = DicomImporter(tmp_path / "plan").load_plan(ct=ct)

    np.testing.assert_allclose(sorted(plan["result"].fluence), sorted(weights), rtol=1e-5)
