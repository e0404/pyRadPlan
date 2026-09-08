"""Fragment species selection over spectra that aggregate isotopes by charge.

``FragmentFluence.A`` is NaN when an entry aggregates every isotope of its charge. Such
entries must still be selected, dose-averaged and matched per machine energy.
"""

import numpy as np
import pytest

from pyRadPlan.bio_models import TabulatedAlphaBetaModel
from pyRadPlan.machines.particles import ParticlePencilBeamKernel
from pyRadPlan.machines.particles._beam_fragment_spectrum import match_fragment_species

N_DEPTHS = 5
N_ENERGIES = 12


def _kernel(species, energy=100.0, seed=0):
    """Pencil-beam kernel whose spectrum carries the given ``(Z, A)`` species."""
    rng = np.random.default_rng(seed)
    energies = np.geomspace(1.0, 400.0, N_ENERGIES)
    spectra = [rng.random((N_ENERGIES, N_DEPTHS)) for _ in species]
    fluence = {
        "spectra": {
            "Z": np.asarray([z for z, _ in species]),
            "A": np.asarray([a for _, a in species], dtype=float),
            "fluenceSpectrum": spectra,
            "energyBin": [energies] * len(species),
            "fluenceDepth": [s.sum(axis=0) for s in spectra],
        }
    }
    return ParticlePencilBeamKernel(
        energy=energy,
        depths=np.linspace(0.0, 50.0, N_DEPTHS),
        Z=np.ones(N_DEPTHS),
        sigma=np.ones(N_DEPTHS),
        Fluence=fluence,
    )


def _expected_dose_averaged(model, kernel, quantity, transform=lambda x: x):
    """Reference dose average over every charged fragment of the spectrum."""
    numerator = np.zeros(N_DEPTHS)
    denominator = np.zeros(N_DEPTHS)
    for frag in kernel.fluence_spectrum.fragments:
        if frag.Z <= 0:
            continue
        sp_ix = np.flatnonzero(model._sp_table["fragments_AZ"][:, 1] == frag.Z)[0]
        q_ix = np.flatnonzero(model._q_table["fragments_AZ"][:, 1] == frag.Z)[0]
        sp = np.interp(
            frag.energy, model._sp_table["energies"][sp_ix], model._sp_table["dE_dx"][sp_ix]
        )
        q = transform(
            np.interp(
                frag.energy, model._q_table["energies"][q_ix], model._q_table[quantity][q_ix]
            )
        )
        denominator += (sp[:, None] * frag.fluence_spectrum).sum(axis=0)
        numerator += (q[:, None] * sp[:, None] * frag.fluence_spectrum).sum(axis=0)
    return numerator / denominator


@pytest.mark.parametrize(
    ("available", "species", "expected"),
    [
        ([[1.0, 1.0], [np.nan, 6.0]], (np.nan, 6.0), [1]),
        ([[1.0, 1.0], [np.nan, 6.0]], (1.0, 1.0), [0]),
        # an aggregate and a single isotope are different fragment populations
        ([[12.0, 6.0]], (np.nan, 6.0), []),
        ([[np.nan, 6.0]], (12.0, 6.0), []),
        ([[1.0, 1.0], [4.0, 2.0]], (np.nan, 2.0), []),
    ],
)
def test_match_fragment_species(available, species, expected):
    assert match_fragment_species(available, species).tolist() == expected


def test_select_fragments_keeps_aggregate_and_explicit_species():
    """A spectrum mixing explicit hydrogen and aggregate carbon keeps both."""
    kernel = _kernel([(1, 1.0), (6, np.nan)])
    selected = TabulatedAlphaBetaModel().select_fragments(kernel.fluence_spectrum)

    assert selected["kernel_ix"] == [0, 1]
    assert np.array_equal(
        np.asarray(selected["fragments_AZ"]),
        np.asarray([[1.0, 1.0], [np.nan, 6.0]]),
        equal_nan=True,
    )
    assert len(selected["sp_table_ix"]) == len(selected["q_table_ix"]) == 2


def test_select_fragments_accepts_aggregate_only_spectrum():
    kernel = _kernel([(6, np.nan)])
    selected = TabulatedAlphaBetaModel().select_fragments(kernel.fluence_spectrum)

    assert selected["kernel_ix"] == [0]
    assert selected["sp_table_ix"] == [
        int(np.flatnonzero(TabulatedAlphaBetaModel()._sp_table["fragments_AZ"][:, 1] == 6)[0])
    ]


def test_select_fragments_can_be_requested_explicitly_as_aggregate():
    kernel = _kernel([(1, 1.0), (6, np.nan)])
    model = TabulatedAlphaBetaModel(fragments_to_include=[[np.nan, 6.0]])
    selected = model.select_fragments(kernel.fluence_spectrum)

    assert selected["kernel_ix"] == [1]


def test_select_fragments_warns_for_genuinely_unavailable_species():
    """An isotope the spectrum does not carry is still reported and skipped."""
    kernel = _kernel([(1, 1.0), (6, np.nan)])
    model = TabulatedAlphaBetaModel(fragments_to_include=[[1.0, 1.0], [4.0, 2.0]])

    with pytest.warns(UserWarning, match="not present"):
        selected = model.select_fragments(kernel.fluence_spectrum)

    assert selected["kernel_ix"] == [0]


def test_select_fragments_does_not_confuse_aggregate_with_isotope():
    """Requesting C-12 must not silently pick up the aggregated carbon entry."""
    kernel = _kernel([(1, 1.0), (6, np.nan)])
    model = TabulatedAlphaBetaModel(fragments_to_include=[[12.0, 6.0]])

    with pytest.warns(UserWarning, match="not present"):
        with pytest.raises(ValueError, match="No fragment of the kernel spectra"):
            model.select_fragments(kernel.fluence_spectrum)


def test_dose_average_includes_the_aggregate_contribution():
    """The dose average over H + aggregate C matches the independent reference."""
    model = TabulatedAlphaBetaModel()
    kernel = _kernel([(1, 1.0), (6, np.nan)])
    tables = model.dose_average(kernel, model.select_fragments(kernel.fluence_spectrum))

    assert np.allclose(tables["alpha"][0], _expected_dose_averaged(model, kernel, "alpha"))
    assert np.allclose(
        tables["sqrt_beta"][0], _expected_dose_averaged(model, kernel, "beta", np.sqrt)
    )

    # dropping the aggregate would give a different (hydrogen-only) answer
    hydrogen_only = model.dose_average(
        kernel,
        TabulatedAlphaBetaModel(fragments_to_include=[[1.0, 1.0]]).select_fragments(
            kernel.fluence_spectrum
        ),
    )
    assert not np.allclose(tables["alpha"][0], hydrogen_only["alpha"][0])


def test_evaluator_selects_aggregate_species_per_energy():
    """Reordering the species of one energy must not change its dose-averaged table."""

    class _Machine:
        def __init__(self):
            self.energies = [100.0, 200.0]
            self.pb_kernels = {
                100.0: _kernel([(1, 1.0), (6, np.nan)], energy=100.0, seed=1),
                200.0: _kernel([(1, 1.0), (6, np.nan)], energy=200.0, seed=2),
            }

    machine = _Machine()
    model = TabulatedAlphaBetaModel()
    voxel_params = {"alpha_x": np.asarray([[0.1]]), "beta_x": np.asarray([[0.05]])}
    ordered = model.evaluator(machine, voxel_params)

    machine.pb_kernels[200.0].fluence_spectrum.fragments = list(
        reversed(machine.pb_kernels[200.0].fluence_spectrum.fragments)
    )
    reordered = model.evaluator(machine, voxel_params)

    for energy in machine.energies:
        for name in ("alpha", "sqrt_beta"):
            assert np.allclose(ordered._tables[energy][name], reordered._tables[energy][name])

    assert np.allclose(
        ordered._tables[200.0]["alpha"],
        _expected_dose_averaged(model, machine.pb_kernels[200.0], "alpha"),
    )


def test_spectrum_get_addresses_aggregate_entries():
    spectrum = _kernel([(1, 1.0), (6, np.nan)]).fluence_spectrum

    assert spectrum.get(6, float("nan")) is spectrum.fragments[1]
    assert spectrum.get(1, 1.0) is spectrum.fragments[0]
    assert spectrum.get(6, 12.0) is None
    assert spectrum.get(2, 4.0) is None


def test_spectrum_tables_are_cached_per_logical_device():
    """Every array_api_strict device shares one DLPack tuple, so it cannot be the cache key."""
    import array_api_compat
    import array_api_strict as xps

    class _Machine:
        energies = [100.0]
        pb_kernels = {100.0: _kernel([(1, 1.0), (6, np.nan)])}

    machine = _Machine()
    evaluator = TabulatedAlphaBetaModel().evaluator(
        machine, {"alpha_x": np.asarray([[0.1]]), "beta_x": np.asarray([[0.05]])}
    )

    # built by hand: xp_utils.to_namespace cannot express array_api_strict's logical devices
    depths = np.asarray(machine.pb_kernels[100.0].depths)
    devices = [xps.Device("CPU_DEVICE"), xps.Device("device1")]
    for device in devices:
        kernel = {"energy": 100.0, "depths": xps.asarray(depths, device=device)}
        quantities = evaluator.kernel_quantities(kernel)
        for values in quantities.values():
            assert array_api_compat.device(values) == device

    assert len(evaluator._converted) == len(devices)
