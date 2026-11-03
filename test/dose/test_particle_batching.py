"""Production particle packets preserve support, quantities and scalar extensibility."""

from types import SimpleNamespace

import array_api_compat
import numpy as np
import pytest
from pydantic import Field

from pyRadPlan import settings
from pyRadPlan.core.xp_utils import to_numpy
from pyRadPlan.machines import load_from_name
from pyRadPlan.dose.engines import ParticleHongPencilBeamEngine as Hong
from pyRadPlan.dose.engines._base_pencilbeam_particle import ParticlePencilBeamEngineAbstract
from pyRadPlan.dose.engines._particle_batch import ParticleBatchProcessor, _SelectedPacket
from pyRadPlan.dose.engines._batching import BixelDelivery, BixelWork


def assert_dij_equal(expected, actual):
    """Compare exact sparse structure and every influence quantity."""
    assert expected.quantities == actual.quantities
    for name in expected.quantities:
        for a, b in zip(getattr(expected, name).flat, getattr(actual, name).flat):
            np.testing.assert_array_equal(a.indptr, b.indptr)
            np.testing.assert_array_equal(a.indices, b.indices)
            np.testing.assert_allclose(a.data, b.data, rtol=2e-6, atol=1e-12)
    for name in ("beam_num", "ray_num", "bixel_num"):
        np.testing.assert_array_equal(getattr(expected, name), getattr(actual, name))


@pytest.mark.parametrize(
    "modality,model",
    [
        ("protons", "single"),
        ("protons", "double"),
        ("carbon", "single"),
        ("carbon", "multi"),
        ("helium", "single"),
        ("oxygen", "single"),
    ],
)
def test_particle_modalities_match_scalar(request, monkeypatch, modality, model):
    """Use existing physical/biological fixtures and verify repeated engine use."""
    monkeypatch.setattr(settings.xp, "prefer_gpu", False)
    monkeypatch.setattr(settings.xp, "preferred_cpu_array_backend", "numpy")
    plan, ct, cst, stf, *_ = request.getfixturevalue(f"test_data_{modality}")
    plan.prop_dose_calc["lateral_model"] = model
    if model == "multi":
        machine = load_from_name(modality, "Generic")
        for kernel in machine.pb_kernels.values():
            kernel.sigma_multi = np.stack([kernel.sigma, 2 * kernel.sigma, 3 * kernel.sigma])
            kernel.weight_multi = np.full((2, kernel.depths.shape[0]), 0.2)
        plan.machine = machine
    scalar, batched = Hong(plan), Hong(plan)
    calls = []
    original = ParticleBatchProcessor.evaluate

    def record(processor, batch):
        calls.append(len(batch))
        return original(processor, batch)

    monkeypatch.setattr(ParticleBatchProcessor, "evaluate", record)
    batched.batch_size = 8
    scalar.console_progress = batched.console_progress = False
    expected = scalar.calc_dose_influence(ct, cst, stf.model_copy(deep=True))
    actual = batched.calc_dose_influence(ct, cst, stf.model_copy(deep=True))
    assert calls
    count = len(calls)
    assert_dij_equal(expected, actual)
    batched.batch_selection_max_entries = 1
    batched.batch_evaluation_max_entries = 1
    assert_dij_equal(expected, batched.calc_dose_influence(ct, cst, stf.model_copy(deep=True)))
    assert len(calls) == count


@pytest.mark.parametrize("backend", ["numpy", "torch", "array_api_strict"])
@pytest.mark.parametrize("reuse", [True, False])
def test_selected_packet_reuse_and_split(backend, reuse):
    """Reuse an intact packet by identity, but repack a changed set of members."""
    module = pytest.importorskip(backend)
    xp = array_api_compat.array_namespace(module.asarray([0.0]))
    proc = object.__new__(ParticleBatchProcessor)
    proc.xp, proc.device = xp, array_api_compat.device(module.asarray([0.0]))
    proc.fields = ["ix", "rad_depths"]
    proc.groups, proc.banks = {0: (0, 0)}, {0: {}}
    arrays = dict(ix=xp.arange(5, dtype=xp.int64), rad_depths=xp.arange(5, dtype=xp.float64))
    packet = _SelectedPacket(
        arrays, xp.asarray([0, 0, 2, 2, 2]), np.array([0, 2, 2, 5]), xp.asarray([0, 0, 0]), {}
    )
    bixels = [
        dict(
            energy_ix=0,
            _batch_packet=(packet, i),
            **{k: v[packet.cuts[i] : packet.cuts[i + 1]] for k, v in arrays.items()},
        )
        for i in range(3)
    ]
    if not reuse:
        bixels = bixels[1:]
    actual = proc._selected_packet(bixels)
    assert (actual is packet) == reuse
    np.testing.assert_array_equal(np.asarray(actual.cuts), [0, 2, 2, 5] if reuse else [0, 0, 3])
    assert all("_batch_packet" not in b for b in bixels)


def test_retained_packet_storage_counts_against_byte_limit():
    """A tiny slice may retain a much larger base packet in the selected queue."""
    proc = object.__new__(ParticleBatchProcessor)
    proc.xp = array_api_compat.array_namespace(np.arange(1))
    proc.engine = SimpleNamespace(batch_max_bytes=1024)
    proc.fields = ["rad_depths"]
    proc.groups, proc.banks = {0: (0, 0)}, {0: {"scratch_per_entry": 0}}
    packet = _SelectedPacket(
        {"rad_depths": np.zeros(1000)},
        np.zeros(1000, dtype=np.int64),
        np.array([0, 1, 1000]),
        np.array([0, 0]),
        {},
    )
    work = BixelWork(
        dict(energy_ix=0, rad_depths=packet.arrays["rad_depths"][:1], _batch_packet=(packet, 0)),
        BixelDelivery(0, 0, 0, 0, 0),
    )
    assert proc.cost(work, "evaluation").nbytes >= 16000


@pytest.mark.parametrize(
    "method",
    [
        "_compute_bixel",
        "_get_bixel_indices_on_ray",
        "_calc_particle_bixel",
        "_prepare_depth_interpolator",
        "_lateral_profile",
        "_build_bio_context",
        "_evaluate_bio_influence",
    ],
)
def test_custom_numerical_overrides_keep_scalar_path(method, caplog):
    """Explicit batching must not silently bypass an existing engine extension."""
    original = getattr(Hong, method)

    def override(self, *args):
        return original(self, *args)

    custom = type("CustomEngine", (Hong,), {method: override})
    engine = object.__new__(custom)
    engine.batch_size = 16
    assert engine._create_bixel_batch_processor({}, None) is None
    assert "custom numerical overrides" in caplog.text


@pytest.mark.parametrize(
    "config",
    [
        dict(batch_size=-1),
        dict(batch_size=True),
        dict(batch_selection_max_entries=0),
        dict(batch_max_bytes=-3),
    ],
)
def test_invalid_batch_settings_fail_before_dose_setup(config):
    """Validate direct engine assignments as well as serialized configuration."""
    engine = object.__new__(Hong)
    for name, value in config.items():
        setattr(engine, name, value)
    with pytest.raises(ValueError):
        engine._init_dose_calc(None, None, None)


def test_budget_validation_does_not_revalidate_unrelated_required_options(monkeypatch):
    """A custom engine's required config must not break the inherited scalar path."""

    class CustomEngine(Hong):
        required_option: int = Field(...)

    engine = object.__new__(CustomEngine)
    engine.required_option = 7
    monkeypatch.setattr(ParticlePencilBeamEngineAbstract, "_init_dose_calc", lambda *args: "ready")
    assert engine._init_dose_calc(None, None, None) == "ready"


@pytest.mark.parametrize("backend", ["array_api_strict", "torch"])
def test_proton_support_on_other_backends(test_data_protons, monkeypatch, backend):
    """Exercise complete support selection, interpolation and output on portable backends."""
    pytest.importorskip(backend)
    monkeypatch.setattr(settings.xp, "prefer_gpu", False)
    monkeypatch.setattr(settings.xp, "preferred_cpu_array_backend", backend)
    plan, ct, cst, stf, *_ = test_data_protons
    scalar, batched = Hong(plan), Hong(plan)
    scalar.console_progress = batched.console_progress = False
    batched.batch_size = 8
    expected = scalar.calc_dose_influence(ct, cst, stf.model_copy(deep=True))
    assert_dij_equal(expected, batched.calc_dose_influence(ct, cst, stf.model_copy(deep=True)))


def test_production_processor_reuse_after_failure(test_data_protons, monkeypatch):
    """A failed packet must not leak pending data into the next engine calculation."""
    monkeypatch.setattr(settings.xp, "prefer_gpu", False)
    monkeypatch.setattr(settings.xp, "preferred_cpu_array_backend", "numpy")
    plan, ct, cst, stf, *_ = test_data_protons
    engine = Hong(plan)
    engine.console_progress = False
    expected = engine.calc_dose_influence(ct, cst, stf.model_copy(deep=True))
    engine.batch_size = 8
    original = ParticleBatchProcessor.evaluate

    def fail(processor, batch):
        raise RuntimeError("injected production packet failure")

    monkeypatch.setattr(ParticleBatchProcessor, "evaluate", fail)
    with pytest.raises(RuntimeError, match="injected production packet failure"):
        engine.calc_dose_influence(ct, cst, stf.model_copy(deep=True))
    monkeypatch.setattr(ParticleBatchProcessor, "evaluate", original)
    assert_dij_equal(expected, engine.calc_dose_influence(ct, cst, stf.model_copy(deep=True)))


@pytest.mark.parametrize("backend", ["numpy", "torch", "array_api_strict", "cupy"])
@pytest.mark.parametrize("model", ["single", "double", "multi", "singleXY"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("lengths", [(2, 0, 3), (0, 0, 0)])
def test_packed_physics_matches_scalar(backend, model, dtype, lengths):
    """Compare real Hong physics for ragged/empty packets and every lateral model."""
    module = pytest.importorskip(backend)
    if backend == "cupy" and not module.cuda.is_available():
        pytest.skip("CUDA unavailable")
    xp = array_api_compat.array_namespace(module.asarray([0.0]))
    n = sum(lengths)
    dtype = getattr(xp, dtype)
    engine = object.__new__(Hong)
    engine.lateral_model, engine._calc_bio_dose, engine._calc_let = model, False, True
    fields = dict(idd=xp.ones(n, dtype=dtype), let=xp.full((n,), 2.0, dtype=dtype))
    for key in engine._lateral_kernel_names():
        shape = (3, n) if key == "sigma_multi" else (2, n) if key == "weight_multi" else (n,)
        fields[key] = xp.full(shape, 0.2 if "weight" in key else 2.0, dtype=dtype)
    proc = object.__new__(ParticleBatchProcessor)
    proc.xp, proc.device, proc.engine = xp, array_api_compat.device(fields["idd"]), engine
    arrays = dict(
        ix=xp.arange(n, dtype=xp.int64),
        rad_depths=xp.ones(n, dtype=dtype),
        radial_dist_sq=xp.ones(n, dtype=dtype),
        lat_dists=xp.ones((n, 2), dtype=dtype),
    )
    proc.fields = list(arrays)
    cuts = np.cumsum([0, *lengths])
    owner = xp.asarray(np.repeat(np.arange(len(lengths)), lengths), dtype=xp.int64)
    bank = dict(end=xp.asarray([10.0], dtype=dtype), depth=lambda *args, **kwargs: fields)
    packet = _SelectedPacket(arrays, owner, cuts, xp.zeros(len(lengths), dtype=xp.int64), bank)
    bixels = [
        dict(
            energy_ix=0,
            sigma_ini_sq=xp.asarray(0.4, dtype=dtype),
            rad_depth_offset=0.0,
            kernel={"lateral_cut_off": {"comp_fac": 1.12}},
            _part=slice(cuts[i], cuts[i + 1]),
            _batch_packet=(packet, i),
            **{k: v[int(cuts[i]) : int(cuts[i + 1]), ...] for k, v in arrays.items()},
        )
        for i in range(len(lengths))
    ]
    engine._interpolate_kernels_in_depth = lambda b: {
        k: v[..., b["_part"]] for k, v in fields.items()
    }
    reference = [dict(b) for b in bixels]
    for b in reference:
        engine._calc_particle_bixel(b)
    batch = tuple(BixelWork(b, BixelDelivery(0, 0, 0, i, i)) for i, b in enumerate(bixels))
    proc.evaluate(batch)
    quantities, _, _ = bixels[0].pop("_batch_output")
    for key, actual in quantities.items():
        expected = xp.concat([b[key] for b in reference])
        np.testing.assert_allclose(to_numpy(actual), to_numpy(expected), rtol=2e-6, atol=1e-12)
    if n:
        for invalid in (-1.0, float("nan")):
            fields["idd"][0] = invalid
            for i, b in enumerate(bixels):
                b["_batch_packet"] = packet, i
            with pytest.raises(ValueError, match="Error in particle dose calculation"):
                proc.evaluate(batch)
