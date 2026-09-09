"""Biological capabilities of the FRED engine.

FRED scores LET itself and derives the biological influence matrices from it, so a LET-based
model must be accepted even for a machine without pencil-beam LET kernels. Conversely, an
evaluator needing further machine kernel inputs has to be rejected before a simulation runs.
"""

import copy
import logging

import numpy as np
import pytest
from scipy import sparse

from pyRadPlan.bio_models import (
    BioModelEvaluator,
    KernelBasedLQModel,
    LQModel,
    ParametricEvaluator,
    Wedenberg,
)
from pyRadPlan.dose.engines import (
    DoseEngineBase,
    ParticleFredMCEngine,
    ParticleHongPencilBeamEngine,
)
from pyRadPlan.plan import IonPlan


class _ExtraKernelInputModel(LQModel):
    """LET-based model whose evaluator also wants a machine kernel FRED cannot supply."""

    model = "fred_extra_kernel_input_test"
    possible_radiation_modes = ("protons",)
    required_quantities = ("physical_dose", "let")

    def evaluator(self, machine, voxel_params):
        return _ExtraKernelInputEvaluator(self)


class _ExtraKernelInputEvaluator(BioModelEvaluator):
    @property
    def kernel_field_names(self):
        return ("depth_factor",)

    @property
    def influence_quantity_names(self):
        return ("alpha_dose", "sqrt_beta_dose")


class _DirectInfluenceModel(LQModel):
    """LET-based model whose evaluator builds the influence quantities without intrinsic
    outputs; the unsupported-input check must not be bypassed by the empty declaration."""

    model = "fred_direct_influence_test"
    possible_radiation_modes = ("protons",)
    required_quantities = ("physical_dose", "let")
    output_quantities = ()

    def evaluator(self, machine, voxel_params):
        return _DirectInfluenceEvaluator(self)


class _DirectInfluenceEvaluator(BioModelEvaluator):
    @property
    def kernel_field_names(self):
        return ("depth_factor",)

    @property
    def influence_quantity_names(self):
        return ("alpha_dose",)


class _NoInfluenceModel(LQModel):
    """LET-based model whose evaluator produces no additive influence quantities at all."""

    model = "fred_no_influence_test"
    possible_radiation_modes = ("protons",)
    required_quantities = ("physical_dose", "let")
    output_quantities = ()

    def evaluator(self, machine, voxel_params):
        return BioModelEvaluator(self)


def _fred_engine(model, calc_bio_dose="auto"):
    """Fully constructed FRED engine (real ``__init__``) without a dose calculation."""
    pln = IonPlan(radiation_mode="protons", machine="Generic", bio_model=model)
    pln.prop_dose_calc = {"engine": "FRED", "calc_let": False, "calc_bio_dose": calc_bio_dose}
    engine = ParticleFredMCEngine(pln)
    engine._machine = None
    return engine


def _dij_inputs():
    return {"alphax": np.full((3, 1), 0.1), "betax": np.full((3, 1), 0.05)}


def _machine_without_let(radiation_mode="protons", machine_name="Generic"):
    machine = copy.deepcopy(DoseEngineBase.load_machine(radiation_mode, machine_name))
    for kernel in machine.pb_kernels.values():
        kernel.let = None
    assert not machine.has_let_kernel
    return machine


# --------------------------------------------------------------- real setup path
def test_fred_accepts_let_model_without_pencil_beam_let_kernels(test_data_protons, monkeypatch):
    """FRED scores LET, so the machine does not have to tabulate it."""
    pln, ct, cst, stf, _dij, _result = test_data_protons
    pln.bio_model = "WED"
    pln.prop_dose_calc = {"engine": "FRED", "external_calculation": True}
    machine = _machine_without_let()
    monkeypatch.setattr(DoseEngineBase, "load_machine", staticmethod(lambda *a, **k: machine))

    engine = ParticleFredMCEngine(pln)
    dij = engine._init_dose_calc(ct, cst, stf)

    assert engine.provided_quantities() == ["physical_dose", "let"]
    assert engine.engine_provided_quantities() == ["let"]
    assert "LETd" in engine.scorers
    assert engine._calc_bio_dose
    assert engine._bio_influence_names == ("alpha_dose", "sqrt_beta_dose")
    assert isinstance(engine._bio_evaluator, ParametricEvaluator)
    assert dij["bio_model"] == Wedenberg()
    assert dij["alpha_dose"] is not None and dij["sqrt_beta_dose"] is not None


def test_pencil_beam_engine_still_needs_tabulated_let(test_data_protons, monkeypatch):
    """The capability check is engine-aware: the pencil-beam engine has no LET of its own."""
    pln, ct, cst, stf, _dij, _result = test_data_protons
    pln.bio_model = "WED"
    machine = _machine_without_let()
    monkeypatch.setattr(DoseEngineBase, "load_machine", staticmethod(lambda *a, **k: machine))

    engine = ParticleHongPencilBeamEngine(pln)
    assert engine.engine_provided_quantities() == []
    with pytest.raises(ValueError, match="Required quantities not provided"):
        engine._init_dose_calc(ct, cst, stf)


def test_fred_setup_stores_constant_rbe(test_data_protons):
    """A constant-RBE model needs no influence matrices but must reach the dij."""
    pln, ct, cst, stf, _dij, _result = test_data_protons
    pln.bio_model = {"model": "constant_rbe", "rbe": 1.2}
    pln.prop_dose_calc = {"engine": "FRED", "external_calculation": True}

    engine = ParticleFredMCEngine(pln)
    dij = engine._init_dose_calc(ct, cst, stf)

    assert dij["bio_model"].rbe == pytest.approx(1.2)
    assert not engine._calc_bio_dose
    assert not engine._use_let_kernel
    assert "LETd" not in engine.scorers


# ------------------------------------------------- unsupported model combinations
def test_fred_skips_non_let_model_with_a_warning(caplog):
    engine = _fred_engine(KernelBasedLQModel())

    with caplog.at_level(logging.WARNING):
        engine._init_bio_model(_dij_inputs())

    assert engine._bio_evaluator is None
    assert not engine._calc_bio_dose
    assert "only supports LET-based evaluators" in caplog.text


def test_fred_rejects_non_let_model_when_requested_explicitly():
    engine = _fred_engine(KernelBasedLQModel(), calc_bio_dose=True)

    with pytest.raises(NotImplementedError, match="only supports LET-based evaluators"):
        engine._init_bio_model(_dij_inputs())

    assert engine._bio_evaluator is None


def test_fred_rejects_evaluator_inputs_it_cannot_supply():
    """A LET model is not enough: FRED cannot interpolate further machine kernels."""
    engine = _fred_engine(_ExtraKernelInputModel(), calc_bio_dose=True)

    with pytest.raises(NotImplementedError, match="depth_factor.*which FRED cannot supply"):
        engine._init_bio_model(_dij_inputs())

    assert engine._bio_evaluator is None


def test_fred_skips_evaluator_inputs_it_cannot_supply_on_auto(caplog):
    engine = _fred_engine(_ExtraKernelInputModel())

    with caplog.at_level(logging.WARNING):
        engine._init_bio_model(_dij_inputs())

    assert "which FRED cannot supply" in caplog.text
    assert engine._bio_evaluator is None
    assert not engine._calc_bio_dose


def test_fred_rejects_unsupported_inputs_without_intrinsic_output_declaration():
    """An evaluator producing influence directly must still be checked for its inputs."""
    engine = _fred_engine(_DirectInfluenceModel(), calc_bio_dose=True)

    with pytest.raises(NotImplementedError, match="depth_factor.*which FRED cannot supply"):
        engine._init_bio_model(_dij_inputs())

    assert engine._bio_evaluator is None


def test_fred_skips_unsupported_inputs_without_intrinsic_output_declaration(caplog):
    engine = _fred_engine(_DirectInfluenceModel())

    with caplog.at_level(logging.WARNING):
        engine._init_bio_model(_dij_inputs())

    assert "which FRED cannot supply" in caplog.text
    assert engine._bio_evaluator is None
    assert not engine._calc_bio_dose
    assert engine._bio_influence_names == ()


def test_fred_rejects_evaluator_without_influence_quantities():
    engine = _fred_engine(_NoInfluenceModel(), calc_bio_dose=True)

    with pytest.raises(NotImplementedError, match="no biological influence quantities"):
        engine._init_bio_model(_dij_inputs())

    assert engine._bio_evaluator is None


def test_fred_constructs_supported_let_evaluator():
    """FRED constructs and allocates the declared outputs of an LET-based evaluator."""
    engine = _fred_engine(Wedenberg())
    allocations = []
    engine._allocate_quantity_matrices = lambda dij, names: allocations.append(tuple(names)) or dij

    engine._init_bio_model(_dij_inputs())

    assert engine._bio_evaluator is not None
    assert engine._calc_bio_dose
    assert engine._bio_influence_names == ("alpha_dose", "sqrt_beta_dose")
    assert allocations == [("alpha_dose", "sqrt_beta_dose")]
    assert "LETd" in engine.scorers


def test_fred_calc_bio_dose_false_skips_everything():
    engine = _fred_engine(Wedenberg(), calc_bio_dose=False)

    engine._init_bio_model(_dij_inputs())

    assert engine._bio_evaluator is None
    assert not engine._calc_bio_dose
    # nothing needs the scored LET any more, so FRED does not request the scorer
    assert not engine._use_let_kernel
    assert "LETd" not in engine.scorers


# ---------------------------------------------------------------- output storage
def test_fred_stores_biological_matrices_derived_from_the_scored_let():
    """The stored matrices match the model evaluated on the LET of every entry."""
    engine = _fred_engine(Wedenberg())
    dij = _dij_inputs()
    engine._allocate_quantity_matrices = lambda d, names: d
    engine._init_bio_model(dij)

    dose = sparse.csc_array(np.asarray([[1.0, 0.0], [2.0, 3.0], [0.0, 0.0]]))
    let_dose = sparse.csc_array(np.asarray([[4.0, 0.0], [8.0, 15.0], [0.0, 0.0]]))
    for name in ("physical_dose", "alpha_dose", "sqrt_beta_dose", "let_dose"):
        dij[name] = np.empty((1, 1, 1), dtype=object)
    dij["physical_dose"].flat[0] = dose
    engine._calc_let = True

    engine._store_let_quantities(dij, let_dose)

    let = np.asarray([4.0, 4.0, 5.0])
    rbe_max = 1.0 + 0.434 * let / (0.1 / 0.05)
    d = np.asarray([1.0, 2.0, 3.0])
    assert np.allclose(np.sort(dij["alpha_dose"].flat[0].data), np.sort(d * rbe_max * 0.1))
    assert np.allclose(np.sort(dij["sqrt_beta_dose"].flat[0].data), np.sort(d * np.sqrt(0.05)))
    assert (dij["let_dose"].flat[0] != let_dose).nnz == 0
