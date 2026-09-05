import logging

import numpy as np
import pytest

from pyRadPlan.bio_models import KernelBasedLQModel, Wedenberg
from pyRadPlan.dose.engines import ParticleFredMCEngine


def _fred_bio_engine(model):
    engine = object.__new__(ParticleFredMCEngine)
    engine.bio_model = model
    engine.calc_bio_dose = "auto"
    engine.calc_let = False
    engine._machine = object()
    engine.scorers = ["Dose"]
    engine._computed_quantities = []
    return engine


def _dij_inputs():
    return {
        "alphax": np.asarray([[0.1]]),
        "betax": np.asarray([[0.05]]),
    }


def test_fred_does_not_construct_non_let_evaluator(caplog):
    """FRED skips kernel evaluator construction and reports the unsupported model."""
    engine = _fred_bio_engine(KernelBasedLQModel())

    with caplog.at_level(logging.WARNING):
        dij = engine._init_bio_model(_dij_inputs())

    assert engine._bio_evaluator is None
    assert not engine._calc_bio_dose
    assert dij["bio_model"] is engine.bio_model
    assert "FRED only supports LET-based biological evaluators" in caplog.text


def test_fred_constructs_supported_let_evaluator():
    """FRED constructs and allocates the declared outputs of an LET-based evaluator."""
    engine = _fred_bio_engine(Wedenberg())
    allocations = []
    engine._allocate_quantity_matrices = lambda dij, names: allocations.append(tuple(names)) or dij

    engine._init_bio_model(_dij_inputs())

    assert engine._bio_evaluator is not None
    assert engine._calc_bio_dose
    assert engine._bio_influence_names == ("alpha_dose", "sqrt_beta_dose")
    assert allocations == [("alpha_dose", "sqrt_beta_dose")]


def test_fred_rejects_explicit_non_let_bio_calculation():
    """An explicit biological request fails without constructing the kernel evaluator."""
    engine = _fred_bio_engine(KernelBasedLQModel())
    engine.calc_bio_dose = True

    with pytest.raises(NotImplementedError, match="only supports LET-based evaluators"):
        engine._init_bio_model(_dij_inputs())

    assert engine._bio_evaluator is None
