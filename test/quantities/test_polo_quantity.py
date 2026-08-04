import pytest
import numpy as np
import array_api_strict as xp

from pyRadPlan.dij import Dij
from pyRadPlan.quantities import POLO, LinearPOLO, QuantityResolver


@pytest.fixture
def sample_base_dij_dict():
    dij_dict = {
        "ct_grid": {
            "resolution": {"x": 1.5, "y": 1.5, "z": 1.5},
            "dimensions": (10, 10, 10),
            "num_of_voxels": 1000,
        },
        "dose_grid": {
            "resolution": {"x": 3.0, "y": 3.0, "z": 3.0},
            "dimensions": (5, 5, 5),
            "num_of_voxels": 125,
        },
        "num_of_beams": 1,
        "total_num_of_bixels": 10,
        "physical_dose": np.empty((1, 1, 1), dtype=object),
        "let_dose": np.empty((1, 1, 1), dtype=object),
        "bixel_num": np.arange(10),
        "ray_num": np.arange(10),
        "beam_num": np.zeros((10,), dtype=np.int64),
    }
    return dij_dict


@pytest.fixture
def sample_dij_dense(sample_base_dij_dict):
    sample_base_dij_dict["physical_dose"].flat[0] = np.ones((125, 10), dtype=np.float32)
    sample_base_dij_dict["let_dose"].flat[0] = np.ones((125, 10), dtype=np.float32)
    dij = Dij.model_validate(sample_base_dij_dict)
    return dij


def test_polo_quantity_initialization_and_compute(sample_dij_dense):
    polo = POLO(sample_dij_dense, risk_voi_name=None)

    risk_mask = np.zeros(125, dtype=bool)
    risk_mask[:10] = True
    polo.binary_risk_mask = polo.array_backend.asarray(risk_mask, dtype=polo._dtype)

    assert polo.identifier == "polo"
    # Dose grid resolution (3, 3, 3) -> v_new = 27 mm^3. k = 27 / 1.08 = 25.0
    assert np.isclose(polo.k, 27.0 / 1.08)

    fluence = xp.arange(10, dtype=xp.float32)
    p_val = polo.compute(fluence)
    assert p_val.flat[0].shape[0] == 125


def test_linear_polo_quantity_compute(sample_dij_dense):
    lin_polo = LinearPOLO(sample_dij_dense, risk_voi_name=None)

    risk_mask = np.zeros(125, dtype=bool)
    risk_mask[:10] = True
    lin_polo.binary_risk_mask = lin_polo.array_backend.asarray(risk_mask, dtype=lin_polo._dtype)

    assert lin_polo.identifier == "linear_polo"

    fluence = xp.arange(10, dtype=xp.float32)
    eta = lin_polo.compute(fluence)
    assert eta.flat[0].shape[0] == 125


def test_polo_quantity_cst_lookup(sample_dij_dense):
    class MockVOI:
        def __init__(self, name, mask):
            self.name = name
            self.mask = mask

    class MockCST:
        def __init__(self, vois):
            self.vois = vois

    risk_mask = np.zeros(125, dtype=bool)
    risk_mask[:15] = True
    cst = MockCST([MockVOI("Ventricles_4mm", risk_mask)])

    polo = POLO(sample_dij_dense, cst=cst, risk_voi_name="Ventricles_4mm")
    assert polo.risk_voi_name == "Ventricles_4mm"
    assert np.array_equal(polo.binary_risk_mask, risk_mask)
