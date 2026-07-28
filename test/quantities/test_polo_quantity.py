import pytest
import numpy as np

from pyRadPlan.quantities import POLO, LinearPOLO


class MockDoseDep:
    def compute(self, w_cache):
        import array_api_compat

        xp = array_api_compat.array_namespace(np.zeros(1))

        class MockRet:
            flat = [xp.asarray([10.0, 20.0, 5.0])]

        return MockRet()

    def compute_chain_derivative(self, grad, w_cache):
        class MockRet:
            flat = [np.array([1.0, 2.0, 3.0])]

        return MockRet()


class MockDij:
    total_voxels = 3
    physical_dose = np.array([1.0], dtype=np.float32)

    def to_namespace(self, xp, device=None):
        return self

    class dose_grid:
        resolution = {"x": 5.0, "y": 5.0, "z": 5.0}


def test_polo_quantity_initialization_and_compute():
    dij = MockDij()
    deps = {"physical_dose": MockDoseDep(), "let_dose": MockDoseDep()}
    polo = POLO(dij, binary_risk_mask=np.array([1, 0, 1]), dependencies=deps)
    polo.binary_risk_mask = polo.array_backend.asarray(polo.binary_risk_mask)
    assert polo.identifier == "polo"
    assert np.isclose(polo.k, 125.0 / 1.08)

    val = polo._compute_quantity_single_scenario(0)
    assert val.shape[0] == 3


def test_linear_polo_quantity_compute():
    dij = MockDij()
    risk_mask = np.array([1, 0, 1])
    deps = {"physical_dose": MockDoseDep(), "let_dose": MockDoseDep()}

    lin_polo = LinearPOLO(dij, binary_risk_mask=risk_mask, dependencies=deps)
    assert lin_polo.identifier == "linear_polo"

    eta = lin_polo._compute_quantity_single_scenario(0)
    assert eta.shape[0] == 3
