import pytest
import SimpleITK as sitk
import numpy as np
from typing import Union

# from pyRadPlan import plot_slice
from pyRadPlan.plan import Plan
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.dij import Dij
from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.machines import IonAccelerator
from pyRadPlan.core import Grid

from pyRadPlan.dose.engines import (
    DoseEngineBase,
)
from pyRadPlan.scenarios import NominalScenario


@pytest.fixture
def fake_dij(test_data_protons):
    dij_matRad = test_data_protons[4]
    dose_mat = np.ones((2000, 12))

    dij_matRad.physical_dose = np.empty((1,), dtype=object)
    dij_matRad.physical_dose[0] = dose_mat
    return dij_matRad


class DummyDoseEngine(DoseEngineBase):
    # short_name = "dummy"
    name = "Dummy Dose Engine"
    possible_radiation_modes = ["dummy"]

    def __init__(self, pln: Union[Plan, dict], fake_dij=[]):
        super().__init__(pln)
        self._fake_dij = fake_dij

    def _calc_dose(self, ct: CT, cst: StructureSet, stf: SteeringInformation) -> dict:
        return self._fake_dij


def test_DoseEngineBase_init(test_data_protons):
    dose_engine = DummyDoseEngine(pln=test_data_protons[0])

    assert isinstance(dose_engine, DoseEngineBase)
    assert dose_engine.name == "Dummy Dose Engine"
    assert dose_engine.is_dose_engine == True
    assert dose_engine.dose_grid["resolution"] == {"x": 10, "y": 10, "z": 10}
    assert dose_engine._ct_grid == None
    assert isinstance(dose_engine.mult_scen, NominalScenario)


def test_assign_properties_from_pln(test_data_protons):
    pln = test_data_protons[0]
    dose_engine = DummyDoseEngine(pln)

    assert dose_engine.dose_grid["resolution"] == {"x": 10, "y": 10, "z": 10}

    pln.prop_dose_calc["dose_grid"]["resolution"] = {"x": 20, "y": 15, "z": 10}
    dose_engine.assign_properties_from_pln(pln)

    assert dose_engine.name == "Dummy Dose Engine"
    assert dose_engine.is_dose_engine == True
    assert dose_engine.dose_grid["resolution"] == {"x": 20, "y": 15, "z": 10}


def test_calc_dose_forward(test_data_protons, fake_dij):
    pln, ct, cst, stf, _, _ = test_data_protons
    dose_engine = DummyDoseEngine(pln, fake_dij=fake_dij)
    result_py = dose_engine.calc_dose_forward(ct, cst, stf, w=None)

    assert isinstance(result_py["physical_dose"], sitk.Image)
    assert isinstance(result_py["let"], sitk.Image)

    result_py = sitk.GetArrayFromImage(result_py["physical_dose"])

    assert result_py.shape == (10, 20, 10)
    assert np.all(result_py == 12)


def test_calc_dose_influence(test_data_protons, fake_dij):
    pln, ct, cst, stf, _, _ = test_data_protons

    dose_engine = DummyDoseEngine(pln, fake_dij=fake_dij)
    result_py = dose_engine.calc_dose_influence(ct, cst, stf)

    assert isinstance(result_py, Dij)


def test_init_dose_calc(test_data_protons):
    pln, ct, cst, stf, _, _ = test_data_protons

    dose_engine = DummyDoseEngine(pln)
    dij = dose_engine._init_dose_calc(ct, cst, stf)

    assert dose_engine._num_of_columns_dij == 12

    assert len(dose_engine._vdose_grid) == 1152
    assert min(dose_engine._vdose_grid) == 211
    assert max(dose_engine._vdose_grid) == 1788

    assert len(dose_engine._vox_world_coords) == 1152
    assert np.array_equal(dose_engine._vox_world_coords[0], (-40.0, -90.0, -40.0))
    assert np.array_equal(dose_engine._vox_world_coords[-1], (30.0, 80.0, 30.0))
    assert np.array_equal(dose_engine._vox_world_coords, dose_engine._vox_world_coords_dose_grid)

    assert dose_engine._vdose_grid_mask.shape == (2000,)
    assert np.array_equal(dose_engine._vdose_grid_mask, dose_engine._vct_grid_mask)
    assert sum(dose_engine._vdose_grid_mask) == 1152

    assert isinstance(dose_engine._machine, IonAccelerator)
    assert isinstance(dose_engine._ct_grid, Grid)
