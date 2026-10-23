"""All of pyRadPlan uses one pint unit registry."""

from pyRadPlan.analysis._dvh import ureg as analysis_ureg
from pyRadPlan.core._units import ureg
from pyRadPlan.quantities import Dose, RBExDose
from pyRadPlan.visualization._plot_slice import ureg as visualization_ureg


def test_single_shared_registry():
    assert analysis_ureg is ureg
    assert visualization_ureg is ureg
    assert Dose.unit._REGISTRY is ureg
    assert RBExDose.unit._REGISTRY is ureg


def test_units_of_different_modules_convert():
    assert (1.0 * Dose.unit).to(analysis_ureg.centigray).magnitude == 100.0
