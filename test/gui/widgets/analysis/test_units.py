import pytest

from pyRadPlan.analysis._dvh import ureg
from pyRadPlan.gui.widgets.analysis._units import UNIT_ALIASES, compare_units, safe_unit


@pytest.mark.parametrize(
    ("primary", "secondary", "expected"),
    [
        ("Gy", "cGy", (True, 0.01)),
        ("Gy", "Gy (RBE)", (True, 1.0)),
        ("Gy", "keV/µm", (False, 1.0)),
        ("", "", (True, 1.0)),
        ("Gy", "", (False, 1.0)),
        ("", "Gy", (False, 1.0)),
        ("bogus", "bogus", (True, 1.0)),
        ("Gy", "bogus", (False, 1.0)),
    ],
)
def test_compare_units(primary, secondary, expected):
    shared, scale = compare_units(primary, secondary)
    assert shared is expected[0]
    assert scale == pytest.approx(expected[1])


def test_safe_unit_aliases():
    assert set(UNIT_ALIASES) == {"Gy (RBE)", "Gy½"}
    assert safe_unit("Gy (RBE)") == ureg.gray
    assert safe_unit("Gy½") == ureg.Unit("gray**0.5")
    assert safe_unit("Gy") == ureg.gray


@pytest.mark.parametrize("unit_str", ["", "bogus"])
def test_safe_unit_fallback(unit_str):
    assert safe_unit(unit_str) == ureg.dimensionless
