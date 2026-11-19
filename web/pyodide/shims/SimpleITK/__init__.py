"""Pyodide stand-in for the subset of SimpleITK that pyRadPlan's planning pipeline uses.

This is NOT SimpleITK. SimpleITK has no WebAssembly build, so the Pyodide harness in
``web/pyodide`` installs this numpy/scipy implementation under the import name
``SimpleITK`` instead. It provides the image container (origin / spacing / direction /
pixel type), the numpy conversions, casts and pixel-wise operators, identity-transform
resampling (nearest, linear, B-spline) with SimpleITK's boundary semantics, mask and
morphology filters and label shape statistics. Anything else raises
``NotImplementedError`` naming the missing feature; image file IO is not available.

The behaviour is checked against real SimpleITK by ``web/pyodide/tests``.
"""

from . import SimpleITK, _filters, _image, _resample  # noqa: F401 - SimpleITK: pickle path

for _module in (_image, _resample, _filters):
    for _name in dir(_module):
        if _name.startswith("sitk") or _name[:1].isupper():
            globals()[_name] = getattr(_module, _name)
del _module, _name

__version__ = "0.1.0+pyradplan.shim"
IS_PYRADPLAN_SHIM = True


def Version():
    """Return the shim version string."""
    return __version__


class ExtractImageFilter:
    """Holds the direction-collapse constants of :func:`Extract`."""

    DIRECTIONCOLLAPSETOIDENTITY = 1
    DIRECTIONCOLLAPSETOSUBMATRIX = 2
    DIRECTIONCOLLAPSETOGUESS = 3


InterpolatorEnum = int
PixelIDValueEnum = int


def __getattr__(name):
    raise AttributeError(
        f"SimpleITK.{name} is not available in the pyRadPlan Pyodide SimpleITK shim "
        "(web/pyodide/shims/SimpleITK)."
    )
