"""
Python package for radiation therapy treatment planning.

We recommend reading the documentation on `GitHub <https://github.com/e0404/pyRadPlan>`_.

This package provides
- Core classes and functions for treatment planning.
- Dose influence matrix calculations and data structures.
- Physical and biological quantities for treatment planning.
- Optimization and analysis tools for radiation therapy plans.
- Sequencing and visualization tools for radiation therapy plans.

Import packages as follows:

    from pyRadPlan import (
        load_tg119,
        PhotonPlan,
        generate_stf,
        calc_dose_influence,
        fluence_optimization,
        plot_slice,
    )

Use the documentation, docstrings or examples for a detailed overview.
"""

from importlib.metadata import version, PackageNotFoundError
import logging

from .plan._plans import Plan, validate_pln, IonPlan, PhotonPlan
from .ct._ct import CT, validate_ct
from .cst._cst import StructureSet, validate_cst
from .stf._generate_stf import generate_stf
from .stf import SteeringInformation, validate_stf
from .dose._calc_dose import calc_dose_influence, calc_dose_forward
from .optimization._fluence_optimization import fluence_optimization
from .analysis._dvh import DVH, DVHCollection
from .visualization import plot_slice
from .io import load_patient, load_tg119
from .core import xp_utils

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass

# Logging is not exposed by default and needs to be configured by the user.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "__version__",
    "xp_utils",
    "Plan",
    "IonPlan",
    "PhotonPlan",
    "validate_pln",
    "CT",
    "validate_ct",
    "StructureSet",
    "validate_cst",
    "generate_stf",
    "calc_dose_influence",
    "calc_dose_forward",
    "fluence_optimization",
    "DVH",
    "DVHCollection",
    "SteeringInformation",
    "validate_stf",
    "plot_slice",
    "load_patient",
    "load_tg119",
]
