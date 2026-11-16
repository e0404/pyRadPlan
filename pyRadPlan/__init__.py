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
from typing import TYPE_CHECKING

from ._lazy import attach_lazy

# The subpackages are imported on first access, so that ``import pyRadPlan`` (e.g. by
# the GUI before it can show its startup splash) does not load the numerical stack.
if TYPE_CHECKING:
    from ._settings import (
        GuiSettings as GuiSettings,
        PyRadPlanSettings as PyRadPlanSettings,
        XpSettings as XpSettings,
        settings as settings,
    )
    from .plan._plans import (
        Plan as Plan,
        validate_pln as validate_pln,
        IonPlan as IonPlan,
        PhotonPlan as PhotonPlan,
    )
    from .ct._ct import CT as CT, validate_ct as validate_ct
    from .cst._cst import StructureSet as StructureSet, validate_cst as validate_cst
    from .stf._generate_stf import generate_stf as generate_stf
    from .stf import SteeringInformation as SteeringInformation, validate_stf as validate_stf
    from .dose._calc_dose import (
        calc_dose_influence as calc_dose_influence,
        calc_dose_forward as calc_dose_forward,
    )
    from .optimization._fluence_optimization import (
        fluence_optimization as fluence_optimization,
        OptInfo as OptInfo,
    )
    from .analysis._dvh import DVH as DVH, DVHCollection as DVHCollection
    from .visualization import (
        plot_slice as plot_slice,
        plot_multiple_slices as plot_multiple_slices,
    )
    from .io import (
        load_patient as load_patient,
        load_data as load_data,
        save_data as save_data,
        load_binary_patient as load_binary_patient,
        load_tg119 as load_tg119,
    )
    from .core import xp_utils as xp_utils

__getattr__, __dir__, __all__ = attach_lazy(
    __name__,
    {
        "._settings": ("GuiSettings", "PyRadPlanSettings", "XpSettings", "settings"),
        ".plan._plans": ("Plan", "validate_pln", "IonPlan", "PhotonPlan"),
        ".ct._ct": ("CT", "validate_ct"),
        ".cst._cst": ("StructureSet", "validate_cst"),
        ".stf._generate_stf": ("generate_stf",),
        ".stf": ("SteeringInformation", "validate_stf"),
        ".dose._calc_dose": ("calc_dose_influence", "calc_dose_forward"),
        ".optimization._fluence_optimization": ("fluence_optimization", "OptInfo"),
        ".analysis._dvh": ("DVH", "DVHCollection"),
        ".visualization": ("plot_slice", "plot_multiple_slices"),
        ".io": ("load_patient", "load_data", "save_data", "load_binary_patient", "load_tg119"),
        ".core": ("xp_utils",),
    },
    eager=("__version__",),
)

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass

# Logging is not exposed by default and needs to be configured by the user.
logging.getLogger(__name__).addHandler(logging.NullHandler())
