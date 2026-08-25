"""Constant relative biological effectiveness (RBE) model."""

from typing import Any

from pyRadPlan.bio_models._base import BiologicalModel
from pyRadPlan.bio_models._evaluator import BioModelEvaluator, ParametricEvaluator


class ConstantRBEModel(BiologicalModel):
    """
    Biological model applying a single, spatially uniform RBE.

    RBE-weighted dose is ``rbe * physical_dose``; no LQ parameters are produced. The
    constant is stored on the dij (``dij.rbe``) so that quantities and results can be
    derived from the physical dose influence matrix alone.

    Parameters
    ----------
    rbe : float, optional
        Constant RBE factor. Defaults to ``1.1``, the clinically adopted proton value.
    """

    model = "constant_rbe"
    required_quantities = ["physical_dose"]
    possible_radiation_modes = ["photons", "protons", "helium", "carbon", "oxygen", "VHEE"]
    default_report_quantity = "rbe_x_dose"

    def __init__(self, rbe: float = 1.1):
        if rbe <= 0:
            raise ValueError("Constant RBE must be positive.")
        self.rbe = float(rbe)

    def dij_scalars(self) -> dict[str, Any]:
        return {"rbe": self.rbe}

    def evaluator(self, machine: Any, voxel_params: dict[str, Any]) -> BioModelEvaluator:
        return ParametricEvaluator(self)
