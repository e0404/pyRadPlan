from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional

from ._evaluator import BioModelEvaluator, ParametricEvaluator

logger = logging.getLogger(__name__)


class BiologicalModel(ABC):
    """
    Abstract base class for biological models.

    A biological model is a lightweight, patient-agnostic bundle of parameters and pure
    functions. Everything that depends on a particular machine or patient geometry (tissue
    class lookup, pre-computed kernel tables, ...) lives in the
    :class:`~pyRadPlan.bio_models.BioModelEvaluator` created by :meth:`evaluator` for one
    dose calculation.

    Class Attributes
    ----------------
    model : str
        Canonical name identifying the biological model (e.g. ``"none"``, ``"LEM"``).
    model_aliases : list[str]
        Alternative names by which this model can be looked up.
    required_quantities : list[str]
        Kernel quantities the machine data must provide (e.g. ``["physical_dose", "let"]``).
    possible_radiation_modes : list[str]
        Radiation modalities this model supports.
    default_report_quantity : str
        Quantity recommended for display and planning by default.
    provides_alpha_beta : bool
        Whether the model yields per-voxel LQ parameters; if so, the dose engine computes
        ``alpha_dose`` / ``sqrt_beta_dose`` influence matrices.
    requires_let : bool
        Whether the dose engine has to provide LET kernels to evaluate the model.
    """

    model: ClassVar[str]
    model_aliases: ClassVar[list[str]] = []
    required_quantities: ClassVar[list[str]] = []
    possible_radiation_modes: ClassVar[list[str]]
    default_report_quantity: ClassVar[str] = "physical_dose"
    provides_alpha_beta: ClassVar[bool] = False
    requires_let: ClassVar[bool] = False

    @abstractmethod
    def evaluator(self, machine: Any, voxel_params: dict[str, Any]) -> BioModelEvaluator:
        """
        Create the evaluator of this model for one machine / patient geometry.

        Parameters
        ----------
        machine
            The (validated) machine the dose is calculated with.
        voxel_params : dict[str, Array]
            Per-voxel tissue parameters on the dose grid, each of shape
            ``(num_voxels, num_ct_scenarios)``. Currently ``"alpha_x"`` and ``"beta_x"``.
        """

    def dij_scalars(self) -> dict[str, Any]:
        """Scalar entries this model contributes to the dij (e.g. a constant RBE)."""
        return {}

    def is_available(
        self,
        radiation_mode: str,
        provided_quantities: Optional[list[str]] = None,
    ) -> tuple[bool, str]:
        """
        Check compatibility with a radiation mode and the quantities a machine provides.

        Returns
        -------
        (available, message)
            ``available`` is ``True`` only when both the radiation mode and all required
            quantities are satisfied.
        """
        messages: list[str] = []

        valid_rad_mode = radiation_mode in self.possible_radiation_modes
        if not valid_rad_mode:
            messages.append(
                f"Radiation mode '{radiation_mode}' is invalid for model '{self.model}'."
            )

        valid_quantities = True
        if provided_quantities is not None and self.required_quantities:
            missing = [q for q in self.required_quantities if q not in provided_quantities]
            if missing:
                valid_quantities = False
                messages.append(
                    "Required quantities not provided by Dose Engine or selected Machine Dataset."
                )

        return (valid_rad_mode and valid_quantities), ", ".join(messages)

    def validate_for(
        self,
        radiation_mode: str,
        provided_quantities: Optional[list[str]] = None,
    ) -> None:
        """Raise ValueError if the model is not available for the given mode/quantities."""
        ok, msg = self.is_available(radiation_mode, provided_quantities)
        if not ok:
            raise ValueError(f"Biological model '{self.model}' not valid: {msg}")


# Backwards-compatible name
BiologicalModelBase = BiologicalModel


class EmptyModel(BiologicalModel):
    """
    Passthrough biological model that applies no biological weighting.

    Compatible with all modalities, requires no kernel data beyond physical dose, and
    recommends reporting physical dose.
    """

    model = "none"
    required_quantities = []
    possible_radiation_modes = ["photons", "protons", "helium", "carbon", "oxygen", "VHEE"]
    default_report_quantity = "physical_dose"

    def evaluator(self, machine: Any, voxel_params: dict[str, Any]) -> BioModelEvaluator:
        return ParametricEvaluator(self)
