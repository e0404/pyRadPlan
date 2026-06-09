from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import ClassVar, Optional, List


logger = logging.getLogger(__name__)


class BiologicalModelBase(ABC):
    """
    Abstract base class for biological models used in dose calculation and
    plan optimisation.

    """

    model: ClassVar[str]
    model_aliases: ClassVar[list[str]] = []  # default: no aliases
    required_quantities: ClassVar[
        list[str]
    ]  # kernels in base data needed for the alpha/beta calculation
    possible_radiation_modes: ClassVar[
        list[str]
    ]  # radiation modalitites compatible with the model
    default_report_quantity: ClassVar[
        str
    ]  # default suggested quantity to use for display and planning

    @abstractmethod
    def calc_biological_quantities_for_bixel(self, bixel: dict, kernels: dict) -> dict:
        raise NotImplementedError(
            "Method '_calc_biological_quantities_for_bixel' must be implemented."
        )

    def is_available(
        self,
        radiation_mode: str,
        provided_quantities: Optional[list[str]] = None,
    ) -> tuple[bool, str]:
        """
        Check whether this model is compatible with the given radiation mode
        and the quantities supplied by the dose engine / machine dataset.

        Parameters
        ----------
        radiation_mode:
            Radiation modality string (e.g. ``"photons"``, ``"protons"``).
        provided_quantities:
            List of quantity names available from the dose engine or machine.

        Returns
        -------
        (available, message)
            ``available`` is ``True`` only when both the radiation mode and
            all required quantities are satisfied.
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
        provided_quantities: Optional[List[str]] = None,
    ) -> None:
        """Raise ValueError if the model is not available for the given mode/quantities."""
        ok, msg = self.is_available(radiation_mode, provided_quantities)
        if not ok:
            raise ValueError(f"Biological model '{self.model}' not valid: {msg}")


class EmptyModel(BiologicalModelBase):
    """
    A model that implements the None model

    """

    model = "none"
    required_quantities = []  # Requires physical dose information
    possible_radiation_modes = [
        "photons",
        "protons",
        "helium",
        "carbon",
        "VHEE",
    ]  # Compatible with all common modalities
    default_report_quantity = "physical_dose"  # Suggested quantity for display and planning

    def calc_biological_quantities_for_bixel(self, bixel: dict, kernels: dict) -> dict:
        """No biological weighting — returns the bixel unchanged."""
        return bixel
