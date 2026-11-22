"""Abstract base class for all pyRadPlan exporters."""

import os
from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Union

import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.result import PlanResult


class BaseExporter(ABC):
    """
    Abstract interface for all exporters.

    An exporter is bound to a single target (a file or a folder) and writes the
    provided pyRadPlan objects to that target.

    Attributes
    ----------
    format : str
        Short format key used to register and look up the exporter (e.g. ``"mat"``).
    name : str
        Human-readable name of the exporter.
    extensions : tuple of str
        File extensions handled by this exporter (lower case, with dot).
    container : bool
        True if the format stores multiple objects in a single file (e.g. ``.mat``);
        False for directory-based formats (e.g. DICOM).
    """

    format: ClassVar[Optional[str]] = None
    name: ClassVar[str] = "Base Exporter"
    extensions: ClassVar[tuple[str, ...]] = ()
    container: ClassVar[bool] = True

    def __init__(self, path: Union[str, os.PathLike]):
        self.path = os.fspath(path)

    @staticmethod
    def resolve_dose(
        dose: Optional[sitk.Image] = None, result: Optional[PlanResult] = None
    ) -> Optional[sitk.Image]:
        """Return the dose to write, falling back to the physical dose of *result*.

        Used by formats that store a single dose image rather than a full result,
        so that passing ``result=`` does not silently write nothing.
        """
        if dose is not None:
            return dose
        if result is not None and "physical_dose" in result:
            return result["physical_dose"].values
        return None

    @abstractmethod
    def save(
        self,
        *,
        ct: Optional[CT] = None,
        cst: Optional[StructureSet] = None,
        dose: Optional[sitk.Image] = None,
        result: Optional[PlanResult] = None,
        **extra,
    ) -> None:
        """
        Write the provided objects to the target.

        Parameters
        ----------
        ct : CT, optional
            The CT to export.
        cst : StructureSet, optional
            The StructureSet to export.
        dose : sitk.Image, optional
            A dose distribution to export.
        result : PlanResult, optional
            A plan result to export. Exporters that cannot store a full result
            ignore it (a bare dose can be passed via *dose* instead).
        **extra
            Additional named objects an exporter may understand.
        """
