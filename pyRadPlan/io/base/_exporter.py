"""Abstract base class for all pyRadPlan exporters."""

import os
from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Union

import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet


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

    @abstractmethod
    def save(
        self,
        *,
        ct: Optional[CT] = None,
        cst: Optional[StructureSet] = None,
        dose: Optional[sitk.Image] = None,
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
        **extra
            Additional named objects an exporter may understand.
        """
