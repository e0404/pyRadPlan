"""Abstract base class for all pyRadPlan importers."""

import os
from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Union

import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet


class BaseImporter(ABC):
    """
    Abstract interface for all importers.

    An importer is bound to a single source (a file or a folder) and provides
    methods to load individual pyRadPlan objects (CT, StructureSet, dose) as well
    as bulk loaders (``load_patient`` and ``load_data``).

    Subclasses must implement :meth:`load_ct` and :meth:`load_cst`. ``load_dose``
    is optional and returns ``None`` by default.

    Attributes
    ----------
    format : str
        Short format key used to register and look up the importer (e.g. ``"mat"``).
    name : str
        Human-readable name of the importer.
    extensions : tuple of str
        File extensions handled by this importer (lower case, with dot).
    """

    format: ClassVar[Optional[str]] = None
    name: ClassVar[str] = "Base Importer"
    extensions: ClassVar[tuple[str, ...]] = ()

    def __init__(self, path: Union[str, os.PathLike]):
        self.path = os.fspath(path)

    @classmethod
    def handles_directory(cls, path: Union[str, os.PathLike]) -> bool:
        """Return True if this importer recognizes the given directory.

        Used by ``detect_format`` to resolve directory inputs. Subclasses that
        import from folders (e.g. DICOM) override this. Default is ``False``.
        """
        return False

    def _require_path(self) -> str:
        """Return ``self.path``, raising if it does not exist."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Path not found: {self.path}")
        return self.path

    @abstractmethod
    def load_ct(self) -> CT:
        """Load and return the CT object."""

    @abstractmethod
    def load_cst(self, ct: Optional[CT] = None) -> StructureSet:
        """
        Load and return the StructureSet.

        Parameters
        ----------
        ct : CT, optional
            The reference CT. If not provided, the importer loads it itself.
        """

    def load_dose(self) -> Optional[sitk.Image]:
        """Load and return a dose distribution as a SimpleITK image, if available."""
        return None

    def load_patient(self) -> tuple[CT, StructureSet]:
        """
        Load CT and StructureSet.

        Returns
        -------
        tuple[CT, StructureSet]
            The CT and StructureSet objects.
        """
        ct = self.load_ct()
        cst = self.load_cst(ct)
        return ct, cst

    def load_data(self) -> dict:
        """
        Load everything available from the source into a dictionary.

        Returns
        -------
        dict
            A dictionary that may contain the keys ``"ct"``, ``"cst"`` and
            ``"dose"``. Missing pieces are simply omitted.
        """
        data = {}

        ct = self.load_ct()
        if ct is not None:
            data["ct"] = ct

        cst = self.load_cst(ct)
        if cst is not None:
            data["cst"] = cst

        dose = self.load_dose()
        if dose is not None:
            data["dose"] = dose

        return data
