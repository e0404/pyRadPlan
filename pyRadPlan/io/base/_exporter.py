"""Abstract base class for all pyRadPlan exporters."""

import os
from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Union

import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet

#: Entry of :attr:`BaseExporter.supports` for exporters that store any named object.
ANY_OBJECT = "*"


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
    supports : frozenset of str
        Names of the objects :meth:`save` writes (``"ct"``, ``"cst"``, ``"dose"``, ``"pln"``,
        ``"stf"``, ``"dij"``, ``"result"``, ``"weights"``, ...), or :data:`ANY_OBJECT` (the
        default) for exporters that receive any named object. :func:`pyRadPlan.io.save_data`
        passes only the declared objects and warns about the others.
    """

    format: ClassVar[Optional[str]] = None
    name: ClassVar[str] = "Base Exporter"
    extensions: ClassVar[tuple[str, ...]] = ()
    container: ClassVar[bool] = True
    supports: ClassVar[frozenset[str]] = frozenset({ANY_OBJECT})

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
