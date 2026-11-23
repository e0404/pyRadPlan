"""Typed container of the objects of a workspace for saving and loading."""

import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional, Union

import SimpleITK as sitk

from pyRadPlan.cst import StructureSet
from pyRadPlan.ct import CT
from pyRadPlan.dij import Dij
from pyRadPlan.plan import Plan
from pyRadPlan.result import PlanResult
from pyRadPlan.stf import SteeringInformation


@dataclass
class ExportBundle:
    """The named objects of a workspace, ready for :func:`~pyRadPlan.io.save_data`.

    A dataclass rather than a pydantic model: the bundle only holds references, whereas a
    ``PyRadPlanBaseModel`` would validate and coerce every field (plans, steering information
    and results are accepted in several representations) and copy the SimpleITK images and
    dictionaries on assignment.

    Parameters
    ----------
    ct : CT, optional
        The CT.
    cst : StructureSet, optional
        The structure set.
    pln : Plan, optional
        The treatment plan.
    stf : SteeringInformation, optional
        The steering information.
    dij : Dij, optional
        The dose influence matrix.
    result : PlanResult, optional
        A plan result, e.g. of a dose calculation or optimization.
    weights : array_like, optional
        Fluence or spot weights.
    dose : sitk.Image, optional
        A dose distribution.
    extras : dict
        Further named objects. The names must not be those of the fields above.
    """

    ct: Optional[CT] = None
    cst: Optional[StructureSet] = None
    pln: Optional[Plan] = None
    stf: Optional[SteeringInformation] = None
    dij: Optional[Dij] = None
    result: Optional[PlanResult] = None
    weights: Optional[Any] = None
    dose: Optional[sitk.Image] = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        clash = sorted(set(self.extras) & set(self._field_names()))
        if clash:
            raise ValueError(f"extras must not use the names of bundle fields: {clash}.")

    @classmethod
    def _field_names(cls) -> list[str]:
        return [f.name for f in fields(cls) if f.name != "extras"]

    def to_dict(self) -> dict[str, Any]:
        """Return the objects that are set as a dictionary, extras included."""
        objects = {
            name: getattr(self, name)
            for name in self._field_names()
            if getattr(self, name) is not None
        }
        objects.update({k: v for k, v in self.extras.items() if v is not None})
        return objects

    def save(
        self,
        file_name: Union[str, os.PathLike],
        format: Optional[str] = None,  # noqa: A002 - same name as in save_data
        **exporter_options: Any,
    ) -> Union[str, list[str]]:
        """Save the bundle with :func:`~pyRadPlan.io.save_data`.

        Objects the format cannot store are dropped with a single warning.

        Parameters
        ----------
        file_name : str or os.PathLike
            Target file or, for DICOM, directory.
        format : str, optional
            Explicit format; by default it follows from the extension of ``file_name``.
        **exporter_options
            Options of the exporter, e.g. ``anonymize=True`` for DICOM.

        Returns
        -------
        str or list[str]
            The path(s) written.
        """
        from ._load_save import save_data  # noqa: PLC0415

        return save_data(
            self, file_name=file_name, format=format, exporter_options=exporter_options
        )

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "ExportBundle":
        """Load a bundle with :func:`~pyRadPlan.io.load_data`.

        Objects that are not fields of the bundle (e.g. ``saved_result_tags``) go to
        :attr:`extras`.

        Parameters
        ----------
        path : str or os.PathLike
            File or DICOM directory to load.

        Returns
        -------
        ExportBundle
            The loaded objects.
        """
        from ._load_save import load_data  # noqa: PLC0415

        data = load_data(path)
        names = cls._field_names()
        known = {k: v for k, v in data.items() if k in names}
        extras = {k: v for k, v in data.items() if k not in names}
        return cls(**known, extras=extras)
