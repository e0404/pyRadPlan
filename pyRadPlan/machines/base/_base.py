from typing import Any, Optional, Annotated
from datetime import datetime
from pydantic import (
    Field,
    StringConstraints,
    field_validator,
)

from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.biology.tissues._base import TissueProperties, default_reference_parameters
from numpydantic import NDArray, Shape


class Machine(PyRadPlanBaseModel):
    """Base class for Machine objects.

    Defines minimum meta-data a machine must hold:

    Attributes
    ----------
    radiation_mode : str
        The radiation mode of the machine.
    description : str
        The description of the machine.
    machine : str
        The name of the machine.
    tissue_properties : Optional[NDArray[Shape["1-*"], TissueProperties]]
        Tissue properties associated with the RBE model data in the machine. Defaults to one class
        of the generic reference parameters (alpha_x=0.1, beta_x=0.05).
    """

    version: int = Field(default=1)
    radiation_mode: str = Field()
    description: str = Field(default="")
    name: Annotated[str, StringConstraints(min_length=1)] = Field(
        alias="machine", default="Generic"
    )
    created_on: Optional[datetime] = Field(default=None)
    last_modified: Optional[datetime] = Field(default=None)
    created_by: Optional[str] = Field(default="")
    last_modified_by: Optional[str] = Field(default="")
    data_type: Optional[str] = Field(default="-")

    tissue_properties: Optional[NDArray[Shape["1-*"], TissueProperties]] = Field(
        [TissueProperties.from_base_data(default_reference_parameters())],
        description="Tissue properties associated with the machine",
    )
    tissue_class_refs: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Kernel set ('kernel_set') and zero-based column ('column') of each tissue "
        "class, aligned with tissue_properties (from tissueParameters ref_Kernel / column).",
    )
    kernel_set_declarations: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Kernel sets declared in the base-data metadata (meta.kernelSets), "
        "normalized by parse_kernel_set_declarations.",
    )

    # Abstract property handled by the individual machines:
    _possible_radiation_modes: list[str]

    @field_validator("created_on", "last_modified", mode="before")
    @classmethod
    def validate_datetime_variants(cls, v):
        # If it is a string, we try some additional formats in addition to
        # pydantics accepted datetime values
        # For example, matRad macines use the format "%d-%b-%Y" for some dates
        if isinstance(v, str):
            try_formats = ["%d-%b-%Y"]

            for fmt in try_formats:
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    pass

        return v

    @property
    def kernel_sets(self) -> dict[str, Any]:
        """
        Kernel sets of the machine's biological base data, keyed by id.

        Built from the declared ``meta.kernelSets`` and ``meta.tissueParameters``, or
        inferred for legacy data (one set per kind present in the kernels), see
        :class:`~pyRadPlan.biology.models.KernelSetInfo`. Machines without pencil-beam kernels
        have none.
        """
        from pyRadPlan.biology.models._kernel_sets import (  # noqa: PLC0415 - avoids an import cycle
            build_kernel_sets,
        )

        tissues = self.tissue_properties
        return build_kernel_sets(
            declarations=self.kernel_set_declarations or (),
            tissue_classes=[] if tissues is None else list(tissues),
            tissue_refs=self.tissue_class_refs,
            kernels=getattr(self, "pb_kernels", None) or {},
            classes_declared="tissue_properties" in self.model_fields_set,
        )

    def provided_quantities(self) -> list[str]:
        """
        Return the quantities provided by this machine.

        Besides ``physical_dose``, the quantities of every kernel set kind present
        (``alpha``/``beta``, ``zs``, ``let``) and ``fluence`` for fragment fluence spectra.
        Where several kernel sets of one kind exist, each variant is reported as well, as
        ``"<quantity>:<variant>"`` (e.g. ``"let:LETt"``).
        """
        from pyRadPlan.biology.models._kernel_sets import (  # noqa: PLC0415 - avoids an import cycle
            KERNEL_SET_QUANTITIES,
            kernel_set_quantities,
        )

        quantities = ["physical_dose"]  # All machines provide physical dose
        sets = self.kernel_sets if getattr(self, "pb_kernels", None) else {}
        from_sets = kernel_set_quantities({k: s for k, s in sets.items() if s.complete})
        if hasattr(self, "has_let_kernel") and self.has_let_kernel or "let" in from_sets:
            quantities.append("let")
        if (
            hasattr(self, "has_alpha_beta_kernels")
            and self.has_alpha_beta_kernels
            or "alpha" in from_sets
        ):
            quantities.append("alpha")
            quantities.append("beta")
        if hasattr(self, "has_fluence_spectrum") and self.has_fluence_spectrum:
            quantities.append("fluence")
        if hasattr(self, "has_zstar_kernel") and self.has_zstar_kernel or "zs" in from_sets:
            quantities.append("zs")
        base = {name for names in KERNEL_SET_QUANTITIES.values() for name in names}
        quantities.extend(q for q in from_sets if q not in base and q not in quantities)
        return quantities
