"""z* kernel formalism: LQ parameters derived from specific-energy (z*) kernels."""

from abc import abstractmethod
from typing import Any, ClassVar, Optional

from pyRadPlan.biology.models._evaluator import BioModelEvaluator, DerivedAlphaBetaEvaluator
from pyRadPlan.biology.tissues import MKMTissueProperties, TissueProperties
from .kernel_based_lq_model import resolved_lookup
from .lq_models import LQModel


class ZStarBasedLQModel(LQModel):
    """
    LQ model computing alpha and beta from pre-computed specific energy (z*) kernels.

    The machine carries a z* kernel set (kind ``zs``) whose tissue classes are identified by
    the parameters of their provenance (e.g. the domain/nucleus geometry for MKM). Subclasses
    implement :meth:`alpha_beta`.

    Attributes
    ----------
    required_quantities : tuple[str, ...]
        ``("physical_dose", "zs")``; the model is only offered for machines whose pencil-beam
        kernels tabulate z*.
    kernel_quantities : tuple[str, ...]
        ``("zs",)``

    Parameters
    ----------
    tissue_lookup : str
        How a voxel's tissue properties select a kernel tissue class (``"exact"``).
    data_model : str, optional
        Provenance of the z* kernel set to use (e.g. ``"MKM"``, ``"mMKM"``).
    kernel_set : str, optional
        Explicit kernel set id, for machines with several sets of one provenance.
    quantity_variants : dict[str, str], optional
        Variant of the z* definition to use, e.g. ``{"zs": "z1D_saturation_corrected"}``.
    """

    required_quantities = ("physical_dose", "zs")
    possible_radiation_modes = ("protons", "helium", "carbon", "oxygen")
    kernel_quantities = ("zs",)
    kernel_set_kind = "zs"
    additional_fields = []

    def __init__(
        self,
        tissue_lookup: str = "exact",
        data_model: Optional[str] = None,
        kernel_set: Optional[str] = None,
        quantity_variants: Optional[dict[str, str]] = None,
    ):
        self.tissue_lookup = tissue_lookup
        self.data_model = data_model
        self.kernel_set = kernel_set
        self.quantity_variants = dict(quantity_variants) if quantity_variants else None

    def evaluator(self, machine: Any, voxel_params: dict[str, Any]) -> BioModelEvaluator:
        """Create the evaluator gathering the z* kernel rows of each voxel's tissue class."""
        self.validate_tissue_parameters(voxel_params.get("tissue_class"))
        lookup, kernel_set = resolved_lookup(self, machine, voxel_params)
        return DerivedAlphaBetaEvaluator(
            self,
            self.kernel_quantities,
            lookup,
            voxel_params,
            self.additional_fields,
            field_map=dict(zip(self.kernel_quantities, kernel_set.fields)),
        )

    def alpha_beta_from_kernel_rows(self, rows: dict[str, Any]) -> tuple[Any, Any]:
        """Return ``(alpha, beta)`` derived from the looked-up z* kernel rows."""
        return self.alpha_beta(rows)

    @abstractmethod
    def alpha_beta(self, rows: dict[str, Any]) -> tuple[Any, Any]:
        """Return (alpha, beta), each scalar or of shape ``(n_voxels,)``."""


class MKM(ZStarBasedLQModel):
    """
    z* kernels: ``alpha = alpha_0 + beta_0 * z*`` and ``beta = beta_0``.

    The z* kernels are selected by the tissue class of each structure (for MKM data the
    domain/nucleus geometry), while ``alpha_0`` / ``beta_0`` are read per voxel from the
    structure's tissue itself. A tissue leaving them unset is rejected before the dose
    calculation instead of evaluating to NaN.
    """

    model = "zstar_kernels"
    name = "z* kernels"
    description = "LQ model with α = α₀ + β₀·z* and β = β₀ from the machine's z* kernels."
    model_aliases: ClassVar[tuple[str, ...]] = ("MKM", "zs_based_lq")
    possible_radiation_modes = ("protons", "helium", "carbon", "oxygen")
    matrad_name = "MKM"
    required_tissue_parameters = ("alpha_0", "beta_0")
    _legacy_tissue_type: ClassVar[type[TissueProperties]] = MKMTissueProperties

    def __init__(
        self,
        tissue_lookup: str = "exact",
        data_model: Optional[str] = None,
        kernel_set: Optional[str] = None,
        quantity_variants: Optional[dict[str, str]] = None,
    ):
        super().__init__(tissue_lookup, data_model, kernel_set, quantity_variants)
        self.additional_fields = list(self.required_tissue_parameters)

    def alpha_beta(self, rows: dict[str, Any]) -> tuple[Any, Any]:
        """Return ``alpha = alpha_0 + beta_0 * zs`` and ``beta = beta_0``."""
        return rows["alpha_0"] + rows["beta_0"] * rows["zs"], rows["beta_0"]


#: Name of the z* kernel formalism by what it reads.
ZStarKernelModel = MKM
