"""α/β kernel formalism: LQ parameters read from per-tissue-class kernels."""

from typing import Any, ClassVar, Optional

import numpy as np

from pyRadPlan.biology.models._evaluator import BioModelEvaluator, PrecomputedAlphaBetaEvaluator
from pyRadPlan.biology.models._kernel_sets import KernelSetInfo, kernel_reference_classes
from pyRadPlan.biology.models._tissue_lookup import TISSUE_LOOKUPS, TissueParameterLookup
from pyRadPlan.biology.tissues import LEMTissueProperties, TissueProperties
from .lq_models import LQModel


def resolved_lookup(model: Any, machine: Any, voxel_params: dict[str, Any]) -> tuple:
    """
    Resolve a kernel-reading model for the patient's tissue classes and build its lookup.

    Every tissue class of the patient (``voxel_params["tissue_class"]``, addressed by the
    per-voxel ``tissue_class_ix``) resolves to one column of the selected kernel set, see
    :meth:`~pyRadPlan.biology.models.BiologicalModel.resolve`.

    Returns
    -------
    (TissueParameterLookup, KernelSetInfo)

    Raises
    ------
    ValueError
        The model cannot be resolved (no or ambiguous kernel set, inconsistent base data,
        unmatched tissues, missing tissue parameters) or the lookup is unknown.
    NotImplementedError
        The tissues resolve to several kernel sets.
    """
    try:
        lookup_cls: type[TissueParameterLookup] = TISSUE_LOOKUPS[model.tissue_lookup]
    except KeyError as exc:
        raise ValueError(
            f"Unknown tissue lookup '{model.tissue_lookup}'. Available: {sorted(TISSUE_LOOKUPS)}"
        ) from exc

    tissues = list(voxel_params.get("tissue_class", ()))
    resolution = model.resolve(machine, tissues)
    resolution.raise_for_errors()
    if len(resolution.kernel_set_ids) > 1:
        raise NotImplementedError(
            f"Biological model '{model.model}' resolves the structures to several kernel sets "
            f"{list(resolution.kernel_set_ids)}; a dose calculation currently evaluates one "
            "kernel set per formalism."
        )
    if resolution.kernel_set is None:
        raise ValueError(f"Biological model '{model.model}' resolved to no kernel set.")
    rows = [entry.column or 0 for entry in resolution.structures]
    return lookup_cls(tissues, range(len(tissues)), rows), resolution.kernel_set


class KernelBasedLQModel(LQModel):
    """
    α/β kernels: LQ parameters read from pre-computed per-tissue-class pencil-beam kernels.

    The machine carries depth-dependent α and β kernels per ion energy with one column per
    tissue class (a kernel set of kind ``alpha_beta``). The kernels may have been computed
    with any radiobiological model (their provenance, e.g. LEM I or mMKM); the tissue
    parameters of each structure must match one tissue class of the selected set.

    Attributes
    ----------
    model : str
        ``"alpha_beta_kernels"`` (aliases ``"kernel_based_lq"``, ``"LEM"``)
    required_quantities : tuple[str, ...]
        ``("physical_dose", "alpha", "beta")``
    kernel_quantities : tuple[str, ...]
        ``("alpha", "beta")`` — the kernel arrays gathered per tissue class.

    Parameters
    ----------
    tissue_lookup : str
        How a voxel's tissue properties select a kernel tissue class (``"exact"``).
    data_model : str, optional
        Provenance of the kernel set to use (e.g. ``"LEM_I"``, ``"mMKM"``), for machines
        carrying α/β kernels of several provenances.
    kernel_set : str, optional
        Explicit kernel set id, for machines with several sets of one provenance.
    """

    model = "alpha_beta_kernels"
    name = "α/β kernels"
    description = "LQ model with α and √β read from per-tissue α/β kernels of the machine."
    model_aliases: ClassVar[tuple[str, ...]] = ("kernel_based_lq", "LEM")
    matrad_name: ClassVar[str] = "LEM"
    required_quantities = ("physical_dose", "alpha", "beta")
    possible_radiation_modes = ("protons", "helium", "carbon", "oxygen")
    kernel_quantities = ("alpha", "beta")
    kernel_set_kind = "alpha_beta"
    _legacy_tissue_type: ClassVar[type[TissueProperties]] = LEMTissueProperties

    def __init__(
        self,
        tissue_lookup: str = "exact",
        data_model: Optional[str] = None,
        kernel_set: Optional[str] = None,
    ):
        self.tissue_lookup = tissue_lookup
        self.data_model = data_model
        self.kernel_set = kernel_set

    def evaluator(self, machine: Any, voxel_params: dict[str, Any]) -> BioModelEvaluator:
        """Create the evaluator gathering the α/β kernel columns of each voxel's tissue."""
        lookup, kernel_set = resolved_lookup(self, machine, voxel_params)
        return PrecomputedAlphaBetaEvaluator(
            self,
            self.kernel_quantities,
            lookup,
            voxel_params,
            field_map=dict(zip(self.kernel_quantities, kernel_set.fields)),
        )

    def kernel_tissue_classes(self, machine: Any) -> list[TissueProperties]:
        """
        Return the tissue classes of the selected α/β kernel set in kernel column order.

        Raises
        ------
        ValueError
            No α/β kernel set can be selected, or the selected set is inconsistent (e.g.
            its declared tissue classes do not correspond to the kernel columns).
        """
        kernel_set = self._selected_set(machine)
        ordered = sorted(
            (info for info in kernel_set.tissue_classes if info.column is not None),
            key=lambda info: info.column,
        )
        return [info.tissue for info in ordered]

    def _selected_set(self, machine: Any) -> KernelSetInfo:
        kernel_set = self.select_kernel_set(machine)
        if kernel_set.problems:
            raise ValueError("; ".join(kernel_set.problems))
        return kernel_set

    def reference_classes(self, machine: Any) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the ``(alpha_x, beta_x)`` tissue classes of the machine's α/β kernel columns.

        See :func:`~pyRadPlan.biology.models.kernel_reference_classes`.
        """
        return kernel_reference_classes(getattr(machine, "pb_kernels", None) or {})

    def alpha_beta_from_kernel_rows(self, rows: dict[str, Any]) -> tuple[Any, Any]:
        """Return the ``(alpha, beta)`` kernel quantities from the looked-up kernel rows."""
        return rows["alpha"], rows["beta"]
