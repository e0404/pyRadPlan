"""Kernel-based Linear-Quadratic (LQ) model."""

from typing import Any, ClassVar

from pyRadPlan.bio_models._evaluator import BioModelEvaluator, KernelBasedEvaluator
from pyRadPlan.bio_models._tissue_lookup import make_tissue_lookup
from .lq_models import LQModel


class KernelBasedLQModel(LQModel):
    """
    LQ model that reads alpha and beta from pre-computed pencil-beam kernels.

    The machine data carries depth-dependent alpha/beta kernels per ion energy and tissue
    class, where tissue classes are identified by their reference ``(alpha_x, beta_x)``
    pair. This is the standard workflow for heavy-ion therapy (LEM-style base data).

    Class Attributes
    ----------------
    model : str
        ``"kernel_based_lq"`` (alias ``"LEM"``)
    required_quantities : list[str]
        ``["physical_dose", "alpha", "beta"]``
    kernel_quantities : list[str]
        ``["alpha", "beta"]`` — the kernel arrays gathered per tissue class.

    Parameters
    ----------
    tissue_lookup : str
        How voxel ``(alpha_x, beta_x)`` pairs select a kernel tissue class (``"exact"``).
    """

    model = "kernel_based_lq"
    model_aliases: ClassVar[list[str]] = ["LEM"]
    required_quantities = ["physical_dose", "alpha", "beta"]
    possible_radiation_modes = ["protons", "helium", "carbon", "oxygen"]
    kernel_quantities = ["alpha", "beta"]

    def __init__(self, tissue_lookup: str = "exact"):
        self.tissue_lookup = tissue_lookup

    def evaluator(self, machine: Any, voxel_params: dict[str, Any]) -> BioModelEvaluator:
        kernel = machine.pb_kernels[machine.energies[0]]
        lookup = make_tissue_lookup(self.tissue_lookup, kernel.alpha_x, kernel.beta_x)
        return KernelBasedEvaluator(self, self.kernel_quantities, lookup, voxel_params)

    def alpha_beta_from_kernel_rows(self, rows: dict[str, Any]) -> tuple[Any, Any]:
        return rows["alpha"], rows["beta"]
