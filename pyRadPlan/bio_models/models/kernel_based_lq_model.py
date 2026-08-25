"""Kernel-based Linear-Quadratic (LQ) model."""

import array_api_compat

from pyRadPlan.machines.particles._base import ParticleAccelerator
from .lq_models import LQModel
from typing import ClassVar, Any


class KernelBasedLQModel(LQModel):
    """
    LQ model that derives alpha and beta from pre-computed pencil-beam kernels.

    Rather than computing alpha and beta analytically, this model looks them up
    from tabulated kernel data stored in the machine dataset, indexed by tissue
    class. Tissue classes are identified by matching each voxel's
    (alpha_x, beta_x) reference pair against the kernel table, so the number
    of distinct tissue classes is driven entirely by the machine data and the
    structure set.

    This is the standard implementation for heavy-ion therapy (LEM-style
    workflows) where separate alpha/beta kernels are pre-computed per ion
    energy and tissue type.

    Class Attributes
    ----------------
    model : str
        ``"kernel_based_lq"``
    model_aliases : list[str]
        ``["LEM"]``
    required_quantities : list[str]
        ``["physical_dose", "alpha", "beta"]``
    kernel_quantities : list[str]
        ``["alpha", "beta"]`` — the kernel arrays that must be present in the
        kernels dict passed to :meth:`calc_biological_quantities_for_bixel`.
    possible_radiation_modes : list[str]
        ``["protons", "helium", "carbon"]``
    default_report_quantity : str
        ``"rbe_x_dose"``
    """

    required_quantities = ["physical_dose", "alpha", "beta"]  # Requires physical dose
    default_report_quantity = "rbe_x_dose"  # Suggested quantity for display and planning
    kernel_quantities = [
        "alpha",
        "beta",
    ]  # requires alpha and beta kernels to compute the biological effect
    model = "kernel_based_lq"
    model_aliases: ClassVar[list[str]] = ["LEM"]
    possible_radiation_modes = [
        "protons",
        "helium",
        "carbon",
        "oxygen",
    ]  # Compatible with common ion modalities

    def calc_biological_quantities_for_bixel(self, bixel: dict, kernels: dict) -> dict:
        """
        Compute tissue-specific alpha and beta values for a bixel.
        """
        bixel = super().calc_biological_quantities_for_bixel(bixel, kernels)
        xp = array_api_compat.array_namespace(bixel["rad_depths"])
        # kernels["alpha"/"beta"] have shape (n_tissue_classes, n_voxels)
        tissue_ix = xp.astype(xp.asarray(bixel["v_tissue_index"]), xp.int64)
        voxel_ix = xp.arange(tissue_ix.shape[0])
        bixel["alpha"] = kernels["alpha"][tissue_ix, voxel_ix]
        bixel["beta"] = kernels["beta"][tissue_ix, voxel_ix]
        return bixel

    def get_tissue_information(
        self, machine: ParticleAccelerator, v_alpha_x: Any, v_beta_x: Any
    ) -> Any:
        """
        Build per-scenario tissue-index vectors.

        """
        kernel = machine.pb_kernels[machine.energies[0]]
        return self.match_tissue_classes(v_alpha_x, v_beta_x, kernel.alpha_x, kernel.beta_x)
