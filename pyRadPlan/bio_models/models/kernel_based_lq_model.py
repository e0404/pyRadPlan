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
    ]  # Compatible with common ion modalities

    def calc_biological_quantities_for_bixel(self, bixel: dict, kernels: dict) -> dict:
        """
        Compute tissue-specific alpha and beta values for a bixel.
        """
        bixel = super().calc_biological_quantities_for_bixel(bixel, kernels)
        xp = array_api_compat.array_namespace(bixel["rad_depths"])
        num_tissue_classes = xp.unique_values(xp.asarray(bixel["v_tissue_index"])).shape[0]
        alpha = xp.zeros_like(bixel["rad_depths"])
        beta = xp.zeros_like(bixel["rad_depths"])
        for i in range(num_tissue_classes):
            mask = bixel["v_tissue_index"] == i
            alpha[mask] = xp.where(mask, kernels["alpha"][i, :], alpha[mask])
            beta[mask] = xp.where(mask, kernels["beta"][i, :], beta[mask])
        bixel["alpha"] = alpha
        bixel["beta"] = beta
        return bixel

    def get_tissue_information(
        self, machine: ParticleAccelerator, v_alpha_x: Any, v_beta_x: Any
    ) -> Any:
        """
        Build per-scenario tissue-index vectors.

        """
        xp = array_api_compat.array_namespace(v_alpha_x)
        num_of_ct_scen = v_alpha_x.shape[1]
        # Initialise output arrays (one per scenario)
        v_tissue_index = xp.zeros(v_alpha_x.shape)

        machine_pairs = xp.asarray(
            list(
                zip(
                    machine.pb_kernels[machine.energies[0]].alpha_x,
                    machine.pb_kernels[machine.energies[0]].beta_x,
                )
            )
        )
        flat_alpha = xp.reshape(v_alpha_x, (-1,))
        flat_beta = xp.reshape(v_beta_x, (-1,))
        unique_alpha_beta_pairs = set(
            (float(flat_alpha[i]), float(flat_beta[i])) for i in range(int(flat_alpha.shape[0]))
        )
        unique_alpha_beta_pairs.discard((0.0, 0.0))
        ix_tissue = []  # tissue index for each unique alpha-beta pair

        for i, (alpha_set, beta_set) in enumerate(unique_alpha_beta_pairs):
            cst_paris = xp.asarray([alpha_set, beta_set])
            matches = xp.all(machine_pairs == cst_paris, axis=1)
            idx = xp.nonzero(matches)[0]
            if idx.shape[0] != 1:
                raise ValueError(
                    f"No matching alpha-beta pair found in machine data for alpha={alpha_set}, beta={beta_set}"
                )
            ix_tissue.append(int(idx[0]))  # assign the first matching index (

        for i in ix_tissue:
            for s in range(num_of_ct_scen):
                alpha_ref = machine.pb_kernels[machine.energies[0]].alpha_x[i].item()
                beta_ref = machine.pb_kernels[machine.energies[0]].beta_x[i].item()
                mask = (v_alpha_x[:, s] == alpha_ref) & (v_beta_x[:, s] == beta_ref)
                col = v_tissue_index[:, s]
                v_tissue_index[:, s] = xp.where(mask, xp.asarray(i, dtype=col.dtype), col)
        return v_tissue_index
