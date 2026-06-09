import array_api_strict as xp
import numpy as np

from pyRadPlan.machines.particles._base import ParticleAccelerator
from .lq_models import LQModel
from typing import ClassVar, Any


class KernelBasedLQModel(LQModel):
    """ """

    required_quantities = ["physical_dose"]  # Requires physical dose
    default_report_quantity = "RBExDose"  # Suggested quantity for display and planning
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
        bixel = super().calc_biological_quantities_for_bixel(bixel, kernels)
        num_tissue_classes = xp.unique_values(xp.asarray(bixel["v_tissue_index"])).shape[0]
        for i in range(num_tissue_classes):
            mask = xp.asarray(bixel["v_tissue_index"] == i)
            bixel["alpha"] = xp.reshape(
                xp.asarray(kernels["alpha"][i], dtype=bixel["alpha"].dtype), (-1,)
            )
            bixel["beta"] = xp.reshape(
                xp.asarray(kernels["beta"][i], dtype=bixel["beta"].dtype), (-1,)
            )
        return bixel

    def get_tissue_information(
        self, machine: ParticleAccelerator, v_alpha_x: Any, v_beta_x: Any
    ) -> Any:
        """
        Build per-scenario tissue-index vectors.

        """
        num_of_ct_scen = v_alpha_x.shape[1]

        # Initialise output arrays (one per scenario)
        v_tissue_index = np.zeros(v_alpha_x.shape)

        machine_pairs = np.array(
            list(
                zip(
                    machine.pb_kernels[machine.energies[0]].alpha_x,
                    machine.pb_kernels[machine.energies[0]].beta_x,
                )
            )
        )
        unique_alpha_beta_pairs = set(zip(v_alpha_x.flat, v_beta_x.flat))
        unique_alpha_beta_pairs.discard((0.0, 0.0))
        ix_tissue = []  # tissue index for each unique alpha-beta pair

        for i, (alpha_set, beta_set) in enumerate(unique_alpha_beta_pairs):
            cst_paris = np.array([alpha_set, beta_set])
            matches = np.all(machine_pairs == cst_paris, axis=1)
            idx = np.nonzero(matches)[0]
            if idx.shape[0] != 1:
                raise ValueError(
                    f"No matching alpha-beta pair found in machine data for alpha={alpha_set}, beta={beta_set}"
                )
            ix_tissue.append(int(idx[0]))  # assign the first matching index (

        for i in ix_tissue:
            for s in range(num_of_ct_scen):
                mask = (v_alpha_x[:, s] == machine.pb_kernels[machine.energies[0]].alpha_x[i]) & (
                    v_beta_x[:, s] == machine.pb_kernels[machine.energies[0]].beta_x[i]
                )
                v_tissue_index[mask, s] = i

        return v_tissue_index
