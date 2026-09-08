"""Kernel-based Linear-Quadratic (LQ) model."""

from typing import Any, ClassVar

import numpy as np

from pyRadPlan.bio_models._evaluator import BioModelEvaluator, KernelBasedEvaluator
from pyRadPlan.bio_models._tissue_lookup import make_tissue_lookup
from .lq_models import LQModel


class KernelBasedLQModel(LQModel):
    """
    LQ model that reads alpha and beta from pre-computed pencil-beam kernels.

    The machine data carries depth-dependent alpha/beta kernels per ion energy and tissue
    class, where tissue classes are identified by their reference ``(alpha_x, beta_x)``
    pair. This is the standard workflow for heavy-ion therapy (LEM-style base data).

    Attributes
    ----------
    model : str
        ``"kernel_based_lq"`` (alias ``"LEM"``)
    required_quantities : tuple[str, ...]
        ``("physical_dose", "alpha", "beta")``
    kernel_quantities : tuple[str, ...]
        ``("alpha", "beta")`` — the kernel arrays gathered per tissue class.

    Parameters
    ----------
    tissue_lookup : str
        How voxel ``(alpha_x, beta_x)`` pairs select a kernel tissue class (``"exact"``).
    """

    model = "kernel_based_lq"
    model_aliases: ClassVar[tuple[str, ...]] = ("LEM",)
    matrad_name: ClassVar[str] = "LEM"
    required_quantities = ("physical_dose", "alpha", "beta")
    possible_radiation_modes = ("protons", "helium", "carbon", "oxygen")
    kernel_quantities = ("alpha", "beta")

    def __init__(self, tissue_lookup: str = "exact"):
        self.tissue_lookup = tissue_lookup

    def evaluator(self, machine: Any, voxel_params: dict[str, Any]) -> BioModelEvaluator:
        """Create the evaluator of this model for this machine."""
        class_alpha_x, class_beta_x = self.reference_classes(machine)
        lookup = make_tissue_lookup(self.tissue_lookup, class_alpha_x, class_beta_x)
        return KernelBasedEvaluator(self, self.kernel_quantities, lookup, voxel_params)

    def reference_classes(self, machine: Any) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the ``(alpha_x, beta_x)`` tissue classes of the machine's alpha/beta kernels.

        The per-bixel alpha/beta kernels are ``(n_classes, n_depths)`` arrays whose rows are
        addressed by a single class index computed once for the whole dose grid. That index is
        only meaningful when every energy tabulates the same tissue classes in the same order,
        so a differing energy is rejected here rather than silently selecting a different
        tissue's alpha/beta at that energy.

        Parameters
        ----------
        machine : ParticleAccelerator
            The machine the dose is calculated with.

        Returns
        -------
        (ndarray, ndarray)
            Reference ``alpha_x`` and ``beta_x`` per tissue class, shape ``(n_classes,)``.

        Raises
        ------
        ValueError
            The machine carries no pencil-beam kernels, an energy is missing its tissue class
            metadata, or the classes differ between energies.
        """
        kernels = getattr(machine, "pb_kernels", None)
        if not kernels:
            raise ValueError(
                f"Biological model '{self.model}' needs pencil-beam kernels with per-tissue-class "
                "alpha / beta data, but the machine carries none."
            )

        reference: tuple[Any, np.ndarray, np.ndarray] = None
        for energy, kernel in kernels.items():
            alpha_x, beta_x = getattr(kernel, "alpha_x", None), getattr(kernel, "beta_x", None)
            if alpha_x is None or beta_x is None:
                raise ValueError(
                    f"Biological model '{self.model}' needs the reference photon alpha_x / beta_x "
                    f"of every kernel tissue class, but the kernel of energy {energy} declares "
                    "none."
                )
            alpha_x = np.reshape(np.asarray(alpha_x, dtype=float), (-1,))
            beta_x = np.reshape(np.asarray(beta_x, dtype=float), (-1,))
            if alpha_x.shape != beta_x.shape:
                raise ValueError(
                    f"Kernel of energy {energy} declares {alpha_x.size} reference alpha_x but "
                    f"{beta_x.size} reference beta_x values."
                )

            if reference is None:
                reference = (energy, alpha_x, beta_x)
                continue

            ref_energy, ref_alpha_x, ref_beta_x = reference
            if not (np.array_equal(alpha_x, ref_alpha_x) and np.array_equal(beta_x, ref_beta_x)):
                raise ValueError(
                    f"Biological model '{self.model}' requires identical tissue classes in the "
                    f"same order for every machine energy. Energy {energy} declares "
                    f"alpha_x={alpha_x.tolist()}, beta_x={beta_x.tolist()}, while energy "
                    f"{ref_energy} declares alpha_x={ref_alpha_x.tolist()}, "
                    f"beta_x={ref_beta_x.tolist()}."
                )

        return reference[1], reference[2]

    def alpha_beta_from_kernel_rows(self, rows: dict[str, Any]) -> tuple[Any, Any]:
        """Return the ``(alpha, beta)`` kernel quantities from the looked-up kernel rows."""
        return rows["alpha"], rows["beta"]
