"""Evaluators: per-dose-calculation state and evaluation of biological models."""

from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING, Any

import array_api_compat

from ._tissue_lookup import TissueParameterLookup

if TYPE_CHECKING:  # pragma: no cover
    from ._base import BiologicalModel


class BioModelEvaluator(ABC):
    """
    Evaluates one :class:`BiologicalModel` for one dose calculation.

    Created by :meth:`BiologicalModel.evaluator` and owned by the dose engine for the
    duration of a single dose calculation. Holds only derived, machine- and
    geometry-specific state; never reuse it across patients or machines.

    The dose engine interacts with a model exclusively through this interface.
    """

    def __init__(self, model: "BiologicalModel"):
        self.model = model

    def dij_scalars(self) -> dict[str, Any]:
        """Scalar entries to store on the dij (e.g. ``{"rbe": 1.1}``)."""
        return self.model.dij_scalars()

    def kernel_quantities(self, kernel: dict[str, Any]) -> dict[str, Any]:
        """
        Depth-dependent arrays the engine must interpolate per bixel for this model.

        Parameters
        ----------
        kernel : dict
            Pencil-beam kernel of the current bixel's energy (array-namespace dict).

        Returns
        -------
        dict[str, Array]
            Arrays of shape ``(n_depths,)`` or ``(n_classes, n_depths)`` keyed by name;
            they are handed back, interpolated, to :meth:`bixel_alpha_beta`.
        """
        return {}

    def bixel_alpha_beta(self, bixel: dict[str, Any], kernels: dict[str, Any]) -> tuple[Any, Any]:
        """
        Per-voxel LQ parameters (alpha, beta) for one bixel.

        Parameters
        ----------
        bixel : dict
            Bixel with at least ``v_alpha_x`` / ``v_beta_x`` (reference LQ parameters of the
            voxels hit) and ``rad_depths``.
        kernels : dict
            Kernel values interpolated at the bixel's radiological depths, including the
            arrays requested through :meth:`kernel_quantities`.
        """
        raise NotImplementedError(
            f"Biological model '{self.model.model}' does not provide alpha/beta values."
        )


class ParametricEvaluator(BioModelEvaluator):
    """
    Evaluator for models that are pure functions of the per-voxel parameters.

    Holds nothing but the model; forwards to :meth:`BiologicalModel.alpha_beta`.
    """

    def bixel_alpha_beta(self, bixel: dict[str, Any], kernels: dict[str, Any]) -> tuple[Any, Any]:
        if not self.model.provides_alpha_beta:
            return super().bixel_alpha_beta(bixel, kernels)
        return self.model.alpha_beta(bixel["v_alpha_x"], bixel["v_beta_x"], kernels)


class KernelBasedEvaluator(BioModelEvaluator):
    """
    Evaluator for models reading pre-tabulated alpha/beta kernels from the machine data.

    The machine kernels carry ``(n_classes, n_depths)`` arrays per tissue class; a
    :class:`TissueParameterLookup` decides which class row each voxel uses.

    Parameters
    ----------
    model : BiologicalModel
        Must implement ``alpha_beta_from_kernel_rows``.
    kernel_fields : list[str]
        Names of the machine kernel arrays to interpolate per bixel (e.g. ``["alpha", "beta"]``).
    lookup : TissueParameterLookup
        Voxel parameter → tissue class mapping.
    voxel_params : dict[str, Array]
        Per-voxel parameters of the whole dose grid; validated up front so that a
        structure without matching base data fails before the dose calculation starts.
    """

    def __init__(
        self,
        model: "BiologicalModel",
        kernel_fields: list[str],
        lookup: TissueParameterLookup,
        voxel_params: dict[str, Any],
    ):
        super().__init__(model)
        self.kernel_fields = list(kernel_fields)
        self.lookup = lookup
        self.lookup.validate(voxel_params)

    def kernel_quantities(self, kernel: dict[str, Any]) -> dict[str, Any]:
        return {name: kernel[name] for name in self.kernel_fields}

    def bixel_alpha_beta(self, bixel: dict[str, Any], kernels: dict[str, Any]) -> tuple[Any, Any]:
        rows = self.lookup.gather(bixel, kernels, self.kernel_fields)
        return self.model.alpha_beta_from_kernel_rows(rows)


class TabulatedSpectrumEvaluator(BioModelEvaluator):
    """
    Evaluator for models whose kernels are dose-averaged from tables over fluence spectra.

    On construction the model's lookup tables are dose-averaged over the fragment fluence
    spectra of every machine energy, producing ``(n_classes, n_depths)`` arrays per energy.
    The machine itself is left untouched. Per bixel, a :class:`TissueParameterLookup`
    selects the class row for each voxel.

    Parameters
    ----------
    model : TabulatedRBEModel
    machine : ParticleAccelerator
        Machine whose kernels carry fragment fluence spectra.
    lookup : TissueParameterLookup
    voxel_params : dict[str, Array]
    """

    def __init__(
        self,
        model: "BiologicalModel",
        machine: Any,
        lookup: TissueParameterLookup,
        voxel_params: dict[str, Any],
    ):
        super().__init__(model)
        self.kernel_fields = list(model.quantities_in_kernel)
        self.lookup = lookup
        self.lookup.validate(voxel_params)

        first_kernel = machine.pb_kernels[machine.energies[0]]
        fragments = model.select_fragments(first_kernel.fluence_spectrum)
        self._tables = {
            float(energy): model.dose_average(kernel, fragments)
            for energy, kernel in machine.pb_kernels.items()
        }
        self._converted: dict[tuple, dict[str, Any]] = {}

    def kernel_quantities(self, kernel: dict[str, Any]) -> dict[str, Any]:
        xp = array_api_compat.array_namespace(kernel["depths"])
        device = array_api_compat.device(kernel["depths"])
        key = (float(kernel["energy"]), xp, device)
        if key not in self._converted:
            self._converted[key] = {
                name: xp.asarray(arr, device=device) for name, arr in self._tables[key[0]].items()
            }
        return self._converted[key]

    def bixel_alpha_beta(self, bixel: dict[str, Any], kernels: dict[str, Any]) -> tuple[Any, Any]:
        rows = self.lookup.gather(bixel, kernels, self.kernel_fields)
        return self.model.alpha_beta_from_kernel_rows(rows)
