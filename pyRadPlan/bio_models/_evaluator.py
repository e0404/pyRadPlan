"""Evaluators: per-dose-calculation state and evaluation of biological models."""

from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING, Any

import array_api_compat

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


class TissueClassEvaluator(BioModelEvaluator):
    """
    Evaluator for models with pre-computed data per discrete tissue class.

    The model's data (machine kernels or tables) exists for a finite set of reference
    ``(alpha_x, beta_x)`` pairs. Voxels are mapped onto those classes; the class-specific
    kernel row is gathered per voxel and converted by the model into (alpha, beta).

    Parameters
    ----------
    model : BiologicalModel
        Must implement ``match_tissue_classes`` and ``alpha_beta_from_kernel_rows``.
    class_alpha_x, class_beta_x : array-like, shape (n_classes,)
        Reference LQ parameters of the available tissue classes.
    voxel_params : dict[str, Array]
        Per-voxel parameters of the whole dose grid; validated up front so that a
        structure without matching base data fails before the dose calculation starts.
    kernel_fields : list[str]
        Names of the ``(n_classes, n_depths)`` kernel arrays to interpolate per bixel.
    """

    def __init__(
        self,
        model: "BiologicalModel",
        class_alpha_x: Any,
        class_beta_x: Any,
        voxel_params: dict[str, Any],
        kernel_fields: list[str],
    ):
        super().__init__(model)
        self.class_alpha_x = class_alpha_x
        self.class_beta_x = class_beta_x
        self.kernel_fields = list(kernel_fields)
        # Fail early with a clear message if any voxel has no matching tissue class.
        self.model.match_tissue_classes(
            voxel_params["alpha_x"], voxel_params["beta_x"], class_alpha_x, class_beta_x
        )

    def kernel_quantities(self, kernel: dict[str, Any]) -> dict[str, Any]:
        return {name: kernel[name] for name in self.kernel_fields}

    def bixel_alpha_beta(self, bixel: dict[str, Any], kernels: dict[str, Any]) -> tuple[Any, Any]:
        xp = array_api_compat.array_namespace(bixel["v_alpha_x"])
        class_ix = self.model.match_tissue_classes(
            bixel["v_alpha_x"], bixel["v_beta_x"], self.class_alpha_x, self.class_beta_x
        )
        voxel_ix = xp.arange(class_ix.shape[0])
        rows = {name: kernels[name][class_ix, voxel_ix] for name in self.kernel_fields}
        return self.model.alpha_beta_from_kernel_rows(rows)


class TabulatedSpectrumEvaluator(TissueClassEvaluator):
    """
    Tissue-class evaluator whose kernel arrays are pre-computed from fluence spectra.

    On construction the model's lookup tables are dose-averaged over the fragment
    fluence spectra of every machine energy, producing ``(n_classes, n_depths)`` arrays
    per energy. The machine itself is left untouched.
    """

    def __init__(self, model: "BiologicalModel", machine: Any, voxel_params: dict[str, Any]):
        first_kernel = machine.pb_kernels[machine.energies[0]]
        fragments = model.select_fragments(first_kernel.fluence_spectrum)
        self._tables = {
            float(energy): model.dose_average(kernel, fragments)
            for energy, kernel in machine.pb_kernels.items()
        }
        self._converted: dict[float, dict[str, Any]] = {}
        super().__init__(
            model,
            model.table_alpha_x,
            model.table_beta_x,
            voxel_params,
            model.quantities_in_kernel,
        )

    def kernel_quantities(self, kernel: dict[str, Any]) -> dict[str, Any]:
        energy = float(kernel["energy"])
        if energy not in self._converted:
            xp = array_api_compat.array_namespace(kernel["depths"])
            self._converted[energy] = {
                name: xp.asarray(arr) for name, arr in self._tables[energy].items()
            }
        return self._converted[energy]
