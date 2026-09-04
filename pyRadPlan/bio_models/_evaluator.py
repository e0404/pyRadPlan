"""Evaluators: per-dose-calculation state and evaluation of biological models."""

from __future__ import annotations

from abc import ABC
from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import array_api_compat

from ._tissue_lookup import TissueParameterLookup

if TYPE_CHECKING:  # pragma: no cover
    from ._base import BiologicalModel


class BioEvaluationContext(Mapping[str, Any]):
    """Named inputs available during one biological-model evaluation.

    The mapping keeps the evaluator contract independent of a particular biological
    methodology. Callers explicitly construct the context from the inputs exposed to the
    model; engine-internal state is not forwarded implicitly.

    The core input vocabulary is ``alpha_x``, ``beta_x``, ``physical_dose`` and ``let``.
    Dose engines supply the applicable subset and may add the model-specific fields declared
    by :meth:`BioModelEvaluator.kernel_quantities`; those fields keep their declared names
    after interpolation. Engine-local geometry and raw kernel objects are not exposed.

    Parameters
    ----------
    inputs : mapping
        Biological inputs keyed by their semantic name.
    """

    def __init__(self, inputs: Mapping[str, Any] | None = None):
        values = {} if inputs is None else dict(inputs)
        if not all(isinstance(name, str) for name in values):
            raise TypeError("Biological evaluation input names must be strings.")
        self._inputs = MappingProxyType(values)

    @property
    def inputs(self) -> Mapping[str, Any]:
        """Read-only view of the named inputs."""
        return self._inputs

    def __getitem__(self, name: str) -> Any:
        return self._inputs[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._inputs)

    def __len__(self) -> int:
        return len(self._inputs)

    def require(self, name: str) -> Any:
        """Return a required input with a model-facing error if it is unavailable."""
        try:
            return self._inputs[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._inputs)) or "none"
            raise ValueError(
                f"Biological evaluation requires input '{name}'; available inputs: {available}."
            ) from exc


class BioModelResult(Mapping[str, Any]):
    """Named quantities produced by a biological-model evaluation.

    A mapping rather than an alpha/beta-specific tuple permits models to return direct RBE,
    effect, survival or future biological endpoints without extending this interface.

    Parameters
    ----------
    quantities : mapping
        Evaluated arrays keyed by their semantic quantity name.
    """

    def __init__(self, quantities: Mapping[str, Any] | None = None):
        values = {} if quantities is None else dict(quantities)
        if not all(isinstance(name, str) for name in values):
            raise TypeError("Biological result quantity names must be strings.")
        self._quantities = MappingProxyType(values)

    @property
    def quantities(self) -> Mapping[str, Any]:
        """Read-only view of the named result quantities."""
        return self._quantities

    def __getitem__(self, name: str) -> Any:
        return self._quantities[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._quantities)

    def __len__(self) -> int:
        return len(self._quantities)

    def require(self, name: str) -> Any:
        """Return a required result with a clear error if the model did not produce it."""
        try:
            return self._quantities[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._quantities)) or "none"
            raise ValueError(
                f"Biological model did not produce quantity '{name}'; produced: {available}."
            ) from exc


class BioModelEvaluator(ABC):
    """
    Evaluates one :class:`BiologicalModel` for one dose calculation.

    Created by :meth:`BiologicalModel.evaluator` and owned by the dose engine for the
    duration of a single dose calculation. Holds only derived, machine- and
    geometry-specific state; never reuse it across patients or machines.

    The dose engine interacts with a model exclusively through this interface.
    """

    def __init__(self, model: BiologicalModel):
        self.model = model

    @property
    def kernel_field_names(self) -> tuple[str, ...]:
        """Names of the depth-dependent kernel fields requested by this evaluator."""
        return ()

    def kernel_quantities(self, kernel: dict[str, Any]) -> dict[str, Any]:
        """
        Depth-dependent arrays the engine must interpolate per bixel for this model.

        Returned keys must match :attr:`kernel_field_names` for every machine energy.

        Parameters
        ----------
        kernel : dict
            Pencil-beam kernel of the current bixel's energy (array-namespace dict).

        Returns
        -------
        dict[str, Array]
            Arrays of shape ``(n_depths,)`` or ``(n_classes, n_depths)`` keyed by name;
            after interpolation they become named inputs to :meth:`evaluate`.
        """
        return {}

    def evaluate(self, context: BioEvaluationContext) -> BioModelResult:
        """Evaluate the model from named inputs and return named biological quantities."""
        raise NotImplementedError(
            f"Biological model '{self.model.model}' does not provide evaluated quantities."
        )


class ParametricEvaluator(BioModelEvaluator):
    """
    Evaluator for models that are pure functions of the per-voxel parameters.

    Holds nothing but the model; forwards to :meth:`BiologicalModel.alpha_beta`.
    """

    def evaluate(self, context: BioEvaluationContext) -> BioModelResult:
        if not self.model.provides("alpha", "beta"):
            raise NotImplementedError(
                f"Biological model '{self.model.model}' does not provide alpha/beta values."
            )
        alpha, beta = self.model.alpha_beta(
            context.require("alpha_x"), context.require("beta_x"), context
        )
        return BioModelResult({"alpha": alpha, "beta": beta})


class _TissueKernelEvaluator(BioModelEvaluator):
    """Shared evaluation of tissue-class kernel fields."""

    kernel_fields: list[str]
    lookup: TissueParameterLookup

    @property
    def kernel_field_names(self) -> tuple[str, ...]:
        """Names of the tissue kernel fields requested by this evaluator."""
        return tuple(self.kernel_fields)

    def evaluate(self, context: BioEvaluationContext) -> BioModelResult:
        rows = self.lookup.gather(
            context.require("alpha_x"),
            context.require("beta_x"),
            context,
            self.kernel_fields,
        )
        alpha, beta = self.model.alpha_beta_from_kernel_rows(rows)
        return BioModelResult({"alpha": alpha, "beta": beta})


class KernelBasedEvaluator(_TissueKernelEvaluator):
    """
    Evaluator for models reading pre-tabulated alpha/beta kernels from the machine data.

    The machine kernels carry ``(n_classes, n_depths)`` arrays per tissue class; a
    :class:`TissueParameterLookup` decides which class row each voxel uses.

    Parameters
    ----------
    model : BiologicalModel
        Must implement ``alpha_beta_from_kernel_rows``.
    kernel_fields : Sequence[str]
        Names of the machine kernel arrays to interpolate per bixel (e.g. ``("alpha", "beta")``).
    lookup : TissueParameterLookup
        Voxel parameter → tissue class mapping.
    voxel_params : dict[str, Array]
        Per-voxel parameters of the whole dose grid; validated up front so that a
        structure without matching base data fails before the dose calculation starts.
    """

    def __init__(
        self,
        model: BiologicalModel,
        kernel_fields: Sequence[str],
        lookup: TissueParameterLookup,
        voxel_params: dict[str, Any],
    ):
        super().__init__(model)
        self.kernel_fields = list(kernel_fields)
        self.lookup = lookup
        self.lookup.validate(voxel_params)

    def kernel_quantities(self, kernel: dict[str, Any]) -> dict[str, Any]:
        return {name: kernel[name] for name in self.kernel_fields}


class TabulatedSpectrumEvaluator(_TissueKernelEvaluator):
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
        model: BiologicalModel,
        machine: Any,
        lookup: TissueParameterLookup,
        voxel_params: dict[str, Any],
    ):
        super().__init__(model)
        self.kernel_fields = list(model.quantities_in_kernel)
        self.lookup = lookup
        self.lookup.validate(voxel_params)

        self._tables = {
            float(energy): model.dose_average(
                kernel, model.select_fragments(kernel.fluence_spectrum)
            )
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
