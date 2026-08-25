"""RBE-weighted dose quantity.

RBExDose can be derived from physical dose in two different ways:

1. `RBExDoseFromAlphaBeta` -- via the linear-quadratic effect model, using
   dose-averaged alpha/beta parameters (alphax, betax) from the dij.
2. `RBExDoseFromConstantRBE` -- as a constant factor times physical dose
   (RBExDose = rbe * dose), e.g. the classic "RBE = 1.1" proton convention.

`RBExDose` is the abstract base class shared by both. It carries the
common metadata (unit, identifier, name) and defines the interface that
subclasses must implement. `RBExDose.resolve_implementation` picks the
subclass based on what the dij provides.
"""

from abc import ABC, abstractmethod

import pint

from ..core.xp_utils.typing import Array
from pyRadPlan.quantities._base import FluenceDependentQuantity

ureg = pint.UnitRegistry()


class RBExDose(FluenceDependentQuantity, ABC):
    """RBE-weighted dose (abstract base).

    Concrete behavior (how the quantity and its chain derivative are
    computed) is provided by subclasses:

    - `RBExDoseFromAlphaBeta`
    - `RBExDoseFromConstantRBE`
    """

    unit = ureg.gray
    dim = 1
    identifier = "rbe_x_dose"
    name = "RBExDose"

    @abstractmethod
    def _compute_quantity_single_scenario(self, scenario_index: int) -> Array:
        """Compute RBExDose for a single scenario."""
        raise NotImplementedError

    @abstractmethod
    def _compute_chain_derivative_single_scenario(
        self, d_quantity: Array, scenario_index: int
    ) -> Array:
        """Backpropagate a gradient w.r.t. RBExDose through this quantity."""
        raise NotImplementedError

    @classmethod
    def resolve_implementation(cls, dij) -> type["RBExDose"]:
        """Pick the concrete RBExDose subclass based on what the dij provides.

        Used by ``QuantityResolver`` when it builds the dependency graph
        (see resolver.py's ``_resolve_implementation``).
        """
        has_alpha_beta = (
            getattr(dij, "alpha_dose", None) is not None
            and getattr(dij, "sqrt_beta_dose", None) is not None
        )

        if has_alpha_beta:
            return RBExDoseFromAlphaBeta
        if getattr(dij, "rbe", None) is not None:
            return RBExDoseFromConstantRBE
        raise ValueError(
            "Cannot compute 'rbe_x_dose': dij provides neither alpha_dose/sqrt_beta_dose "
            "matrices nor a constant 'rbe'."
        )


class RBExDoseFromAlphaBeta(RBExDose):
    """RBExDose computed from the linear-quadratic effect model."""

    required_dependencies = ("effect",)

    def _compute_quantity_single_scenario(self, scenario_index: int) -> Array:
        # TODO: correct handling of ct scenario
        xp = self.array_backend
        dtype_xp = self._dtype

        effect = self._deps["effect"].compute(self._w_cache)

        # Wrap all numpy slices upfront
        alphax = xp.asarray(self._dij.alphax[:, scenario_index])
        betax = xp.asarray(self._dij.betax[:, scenario_index])
        ix = xp.asarray(self._dij.betax[:, scenario_index] > 0)
        effect_slice = xp.asarray(effect.flat[scenario_index])

        gamma = xp.zeros(betax.shape, dtype=dtype_xp)
        gamma[ix] = alphax[ix] / betax[ix] / 2

        rbe_x_dose = xp.zeros(effect_slice.shape, dtype=dtype_xp)
        rbe_x_dose[ix] = xp.sqrt(gamma[ix] ** 2 + effect_slice[ix] / betax[ix]) - gamma[ix]
        return rbe_x_dose

    def _compute_chain_derivative_single_scenario(
        self, d_quantity: Array, scenario_index: int
    ) -> Array:
        # TODO: correct handling of ct scenarios
        xp = self.array_backend
        dtype_xp = self._dtype

        d_quantity = xp.reshape(d_quantity, (-1,))

        # Wrap all numpy slices upfront
        alphax = xp.asarray(self._dij.alphax[:, scenario_index])
        betax = xp.asarray(self._dij.betax[:, scenario_index])
        ix = xp.asarray(self._dij.betax[:, scenario_index] > 0)

        gamma = xp.zeros(betax.shape, dtype=dtype_xp)
        gamma[ix] = alphax[ix] / betax[ix] / 2

        # Sync effect's fluence cache to ours, then route the gradient through it.
        effect = self._deps["effect"].compute(self._w_cache)
        effect_slice = xp.asarray(effect.flat[scenario_index])
        effect_slice[ix] = effect_slice[ix] + gamma[ix]

        fgrad = xp.zeros(d_quantity.shape, dtype=dtype_xp)
        fgrad[ix] = d_quantity[ix] / (2 * betax[ix] * effect_slice[ix])

        fgrad = xp.reshape(fgrad, (1, -1))
        return self._deps["effect"]._compute_chain_derivative_single_scenario(
            fgrad, scenario_index
        )


class RBExDoseFromConstantRBE(RBExDose):
    """RBExDose computed as a constant RBE factor times physical dose.

    RBExDose = rbe * dose, so the chain derivative is just a scalar
    multiple of the dose's own chain derivative.
    """

    required_dependencies = ("physical_dose",)

    def _compute_quantity_single_scenario(self, scenario_index: int) -> Array:
        return self.array_backend.asarray(
            self._dij.rbe * self._dij.physical_dose.flat[scenario_index] @ self._w_cache,
            copy=False,
        )

    def _compute_chain_derivative_single_scenario(
        self, d_quantity: Array, scenario_index: int
    ) -> Array:
        # Transpose form is array-api compliant (scipy / array_api_strict compatibility).
        return self.array_backend.asarray(
            self._dij.rbe * self._dij.physical_dose.flat[scenario_index].__rmatmul__(d_quantity),
            copy=False,
        )
