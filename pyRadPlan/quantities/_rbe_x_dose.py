"""RBE-weighted dose quantity."""

from typing import Any, Optional

import array_api_compat
import pint

from ..core.xp_utils.typing import Array
from pyRadPlan.quantities._base import FluenceDependentQuantity

ureg = pint.UnitRegistry()


def lq_inverse_dose(effect: Array, alpha_x: Array, beta_x: Array) -> Array:
    """
    Photon-equivalent (RBE-weighted) dose of an LQ effect: the dose ``d`` solving
    ``alpha_x * d + beta_x * d**2 = effect``. Zero where ``beta_x`` is zero.
    """
    xp = array_api_compat.array_namespace(effect)
    valid = beta_x > 0
    dose = xp.zeros(effect.shape, dtype=effect.dtype)
    dose[valid] = (
        xp.sqrt(alpha_x[valid] ** 2 + 4 * beta_x[valid] * effect[valid]) - alpha_x[valid]
    ) / (2 * beta_x[valid])
    return dose


class RBExDose(FluenceDependentQuantity):
    """
    RBE-weighted dose.

    Computed along one of two paths, chosen from the dij's biological model
    (:meth:`choose_path`):

    - ``"effect"``: LQ inversion of the ``effect`` quantity with the reference photon
      parameters ``alphax`` / ``betax`` of the dij.
    - ``"constant"``: ``rbe * physical_dose`` with the constant RBE of the dij's model.
    """

    unit = ureg.gray
    dim = 1
    identifier = "rbe_x_dose"
    name = "RBExDose"

    optional_dependencies = ("effect", "physical_dose")

    def __init__(self, dij, *, mode=None, dependencies=None, scenarios=None):
        super().__init__(dij, mode=mode, dependencies=dependencies, scenarios=scenarios)
        self._path = self.choose_path(self._dij, has_effect="effect" in self._deps)
        if self._path == "constant" and "physical_dose" not in self._deps:
            raise ValueError("RBExDose with a constant RBE needs the 'physical_dose' quantity.")

    @property
    def path(self) -> str:
        """``"effect"`` or ``"constant"``."""
        return self._path

    @staticmethod
    def choose_path(dij: Any, has_effect: bool, strict: bool = True) -> Optional[str]:
        """
        Decide how RBE-weighted dose is derived from a dij.

        The dij's ``bio_model`` decides: a model providing alpha/beta requires the LQ
        effect (i.e. ``alpha_dose`` / ``sqrt_beta_dose`` matrices), a constant-RBE model
        scales the physical dose. Without a model, the effect is used when available.

        Parameters
        ----------
        dij : Dij
        has_effect : bool
            Whether the ``effect`` quantity can be computed for this dij.
        strict : bool
            Raise if no path is possible (default); otherwise return ``None``.

        Returns
        -------
        ``"effect"``, ``"constant"`` or ``None``
        """
        model = getattr(dij, "bio_model", None)
        rbe = getattr(dij, "rbe", None)
        if model is not None and model.provides_alpha_beta:
            if has_effect:
                return "effect"
            if not strict:
                return None
            raise ValueError(
                f"'rbe_x_dose' with biological model {model!r} needs alpha_dose / "
                "sqrt_beta_dose influence matrices, but the dij has none "
                "(computed with calc_bio_dose switched off?)."
            )
        if rbe is not None:
            return "constant"
        if has_effect:
            return "effect"
        if not strict:
            return None
        raise ValueError(
            "Cannot compute 'rbe_x_dose': dij provides neither alpha_dose/sqrt_beta_dose "
            "matrices nor a constant RBE."
        )

    # ------------------------------------------------------------------ effect path
    def _reference_params(self, scenario_index: int):
        # TODO: correct handling of ct scenarios
        xp = self.array_backend
        alphax = xp.asarray(self._dij.alphax[:, scenario_index])
        betax = xp.asarray(self._dij.betax[:, scenario_index])
        return alphax, betax

    def _compute_quantity_single_scenario(self, scenario_index: int) -> Array:
        xp = self.array_backend
        if self._path == "constant":
            return xp.asarray(
                self._dij.rbe * self._dij.physical_dose.flat[scenario_index] @ self._w_cache,
                copy=False,
            )
        effect = self._deps["effect"].compute(self._w_cache)
        alphax, betax = self._reference_params(scenario_index)
        effect_slice = xp.astype(xp.asarray(effect.flat[scenario_index]), self._dtype)
        return lq_inverse_dose(effect_slice, alphax, betax)

    def _compute_chain_derivative_single_scenario(
        self, d_quantity: Array, scenario_index: int
    ) -> Array:
        xp = self.array_backend
        if self._path == "constant":
            # Transpose form is array-api compliant (scipy / array_api_strict compatibility).
            return xp.asarray(
                self._dij.rbe
                * self._dij.physical_dose.flat[scenario_index].__rmatmul__(d_quantity),
                copy=False,
            )

        dtype_xp = self._dtype
        d_quantity = xp.reshape(d_quantity, (-1,))
        alphax, betax = self._reference_params(scenario_index)
        ix = betax > 0

        gamma = xp.zeros(betax.shape, dtype=dtype_xp)
        gamma[ix] = alphax[ix] / betax[ix] / 2

        # d(rbe_x_dose)/d(effect) = 1 / (2 * beta_x * (rbe_x_dose + gamma))
        effect = self._deps["effect"].compute(self._w_cache)
        effect_slice = xp.asarray(effect.flat[scenario_index])
        effect_slice[ix] = effect_slice[ix] + gamma[ix]

        fgrad = xp.zeros(d_quantity.shape, dtype=dtype_xp)
        fgrad[ix] = d_quantity[ix] / (2 * betax[ix] * effect_slice[ix])

        fgrad = xp.reshape(fgrad, (1, -1))
        return self._deps["effect"]._compute_chain_derivative_single_scenario(
            fgrad, scenario_index
        )
