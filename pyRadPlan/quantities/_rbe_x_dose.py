"""RBE-weighted dose quantity."""

from typing import Any, Optional

import array_api_compat

from ..core.xp_utils.typing import Array
from pyRadPlan.quantities._base import FluenceDependentQuantity

from pyRadPlan.core._units import ureg


def lq_inverse_dose(effect: Array, alpha_x: Array, beta_x: Array) -> Array:
    """
    Photon-equivalent (RBE-weighted) dose of an LQ effect.

    The non-negative dose ``d`` solving ``alpha_x * d + beta_x * d**2 = effect``, evaluated as
    ``2 * effect / (sqrt(alpha_x**2 + 4 * beta_x * effect) + alpha_x)``. Unlike the textbook
    root this has no cancellation at low effect, which keeps single precision accurate, and
    gives the linear solution ``effect / alpha_x`` for a zero ``beta_x``. Voxels without a
    positive denominator (no positive reference parameters) get zero dose.
    """
    xp = array_api_compat.array_namespace(effect)
    denominator = xp.sqrt(alpha_x**2 + 4 * beta_x * effect) + alpha_x
    positive = denominator > 0
    return xp.where(positive, 2 * effect / xp.where(positive, denominator, 1.0), 0.0)


def lq_inverse_dose_derivative(rbe_x_dose: Array, alpha_x: Array, beta_x: Array) -> Array:
    """
    Differentiate :func:`lq_inverse_dose` with respect to the effect.

    From ``alpha_x * d + beta_x * d**2 = effect``: ``1 / (alpha_x + 2 * beta_x * d)``, zero
    where the denominator is not positive.
    """
    xp = array_api_compat.array_namespace(rbe_x_dose)
    denominator = alpha_x + 2 * beta_x * rbe_x_dose
    positive = denominator > 0
    return xp.where(positive, 1 / xp.where(positive, denominator, 1.0), 0.0)


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
    biological = True
    name = "RBExDose"
    fraction_exponent = 1.0

    optional_dependencies = ("effect", "physical_dose")

    def __init__(self, dij, *, mode=None, dependencies=None, scenarios=None):
        super().__init__(dij, mode=mode, dependencies=dependencies, scenarios=scenarios)
        self._reference_params_cache: dict[int, tuple[Array, Array]] = {}
        # The alpha_x / beta_x fallbacks of alpha_dose / sqrt_beta_dose give a photon effect
        # (RBE = 1), so only real LQ matrices qualify for the effect path.
        has_lq = self._dij.alpha_dose is not None and self._dij.sqrt_beta_dose is not None
        self._path = self.choose_path(self._dij, has_effect=has_lq and "effect" in self._deps)
        if self._path == "constant":
            if "physical_dose" not in self._deps:
                raise ValueError(
                    "RBExDose with a constant RBE needs the 'physical_dose' quantity."
                )
            self._deps.pop("effect", None)

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
        if model is not None and model.provides("alpha", "beta"):
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
        # Cast once to the quantity's dtype: the dij stores them in double precision, which
        # would promote every evaluation (and the upstream gradient) to slow float64 math.
        if scenario_index not in self._reference_params_cache:
            xp = self.array_backend
            alphax = xp.asarray(self._dij.alphax[:, scenario_index])
            betax = xp.asarray(self._dij.betax[:, scenario_index])
            self._reference_params_cache[scenario_index] = (
                xp.astype(alphax, self._dtype),
                xp.astype(betax, self._dtype),
            )
        return self._reference_params_cache[scenario_index]

    def _compute_quantity_single_scenario(self, scenario_index: int) -> Array:
        xp = self.array_backend
        if self._path == "constant":
            # Scale the already resolved physical dose vector; `rbe * matrix @ w` would
            # instead build a scaled copy of the whole influence matrix per call.
            physical_dose = self._deps["physical_dose"].compute(self._w_cache)
            return xp.asarray(self._dij.rbe * physical_dose.flat[scenario_index], copy=False)
        effect = self._deps["effect"].compute(self._w_cache)
        alphax, betax = self._reference_params(scenario_index)
        effect_slice = xp.astype(xp.asarray(effect.flat[scenario_index]), self._dtype)
        return lq_inverse_dose(effect_slice, alphax, betax)

    def _compute_chain_derivative_single_scenario(
        self, d_quantity: Array, scenario_index: int
    ) -> Array:
        xp = self.array_backend
        if self._path == "constant":
            # d(rbe * physical_dose)/dw = rbe * d(physical_dose)/dw; chaining through the
            # physical-dose quantity avoids scaling the influence matrix itself.
            return self._deps["physical_dose"]._compute_chain_derivative_single_scenario(
                xp.asarray(self._dij.rbe * d_quantity), scenario_index
            )

        alphax, betax = self._reference_params(scenario_index)
        # One row per upstream gradient (e.g. per objective without the jacobian bypass).
        d_quantity = xp.reshape(d_quantity, (-1, alphax.shape[0]))
        # The forward cache holds the RBE-weighted dose at the fluence being differentiated.
        rbe_x_dose = self._q_cache.flat[scenario_index]
        fgrad = d_quantity * lq_inverse_dose_derivative(rbe_x_dose, alphax, betax)
        return self._deps["effect"]._compute_chain_derivative_single_scenario(
            fgrad, scenario_index
        )
