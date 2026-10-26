from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Union

import logging

import numpy as np
import array_api_compat
import SimpleITK as sitk

from ..core.xp_utils.typing import Array, ArrayNamespace


import pint

from pyRadPlan.dij import Dij
from pyRadPlan.core import xp_utils as compute_backend


logger = logging.getLogger(__name__)


class RTQuantity(ABC):
    # TODO: missing description
    name: ClassVar[str]
    identifier: ClassVar[str]
    unit: ClassVar[pint.Unit]
    dim: ClassVar[int]  # To differentiate between scalar and vector quantities
    #: Whether the quantity describes a biological effect (e.g. RBE-weighted dose or LQ
    #: effect) rather than a physical one.
    biological: ClassVar[bool] = False
    precision: str

    def __init__(self, dij: Dij, scenarios=None):
        if scenarios is None:
            scenarios = [0]
        self.scenarios = np.asarray(scenarios, dtype=np.int64)


class FluenceDependentQuantity(RTQuantity, ABC):
    """
    Base class for quantities that depend on fluence distributions.

    Concrete subclasses declare how they relate to a :class:`Dij` via two ClassVars:

    - :attr:`required_dependencies`: identifiers of upstream quantities that are always
      needed to compute this quantity. The resolver must produce them; failure raises.
    - :attr:`optional_dependencies`: identifiers of upstream quantities used as a fallback
      computation path when no direct Dij matrix is available.
    - :attr:`indirect_dij_fields`: Dij attributes (e.g. ``alphax``) the indirect path reads
      directly; if any is missing, the indirect path is unavailable.

    The resolution mode (``"direct"`` vs ``"indirect"``) is decided at instantiation time:
    if the dij carries an attribute whose name matches this quantity's ``identifier``, the
    direct path wins; otherwise the indirect path is used and ``required + optional`` deps
    must resolve.  This enforces a consistent naming convention: a Dij matrix attribute must
    share its name with the quantity identifier it represents.
    """

    # ClassVar metadata describing how this quantity is computed.
    required_dependencies: ClassVar[tuple[str, ...]] = ()
    optional_dependencies: ClassVar[tuple[str, ...]] = ()
    indirect_dij_fields: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def has_indirect_path(cls, dij: Dij) -> bool:
        """Whether the dependency-based path is available for ``dij``."""
        if not (cls.required_dependencies or cls.optional_dependencies):
            return False
        return all(getattr(dij, field, None) is not None for field in cls.indirect_dij_fields)

    array_backend: ArrayNamespace

    def __init__(
        self,
        dij: Dij,
        *,
        mode: Optional[str] = None,
        dependencies: Optional[dict[str, "FluenceDependentQuantity"]] = None,
        scenarios=None,
    ):
        super().__init__(dij, scenarios=scenarios)
        xp = compute_backend.choose_array_api_namespace()
        device = compute_backend.choose_device(xp)

        logger.info("Optimization will use array backend: %s", xp.__name__)

        self.array_backend: ArrayNamespace = xp

        # The resolver pre-converts the dij to the target namespace and supplies
        # `mode` + `dependencies`. When invoked directly (e.g. from tests) we do the
        # conversion ourselves and build deps lazily via the resolver.
        if mode is None:
            self._dij = dij.to_namespace(xp, device=device)
            self.device = device
            self._mode = self._choose_mode()
            self._deps: dict[str, FluenceDependentQuantity] = dict(dependencies or {})
            if self._mode == "indirect" and not self._deps:
                # Lazy import to avoid circular dependency at module import time.
                from ._resolver import QuantityResolver  # noqa: PLC0415

                resolver = QuantityResolver(
                    self._dij, scenarios=self.scenarios, _dij_already_in_namespace=True
                )
                for dep_id in self.required_dependencies:
                    self._deps[dep_id] = resolver.get(dep_id)
                for dep_id in self.optional_dependencies:
                    dep = resolver.try_get(dep_id)
                    if dep is not None:
                        self._deps[dep_id] = dep
        else:
            if mode not in ("direct", "indirect"):
                raise ValueError(f"Unknown quantity mode: {mode!r}")
            self._dij = dij
            self._mode = mode
            self._deps = dict(dependencies or {})
            self.device = device

        self._validate_dependencies()

        # dtype is inferred from the matrix actually driving the computation in
        # direct mode, and from the canonical `physical_dose` container otherwise.
        self._dtype = self._infer_dtype(xp)

        # Fluence cache for derivative calculation
        self._w_cache: Union[Array, None] = None
        self._w_grad_cache: Union[Array, None] = None
        # Quantity vector cache (shape mirrors the scenario container layout)
        self._q_cache = np.empty_like(getattr(self._dij, "physical_dose"), dtype=object)
        self._qgrad_cache = np.empty_like(self._q_cache)

    @property
    def mode(self) -> str:
        """The resolution mode chosen for this instance: ``"direct"`` or ``"indirect"``."""
        return self._mode

    @property
    def dependencies(self) -> dict[str, "FluenceDependentQuantity"]:
        """Mapping of identifier to resolved upstream quantity instance."""
        return self._deps

    def _choose_mode(self) -> str:
        """Pick the resolution mode for this dij; direct wins when both are available."""
        if getattr(self._dij, self.identifier, None) is not None:
            return "direct"
        if self.has_indirect_path(self._dij):
            return "indirect"
        raise ValueError(
            f"Quantity {type(self).__name__!r} cannot be computed: dij has no attribute "
            f"{self.identifier!r} and no usable dependency path."
        )

    def _validate_dependencies(self) -> None:
        if self._mode == "indirect":
            for dep_id in self.required_dependencies:
                if dep_id not in self._deps:
                    raise ValueError(
                        f"Required dependency {dep_id!r} missing for {type(self).__name__!r}."
                    )

    def check_fit_domain(self, regions: dict[str, np.ndarray]) -> list[str]:
        """
        Warn about a plan outside the conditions the quantity's model was fitted for.

        Called by the planning problem once the quantities are resolved. The default has
        no fitted model and returns no warnings.

        Parameters
        ----------
        regions : dict[str, np.ndarray]
            Boolean voxel masks on the dose grid, after overlap handling, of the VOIs whose
            objectives use this quantity, keyed by VOI name.

        Returns
        -------
        list[str]
            One message per violated condition.
        """
        return []

    def _infer_dtype(self, xp: ArrayNamespace):
        if self._mode == "direct":
            container = getattr(self._dij, self.identifier)
            sample = container.flat[0]
        else:
            sample = self._dij.physical_dose.flat[0]
        try:
            return getattr(xp, sample.dtype.name)
        except AttributeError:
            return sample.dtype

    def __call__(self, fluence: Array) -> Array:
        """
        Make the quantity callable by calling the compute method.

        Parameters
        ----------
        fluence : Array
            Fluence vector.

        Returns
        -------
        NDArray
            Quantity vector.
        """

        return self.compute(fluence)

    def _as_backend_array(self, fluence: Array) -> Array:
        """
        Move *fluence* onto this quantity's array namespace and device if it is elsewhere.

        A quantity binds its dij to the globally preferred namespace at construction time,
        which is not necessarily the namespace the caller holds its fluence in. The optimizer
        returns fluence as NumPy (``PlanningProblem.solve`` calls ``to_numpy``), so evaluating
        a quantity on an optimization result would otherwise mix, e.g., a CuPy dij with a
        NumPy fluence. Arrays already in the right place are returned untouched.
        """
        if array_api_compat.array_namespace(fluence) is self.array_backend and (
            self.device is None or array_api_compat.device(fluence) == self.device
        ):
            return fluence

        return compute_backend.from_numpy(
            self.array_backend, compute_backend.to_numpy(fluence), device=self.device
        )

    def compute(self, fluence: Array) -> Array:
        """
        Forward calculation of the quantity from the fluence.

        Parameters
        ----------
        fluence : ArrayLike
            Fluence vector.

        Returns
        -------
        NDArray
            Quantity vector.
        """

        fluence = self._as_backend_array(fluence)
        xp = array_api_compat.array_namespace(fluence)

        if not xp.isdtype(fluence.dtype, self._dtype):
            fluence = xp.asarray(fluence, dtype=self._dtype)

        # check if we need to update the cache
        if self._w_cache is None or not xp.all(self._w_cache == fluence):
            self._w_cache = xp.asarray(fluence, copy=True)
            self._compute_quantity_cache()

        return self._q_cache

    def compute_chain_derivative(self, d_quantity: Array, fluence: Array) -> Array:
        """
        Fluence Derivative of the quantity w.r.t. to the quantity derivative.

        Parameters
        ----------
        d_quantity : ArrayLike
            Derivative of w.r.t. to the quantity.
        fluence : ArrayLike
            Fluence vector.

        Returns
        -------
        NDArray
            Derivative of the quantity w.r.t. the fluence.
        """

        xp = array_api_compat.array_namespace(d_quantity, fluence)

        if not xp.isdtype(fluence.dtype, self._dtype):
            fluence = xp.asarray(fluence, dtype=self._dtype)

        if self._w_grad_cache is None or not xp.all(self._w_grad_cache == fluence):
            self._w_grad_cache = xp.asarray(fluence, copy=True)
            self._compute_chain_derivative_cache(d_quantity)

        return self._qgrad_cache

    def compute_dose_grid(self, fluence: Array, scenario_index: int = 0) -> "sitk.Image":
        """
        Compute the quantity and return it as a SimpleITK image on the dose grid.

        Parameters
        ----------
        fluence : Array
            Fluence vector.
        scenario_index : int
            The scenario index to compute the quantity for (default: 0).

        Returns
        -------
        sitk.Image
            The quantity as a SimpleITK image on the dose grid.
        """
        q_arr = compute_backend.to_numpy(self.compute(fluence).flat[scenario_index])
        q_img = sitk.GetImageFromArray(q_arr.reshape(self._dij.dose_grid.dimensions[::-1]))
        q_img.SetOrigin(tuple(self._dij.dose_grid.origin.tolist()))
        q_img.SetSpacing(tuple(self._dij.dose_grid.resolution_vector.tolist()))
        q_img.SetDirection(tuple(self._dij.dose_grid.direction.ravel().tolist()))

        return q_img

    def compute_ct_grid(self, fluence: Array, scenario_index: int = 0) -> "sitk.Image":
        """
        Compute the quantity and return it as a SimpleITK image on the CT grid.

        Parameters
        ----------
        fluence : Array
            Fluence vector.
        scenario_index : int
            The scenario index to compute the quantity for (default: 0).

        Returns
        -------
        sitk.Image
            The quantity as a SimpleITK image on the CT grid.
        """
        q_img = self.compute_dose_grid(fluence, scenario_index=scenario_index)

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputDirection(tuple(self._dij.ct_grid.direction.ravel().tolist()))
        resampler.SetOutputOrigin(tuple(self._dij.ct_grid.origin.tolist()))
        resampler.SetOutputSpacing(tuple(self._dij.ct_grid.resolution_vector.tolist()))
        resampler.SetSize(tuple(self._dij.ct_grid.dimensions))

        return resampler.Execute(q_img)

    def _compute_quantity_cache(self):
        """Compute the quantity from the fluence and write it into the cache."""

        for scenario_index in self.scenarios:
            self._q_cache.flat[scenario_index] = self._compute_quantity_single_scenario(
                scenario_index
            )

    def _compute_chain_derivative_cache(self, d_quantity: Array) -> Array:
        """Compute the fluence derivative from the quantity derivative into the cache."""

        for scenario_index in self.scenarios:
            self._qgrad_cache.flat[scenario_index] = (
                self._compute_chain_derivative_single_scenario(d_quantity, scenario_index)
            )

    @abstractmethod
    def _compute_quantity_single_scenario(self, scenario_index: int) -> Array:
        """
        Calculate the quantity in a specific scenario.

        Parameters
        ----------
        scenario_index : int
            Scenario index.

        Returns
        -------
        Array
            Quantity in the scenario.
        """

    @abstractmethod
    def _compute_chain_derivative_single_scenario(
        self, d_quantity: Array, scenario_index: int
    ) -> Array:
        """
        Calculate the derivative of the quantity w.r.t. the fluence in a specific scenario.

        Parameters
        ----------
        d_quantity : Array
            Derivative w.r.t. to the quantity.
        scenario_index : int
            Scenario index.

        Returns
        -------
        Array
            Derivative of the quantity w.r.t. the fluence in the scenario.
        """
