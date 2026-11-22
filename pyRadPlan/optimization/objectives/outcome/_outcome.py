"""Base class for outcome (e.g. NTCP) model objectives."""

from abc import abstractmethod
from typing import Any, ClassVar, Optional
import logging

import array_api_compat
import numpy as np
import SimpleITK as sitk
from pydantic import Field, PrivateAttr

from pyRadPlan.core.resample import resample_image
from pyRadPlan.optimization.objectives._objective import Objective

logger = logging.getLogger(__name__)


class OutcomeObjective(Objective):
    """Objective returning a predicted outcome probability of the dose.

    The model sees the dose of the VOI the objective is attached to, with 0 Gy
    everywhere else. To feed it the full dose, attach it to a helper VOI covering the
    dose grid with ``apply_overlap=None``.

    Subclasses implement :meth:`_predict` and :meth:`_predict_with_gradient` on the
    full dose grid.

    Parameters
    ----------
    monitor_only : bool
        Return 0 and a zero gradient, but still record the predicted probability.
    warmup_iterations : int
        Number of gradient evaluations returning a zero gradient before the model
        gradient is used.
    num_of_fractions : int, optional
        For models trained on total-course dose: the per-fraction dose is multiplied by
        this before it reaches the model.
    transform : bool
        Apply pyanno4rt's SALU transform: above ``p = 0.5`` the value becomes
        ``0.25 * logit(p) + 0.5``, which keeps the gradient from vanishing.
    gradient_vois : list[str], optional
        Optimize the outcome through the dose in these VOIs only. When the warmup ends,
        the dose elsewhere is frozen at the current iterate, so the model keeps a plausible
        full-dose input and the masked gradient stays exact. :attr:`history` then records
        the prediction on that input, not on the plan's full dose.
    """

    has_hessian: ClassVar[bool] = False

    monitor_only: bool = Field(default=False)
    warmup_iterations: int = Field(default=0, ge=0)
    num_of_fractions: Optional[int] = Field(default=None, gt=0)
    transform: bool = Field(default=False)
    gradient_vois: Optional[list[str]] = Field(default=None, min_length=1)

    _history: list[float] = PrivateAttr(default_factory=list)
    _iteration: int = PrivateAttr(default=0)
    # VOI voxel indices on the dose grid; None when the VOI covers the whole grid.
    _dose_index: Any = PrivateAttr(default=None)
    _dose_n: Optional[int] = PrivateAttr(default=None)
    _gradient_image: Optional[sitk.Image] = PrivateAttr(default=None)
    _gradient_mask: Optional[np.ndarray] = PrivateAttr(default=None)
    # Set when the warmup ends: the frozen full-grid dose and the mask in its namespace.
    _frozen_dose: Any = PrivateAttr(default=None)
    _frozen_mask: Any = PrivateAttr(default=None)

    def __deepcopy__(self, memo=None):
        """Share runtime handles (models, extractors); they don't deepcopy reliably."""
        copied = self.model_copy()
        copied._history = list(self._history)
        return copied

    @property
    def history(self) -> list[float]:
        """Predicted probabilities of all objective evaluations of this optimization."""
        return self._history

    def setup(self, ct, cst) -> None:
        """Keep the union of the ``gradient_vois`` masks."""
        self._gradient_image = None
        if self.gradient_vois is None:
            return
        vois = {voi.name: voi for voi in cst.vois}
        missing = [name for name in self.gradient_vois if name not in vois]
        if missing:
            raise ValueError(f"gradient_vois not in the structure set: {missing}")
        union = None
        for name in self.gradient_vois:
            mask = vois[name].mask
            if mask.GetDimension() == 4:
                mask = mask[:, :, :, 0]
            mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
            union = mask if union is None else union | mask
        self._gradient_image = union

    def preprocess_image_reference_parameters(self, target_grid, index_list=None) -> None:
        """Set up the dose grid; called once per optimization."""
        super().preprocess_image_reference_parameters(target_grid, index_list)
        self._history = []
        self._iteration = 0
        self._dose_n = int(np.prod(target_grid.dimensions))
        covers_grid = index_list is None or np.size(index_list) == self._dose_n
        self._dose_index = None if covers_grid else np.asarray(index_list)
        self._gradient_mask = None
        self._frozen_dose = None
        self._frozen_mask = None
        if self._gradient_image is not None:
            mask = resample_image(
                self._gradient_image,
                sitk.sitkNearestNeighbor,
                target_grid=target_grid,
                extrapolate=False,
            )
            self._gradient_mask = sitk.GetArrayViewFromImage(mask).ravel() > 0
            if not self._gradient_mask.any():
                raise ValueError(f"gradient_vois {self.gradient_vois} are empty on the dose grid")
        self._set_dose_grid(target_grid)

    def compute_objective(self, dose_values):
        """Compute the (transformed) predicted probability; 0 when *monitor_only*."""
        xp = array_api_compat.array_namespace(dose_values)
        prob = self._predict(self._model_input(self._to_full_grid(dose_values, xp), xp))
        self._history.append(prob)

        value = 0.0 if self.monitor_only else self._transform(prob)[0]
        return xp.asarray(
            value, dtype=dose_values.dtype, device=array_api_compat.device(dose_values)
        )

    def compute_gradient(self, dose_values):
        """Compute the gradient of the predicted probability w.r.t. dose."""
        xp = array_api_compat.array_namespace(dose_values)
        self._iteration += 1

        if self.monitor_only or self._iteration <= self.warmup_iterations:
            if self._iteration == self.warmup_iterations and not self.monitor_only:
                logger.info(
                    "%s warmup complete after %d iterations.", self.name, self.warmup_iterations
                )
            return xp.zeros(
                array_api_compat.size(dose_values),
                dtype=dose_values.dtype,
                device=array_api_compat.device(dose_values),
            )

        full = self._to_full_grid(dose_values, xp)
        if self._gradient_mask is not None and self._frozen_dose is None:
            # A copy: the solver may reuse the buffer behind dose_values.
            self._frozen_dose = xp.asarray(full, copy=True)
            self._frozen_mask = xp.asarray(
                self._gradient_mask, device=array_api_compat.device(full)
            )
        prob, grad = self._predict_with_gradient(self._model_input(full, xp))
        grad = grad * (self._transform(prob)[1] * (self.num_of_fractions or 1))
        if self._frozen_mask is not None:
            grad = xp.where(self._frozen_mask, grad, xp.zeros_like(grad))
        return self._from_full_grid(grad, dose_values, xp)

    def report_info(self) -> dict:
        """Return the probability history of the last optimization."""
        return {"history": list(self._history)} if self._history else {}

    def _transform(self, p: float) -> tuple[float, float]:
        """Return the objective value and its derivative w.r.t. ``p``."""
        if not self.transform or p <= 0.5:
            return p, 1.0
        return 0.25 * float(np.log(p / (1.0 - p))) + 0.5, 0.25 / (p - p**2)

    def _set_dose_grid(self, dose_grid) -> None:
        """Set up model-specific dose grid handling."""

    @abstractmethod
    def _predict(self, dose) -> float:
        """Predict the probability for a full-grid dose."""

    @abstractmethod
    def _predict_with_gradient(self, dose) -> tuple[float, Any]:
        """Predict the probability and its full-grid gradient, in the namespace of ``dose``."""

    def _to_full_grid(self, dose_values, xp):
        """Scatter VOI dose onto the full dose grid and scale it to model dose."""
        if self._dose_index is not None:
            device = array_api_compat.device(dose_values)
            full = xp.zeros(self._dose_n, dtype=dose_values.dtype, device=device)
            full[xp.asarray(self._dose_index, device=device)] = dose_values
            dose_values = full
        if self.num_of_fractions is not None:
            dose_values = dose_values * self.num_of_fractions
        return dose_values

    def _model_input(self, full_dose, xp):
        """Replace the dose outside ``gradient_vois`` with the frozen dose, once frozen."""
        if self._frozen_dose is None:
            return full_dose
        return xp.where(self._frozen_mask, full_dose, self._frozen_dose)

    def _from_full_grid(self, grad_full, dose_values, xp):
        """Gather a full-grid gradient back to the VOI voxels."""
        if self._dose_index is None:
            return grad_full
        return grad_full[xp.asarray(self._dose_index, device=array_api_compat.device(dose_values))]
