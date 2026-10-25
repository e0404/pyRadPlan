"""CNN-based NTCP objective.

Uses a pre-trained CNN loaded via :func:`pyRadPlan.ai.modelhub.load_model` to
predict an NTCP and compute its gradient w.r.t. dose for use in optimisation.
All model-specific behaviour (grid geometry, cropping, CT/mask resampling, dose
normalisation, differentiable interpolation) lives in the preprocessor that
ships *with* the model on HuggingFace, so this objective stays thin and
backend-agnostic (numpy / cupy / torch via ``xp_utils``).

The shipped preprocessor is expected to expose:

- ``configure(ct, cst, cst_masks=None, device=None)`` — one-time CT/mask setup.
- ``set_dose_grid(grid)`` — precompute dose<->model sampling coordinates.
- ``preprocess(dose, requires_grad=False)`` — model inputs, called as
  ``model(*inputs)``; with ``requires_grad`` they are differentiable w.r.t. dose.
- ``postprocess(outputs)`` — map the model output to a scalar probability.
- ``gradient_to_dose_grid()`` — gradient of the last backward w.r.t. the flat
  dose-grid dose.

Example
-------
>>> ntcp_obj = OutcomeCNN(
...     priority=10000.0,
...     model="pyRadPlan-outcome-ORPDenseNet-tg119",  # HF name, local dir, or (model, preprocessor)
...     warmup_iterations=5,
...     ct=ct,
...     cst=cst,
...     cst_masks=["Core"],
... )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal, Optional
import logging

import array_api_compat
import numpy as np
from pydantic import Field, PrivateAttr, model_validator

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from pyRadPlan.core import xp_utils
from pyRadPlan.ai.modelhub import load_model
from pyRadPlan.optimization.objectives._objective import Objective

logger = logging.getLogger(__name__)


def _default_model_device() -> str:
    """Device for a model the objective loads itself, when none was requested."""
    return str(xp_utils.choose_device(array_api_compat.array_namespace(torch.empty(0))))


class OutcomeCNN(Objective):
    """CNN-based NTCP objective.

    The CNN sees the dose of the VOI the objective is attached to, with 0 Gy
    everywhere else. Attach it to a HELPER VOI covering the dose grid with
    ``apply_overlap=None``, so overlap priorities do not cut voxels out of it.

    Parameters
    ----------
    model : str or tuple
        Model identifier passed to :func:`pyRadPlan.ai.modelhub.load_model`: a
        HuggingFace model name, a local directory path, or a preloaded
        ``(model, preprocessor)`` tuple used directly.
    device : str, optional
        Device to run the model on, e.g. ``"cuda"`` or ``"cpu"``. Defaults to following
        :attr:`XpSettings.prefer_gpu`.
    monitor_only : bool
        When *True* the objective returns 0 to the optimiser (and a zero
        gradient) but still logs the predicted probability.
    warmup_iterations : int
        Number of gradient evaluations before the CNN gradient is activated.
        During warmup the probability is logged but a zero gradient is returned.
    cst_masks : list[str], optional
        VOI names for the mask channels. Ordering matters unless the model's
        config collapses the masks into a single channel.
    num_of_fractions : int, optional
        For models trained on total-course dose: the per-fraction dose of the
        optimizer is multiplied by this before it reaches the model. ``None``
        passes the dose unchanged.
    """

    # -- Objective metadata --
    name: Literal["Outcome CNN"] = "Outcome CNN"
    has_hessian: ClassVar[bool] = False

    # -- User-facing fields --
    model: str = Field(description="Model name, local directory, or preloaded objects.")
    device: Optional[str] = Field(
        default=None,
        description="Device to run the model on (e.g. 'cuda'); None follows xp.prefer_gpu.",
    )
    monitor_only: bool = Field(default=False)
    warmup_iterations: int = Field(default=0, ge=0)
    cst_masks: Optional[list[str]] = Field(default=None)
    num_of_fractions: Optional[int] = Field(default=None, gt=0)
    ntcp_history: list[float] = Field(default_factory=list, exclude=True)

    # -- Internal state (excluded from serialisation) --
    nn_model: Any = Field(default=None, exclude=True)
    preprocessor: Any = Field(default=None, exclude=True)
    iteration_count: int = Field(default=0, exclude=True)

    _dose_dims: Optional[tuple] = PrivateAttr(default=None)
    # VOI voxel indices on the dose grid; None when the VOI covers the whole grid.
    _dose_index: Any = PrivateAttr(default=None)
    _dose_n: Optional[int] = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _init(cls, data: Any) -> Any:
        """Load model + preprocessor and configure it with CT / structures."""
        if not isinstance(data, dict):
            return data
        if torch is None:
            raise RuntimeError("PyTorch is required for OutcomeCNN.")

        # Already initialised (copy / re-validation).
        if data.get("preprocessor") is not None:
            return data

        ct = data.pop("ct", None)
        cst = data.pop("cst", None)

        # Without planning data we cannot configure the preprocessor (e.g.
        # deserialisation). Leave it uninitialised; _check_ready() guards use.
        if ct is None or cst is None:
            return data

        spec = data.get("model")
        nn_model, preprocessor = cls._resolve_model(spec, data.get("device"))
        if not isinstance(spec, str):
            # Preloaded objects — keep the serialised field a plain string.
            data["model"] = "<preloaded>"

        # Read back where the model actually ended up rather than trusting the request,
        # so the preprocessor follows it even when placement came from the settings.
        model_device = next(nn_model.parameters()).device
        preprocessor.configure(ct, cst, cst_masks=data.get("cst_masks"), device=model_device)

        data["nn_model"] = nn_model
        data["preprocessor"] = preprocessor
        return data

    @staticmethod
    def _resolve_model(spec: Any, device: Optional[str] = None) -> tuple[Any, Any]:
        """Resolve ``model`` into an ``(nn_model, preprocessor)`` pair."""
        if isinstance(spec, (tuple, list)) and len(spec) == 2 and hasattr(spec[0], "forward"):
            nn_model, preprocessor = spec[0], spec[1]
            if device is not None and hasattr(nn_model, "to"):
                nn_model.to(device)
            return nn_model, preprocessor
        target = device if device is not None else _default_model_device()
        if isinstance(spec, str):
            if Path(spec).is_dir():
                return load_model(local_dir=spec, device=target)
            return load_model(name=spec, device=target)
        raise TypeError(
            "OutcomeCNN 'model' must be a model name, a local directory path, "
            "or a preloaded (model, preprocessor) tuple."
        )

    def __deepcopy__(self, memo=None):
        """Share the model and preprocessor: torch modules and autograd tensors don't deepcopy."""
        return self.model_copy(update={"ntcp_history": list(self.ntcp_history)})

    # ------------------------------------------------------------------
    # Objective interface
    # ------------------------------------------------------------------

    def preprocess_image_reference_parameters(self, target_grid, index_list=None) -> None:
        """Wire the optimisation dose grid into the preprocessor.

        Called once by the planning problem for every objective with
        ``target_grid=dij.dose_grid``; this is where the CNN objective learns the
        grid its dose vector lives on.
        """
        super().preprocess_image_reference_parameters(target_grid, index_list)
        if self.preprocessor is not None:
            self.preprocessor.set_dose_grid(target_grid)
            self._dose_dims = tuple(int(d) for d in target_grid.dimensions)
            self._dose_n = int(np.prod(self._dose_dims))
            # The optimizer hands us only the voxels inside this objective's VOI
            # (index_list); remember them so we can scatter onto the full dose grid
            # the CNN needs and gather the gradient back.
            covers_grid = index_list is None or np.size(index_list) == self._dose_n
            self._dose_index = None if covers_grid else np.asarray(index_list)

    def compute_objective(self, dose_values):
        """Compute the NTCP (0.0 when *monitor_only*).

        Returns a 0-d array in the namespace and device of ``dose_values``, as the other
        objectives do: the model itself yields a Python float, and backends such as cupy
        reject scalars where the caller stacks the per-scenario values.
        """
        self._check_ready()

        xp = array_api_compat.array_namespace(dose_values)
        prob = self._forward_no_grad(self._to_full_grid(dose_values, xp))
        self.ntcp_history.append(prob)

        value = 0.0 if self.monitor_only else prob
        return xp.asarray(
            value, dtype=dose_values.dtype, device=array_api_compat.device(dose_values)
        )

    def compute_gradient(self, dose_values):
        """Compute the gradient of the NTCP w.r.t. dose.

        Returns a 1-D array matching ``dose_values`` in the same array namespace
        and device as the input.
        """
        self._check_ready()

        xp = array_api_compat.array_namespace(dose_values)
        self.iteration_count += 1
        n = array_api_compat.size(dose_values)

        # Match what the real gradient path below returns (via _from_full_grid), so the
        # shortcut cannot hand back an array on a different device or in a different dtype.
        zeros_like_input = {
            "dtype": dose_values.dtype,
            "device": array_api_compat.device(dose_values),
        }

        if self.monitor_only:
            return xp.zeros(n, **zeros_like_input)

        if self.iteration_count <= self.warmup_iterations:
            if self.iteration_count == self.warmup_iterations:
                logger.info(
                    "Outcome CNN warmup complete after %d iterations. Activating gradient.",
                    self.warmup_iterations,
                )
            return xp.zeros(n, **zeros_like_input)

        inputs = self.preprocessor.preprocess(
            self._to_full_grid(dose_values, xp), requires_grad=True
        )
        out = self.nn_model(*inputs)
        prob = self.preprocessor.postprocess(out)
        prob.backward()

        grad = self.preprocessor.gradient_to_dose_grid()
        if self.num_of_fractions is not None:
            grad = grad * self.num_of_fractions

        # gradient_to_dose_grid returns the full dose grid; restrict back to the
        # objective's VOI voxels so it matches the optimizer's input vector.
        return self._from_full_grid(xp_utils.to_namespace(xp, grad), dose_values, xp)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def report_info(self) -> dict:
        """Return NTCP history collected during optimization."""
        if self.ntcp_history:
            return {"ntcp_history": list(self.ntcp_history)}
        return {}

    def reset_history(self) -> None:
        """Clear history and iteration counter."""
        self.ntcp_history.clear()
        self.iteration_count = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_ready(self) -> None:
        if self.preprocessor is None or self.nn_model is None:
            raise RuntimeError("OutcomeCNN not initialised — provide ct and cst at construction.")

    def _to_full_grid(self, dose_values, xp):
        """Scatter VOI-restricted dose onto the full dose grid (zeros elsewhere), as model dose."""
        if self._dose_index is not None:
            device = array_api_compat.device(dose_values)
            full = xp.zeros(self._dose_n, dtype=dose_values.dtype, device=device)
            full[xp.asarray(self._dose_index, device=device)] = dose_values
            dose_values = full
        if self.num_of_fractions is not None:
            dose_values = dose_values * self.num_of_fractions
        return dose_values

    def _from_full_grid(self, grad_full, dose_values, xp):
        """Gather a full-grid gradient back to the objective's VOI voxels."""
        if self._dose_index is None:
            return grad_full
        return grad_full[xp.asarray(self._dose_index, device=array_api_compat.device(dose_values))]

    def _forward_no_grad(self, dose_values) -> float:
        """Run a forward pass without gradient tracking."""
        inputs = self.preprocessor.preprocess(dose_values, requires_grad=False)
        with torch.no_grad():
            out = self.nn_model(*inputs)
            return float(self.preprocessor.postprocess(out).item())
