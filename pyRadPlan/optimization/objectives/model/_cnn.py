"""CNN-based outcome objective (NTCP / TCP).

Uses a pre-trained CNN loaded via :func:`pyRadPlan.ai.modelhub.load_model` to
predict an outcome probability and compute its gradient w.r.t. dose for use in
optimisation. All model-specific behaviour (grid geometry, cropping, CT/mask
resampling, dose normalisation, differentiable interpolation) lives in the
preprocessor that ships *with* the model on HuggingFace, so this objective stays
thin and backend-agnostic (numpy / cupy / torch via ``xp_utils``).

The shipped preprocessor is expected to expose:

- ``configure(ct, cst, cst_masks=None, device=None)`` — one-time CT/mask setup.
- ``set_dose_grid(grid)`` — precompute dose<->model sampling coordinates.
- ``preprocess(dose, requires_grad=False)`` — model inputs, called as
  ``model(*inputs)``; with ``requires_grad`` the model-grid dose is the autograd
  leaf.
- ``postprocess(outputs)`` — map the model output to a scalar probability.
- ``gradient_to_dose_grid()`` — map ``dose_leaf.grad`` back onto the dose grid.

Example
-------
>>> ntcp_obj = OutcomeCNN(
...     priority=10000.0,
...     model="pyRadPlan-outcome-ORPDenseNet-tg119",  # HF name, local dir, or (model, preprocessor)
...     mode="NTCP",
...     warmup_iterations=5,
...     ct=ct,
...     cst=cst,
...     cst_masks=["Core"],
... )
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional
import logging

import array_api_compat
import numpy as np
from pydantic import Field, PrivateAttr, model_validator
from pydantic_core import PydanticUndefined

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
    """CNN-based outcome objective (NTCP or TCP).

    Parameters
    ----------
    model : str or tuple
        Model identifier passed to :func:`pyRadPlan.ai.modelhub.load_model`: a
        HuggingFace model name, a local directory path, or a preloaded
        ``(model, preprocessor)`` tuple used directly.
    mode : ``"NTCP"`` or ``"TCP"``
        ``"NTCP"`` minimises the predicted probability; ``"TCP"`` maximises it
        (i.e. minimises the negative).
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
    debug_plots : str, optional
        Directory to save per-iteration gradient debug plots (requires
        matplotlib).
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
    mode: Literal["NTCP", "TCP"] = Field(default="NTCP")
    monitor_only: bool = Field(default=False)
    warmup_iterations: int = Field(default=0, ge=0)
    cst_masks: Optional[list[str]] = Field(default=None)
    debug_plots: Optional[str] = Field(
        default=None, description="Directory to save gradient debug plots."
    )
    ntcp_history: list[float] = Field(default_factory=list)

    # -- Internal state (excluded from serialisation) --
    nn_model: Any = Field(default=None, exclude=True)
    preprocessor: Any = Field(default=None, exclude=True)
    iteration_count: int = Field(default=0, exclude=True)

    _dose_dims: Optional[tuple] = PrivateAttr(default=None)
    _debug_index: int = PrivateAttr(default=0)
    # Mapping of the (VOI-restricted) optimization vector onto the full dose grid.
    _dose_index: Any = PrivateAttr(default=None)  # int indices of the objective's VOI (numpy)
    _dose_n: Optional[int] = PrivateAttr(default=None)  # number of dose-grid voxels
    _dose_index_xp: Any = PrivateAttr(default=None)  # cached index in the active namespace
    _dose_index_key: Any = PrivateAttr(default=None)

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
        data.pop("stf", None)  # accepted for convenience, not used
        data.pop("pln", None)  # dose grid is resolved via the optimizer hook

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
        """Deep-copy without copying the loaded model / preprocessor.

        The optimizer deep-copies the whole ``cst`` (e.g. in
        ``apply_overlap_priorities``). The torch model and the preprocessor hold
        modules and autograd-graph tensors that are wasteful (and, for non-leaf
        tensors, impossible) to ``deepcopy``. They are read-only runtime handles,
        so the copy shares them; transient index caches are reset.
        """
        if memo is None:
            memo = {}
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new

        shared = {"nn_model", "preprocessor"}
        new_dict = {
            k: (v if k in shared else copy.deepcopy(v, memo)) for k, v in self.__dict__.items()
        }
        object.__setattr__(new, "__dict__", new_dict)
        object.__setattr__(new, "__pydantic_fields_set__", set(self.__pydantic_fields_set__))
        object.__setattr__(
            new,
            "__pydantic_extra__",
            None
            if self.__pydantic_extra__ is None
            else copy.deepcopy(self.__pydantic_extra__, memo),
        )
        if self.__pydantic_private__ is None:
            object.__setattr__(new, "__pydantic_private__", None)
        else:
            reset = {"_dose_index_xp", "_dose_index_key"}
            private = {
                k: (None if k in reset else copy.deepcopy(v, memo))
                for k, v in self.__pydantic_private__.items()
                if v is not PydanticUndefined
            }
            object.__setattr__(new, "__pydantic_private__", private)
        return new

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
            self._dose_index = None if index_list is None else np.asarray(index_list)
            self._dose_index_xp = None
            self._dose_index_key = None

    def compute_objective(self, dose_values):
        """Compute the outcome probability (0.0 when *monitor_only*).

        Returns a 0-d array in the namespace and device of ``dose_values``, as the other
        objectives do: the model itself yields a Python float, and backends such as cupy
        reject scalars where the caller stacks the per-scenario values.
        """
        self._check_ready()

        xp = array_api_compat.array_namespace(dose_values)
        prob = self._forward_no_grad(self._to_full_grid(dose_values, xp))
        self.ntcp_history.append(prob)

        if self.monitor_only:
            value = 0.0
        else:
            value = -prob if self.mode == "TCP" else prob
        return xp.asarray(
            value, dtype=dose_values.dtype, device=array_api_compat.device(dose_values)
        )

    def compute_gradient(self, dose_values):
        """Compute the gradient of the outcome probability w.r.t. dose.

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
        if self.mode == "TCP":
            grad = -grad

        if self.debug_plots is not None:
            self._save_grad_plot()

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

    def set_dose_grid(self, dose_grid) -> None:
        """Update the dose grid (recomputes sampling coordinates)."""
        self._check_ready()
        self.preprocessor.set_dose_grid(dose_grid)
        self._dose_dims = tuple(int(d) for d in dose_grid.dimensions)

    def reset_history(self) -> None:
        """Clear history and iteration counter."""
        self.ntcp_history.clear()
        self.iteration_count = 0

    def get_ntcp_history(self) -> list[float]:
        """Return a copy of the probability history."""
        return list(self.ntcp_history)

    # ------------------------------------------------------------------
    # Private helpers
    # These are needed for future work where someone passes e.g. only the
    # VOI dose rather than the full dose grid.
    # ------------------------------------------------------------------

    def _check_ready(self) -> None:
        if self.preprocessor is None or self.nn_model is None:
            raise RuntimeError("OutcomeCNN not initialised — provide ct and cst at construction.")

    def _needs_scatter(self, dose_values) -> bool:
        """Whether ``dose_values`` is a VOI subset that must be scattered to the full grid."""
        return (
            self._dose_index is not None
            and self._dose_n is not None
            and array_api_compat.size(dose_values) != self._dose_n
        )

    def _index_on(self, dose_values, xp):
        """VOI indices in the namespace/device of ``dose_values`` (cached).

        The cache key is the namespace *name* and device string (never the module
        object itself, which would break ``model_copy(deep=True)`` on the objective).
        """
        device = array_api_compat.device(dose_values)
        key = (getattr(xp, "__name__", str(xp)), str(device))
        if self._dose_index_key != key:
            self._dose_index_xp = xp.asarray(self._dose_index, device=device)
            self._dose_index_key = key
        return self._dose_index_xp

    def _to_full_grid(self, dose_values, xp):
        """Scatter VOI-restricted dose onto the full dose grid (zeros elsewhere)."""
        if not self._needs_scatter(dose_values):
            return dose_values
        full = xp.zeros(
            self._dose_n, dtype=dose_values.dtype, device=array_api_compat.device(dose_values)
        )
        full[self._index_on(dose_values, xp)] = dose_values
        return full

    def _from_full_grid(self, grad_full, dose_values, xp):
        """Gather a full-grid gradient back to the objective's VOI voxels."""
        if not self._needs_scatter(dose_values):
            return grad_full
        return grad_full[self._index_on(dose_values, xp)]

    def _forward_no_grad(self, dose_values) -> float:
        """Run a forward pass without gradient tracking."""
        inputs = self.preprocessor.preprocess(dose_values, requires_grad=False)
        with torch.no_grad():
            out = self.nn_model(*inputs)
            return float(self.preprocessor.postprocess(out).item())

    def _save_grad_plot(self) -> None:
        """Save a diverging heatmap of the *model-input* dose gradient."""
        try:
            import matplotlib  # noqa: PLC0415

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError:
            logger.warning("matplotlib not available; skipping debug_plots.")
            return

        # Prefer the dedicated model-input gradient; fall back to the leaf grad for
        # preprocessors whose autograd leaf is itself the model-grid dose.
        model_grad = getattr(self.preprocessor, "model_input_grad", None)
        if model_grad is None:
            model_grad = self.preprocessor.dose_leaf.grad
        if model_grad is None:
            logger.warning("No model-input gradient available; skipping debug plot.")
            return

        grad = xp_utils.to_numpy(model_grad).squeeze()  # (mZ, mY, mX)
        mask = xp_utils.to_numpy(self.preprocessor.mask_tensor)[0].max(axis=0)  # (mZ, mY, mX)

        if self.mode == "TCP":
            grad = -grad

        # Axial slice with the most mask (fallback: strongest gradient).
        if mask.any():
            z = int(mask.sum(axis=(1, 2)).argmax())
        else:
            z = int(np.abs(grad).sum(axis=(1, 2)).argmax())

        grad_slice = grad[z]  # (mY, mX)
        mask_slice = mask[z]
        amax = float(np.abs(grad_slice).max()) or 1.0  # symmetric range -> 0 is white

        out_dir = Path(self.debug_plots)
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots()
        im = ax.imshow(grad_slice, cmap="bwr", vmin=-amax, vmax=amax, origin="lower")
        fig.colorbar(im, ax=ax, label="gradient")
        if mask_slice.any():
            ax.contour(mask_slice, levels=[0.5], colors="k", linewidths=0.8)
        my, mx = grad_slice.shape
        ax.set_title(f"Model-input dose gradient (iter {self.iteration_count}, z={z}, {mx}x{my})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.savefig(out_dir / f"grad_{self._debug_index:04d}.png", dpi=120)
        plt.close(fig)
        self._debug_index += 1
