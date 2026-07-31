"""CNN-based outcome objective (NTCP / TCP).

Uses a pre-trained CNN loaded via :func:`pyRadPlan.ai_models.load_model` to
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

from pathlib import Path
from typing import Any, ClassVar, Literal, Optional
import logging

import array_api_compat
from pydantic import Field, PrivateAttr, model_validator

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from pyRadPlan.core import xp_utils
from pyRadPlan.ai_models import load_model
from pyRadPlan.optimization.objectives._objective import Objective

logger = logging.getLogger(__name__)


class OutcomeCNN(Objective):
    """CNN-based outcome objective (NTCP or TCP).

    Parameters
    ----------
    model : str or tuple
        Model identifier passed to :func:`pyRadPlan.ai_models.load_model`: a
        HuggingFace model name, a local directory path, or a preloaded
        ``(model, preprocessor)`` tuple used directly.
    mode : ``"NTCP"`` or ``"TCP"``
        ``"NTCP"`` minimises the predicted probability; ``"TCP"`` maximises it
        (i.e. minimises the negative).
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
        nn_model, preprocessor = cls._resolve_model(spec)
        if not isinstance(spec, str):
            # Preloaded objects — keep the serialised field a plain string.
            data["model"] = "<preloaded>"

        device = next(nn_model.parameters()).device
        preprocessor.configure(ct, cst, cst_masks=data.get("cst_masks"), device=device)

        data["nn_model"] = nn_model
        data["preprocessor"] = preprocessor
        return data

    @staticmethod
    def _resolve_model(spec: Any) -> tuple[Any, Any]:
        """Resolve ``model`` into an ``(nn_model, preprocessor)`` pair."""
        if isinstance(spec, (tuple, list)) and len(spec) == 2 and hasattr(spec[0], "forward"):
            return spec[0], spec[1]
        if isinstance(spec, str):
            if Path(spec).is_dir():
                return load_model(local_dir=spec)
            return load_model(name=spec)
        raise TypeError(
            "OutcomeCNN 'model' must be a model name, a local directory path, "
            "or a preloaded (model, preprocessor) tuple."
        )

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

    def compute_objective(self, dose_values) -> float:
        """Compute the outcome probability (0.0 when *monitor_only*)."""
        self._check_ready()

        prob = self._forward_no_grad(dose_values)
        self.ntcp_history.append(prob)

        if self.monitor_only:
            return 0.0
        return -prob if self.mode == "TCP" else prob

    def compute_gradient(self, dose_values):
        """Compute the gradient of the outcome probability w.r.t. dose.

        Returns a 1-D array matching ``dose_values`` in the same array namespace
        and device as the input.
        """
        self._check_ready()

        xp = array_api_compat.array_namespace(dose_values)
        self.iteration_count += 1
        n = array_api_compat.size(dose_values)

        if self.monitor_only:
            return xp.zeros(n, dtype=xp.float32)

        if self.iteration_count <= self.warmup_iterations:
            if self.iteration_count == self.warmup_iterations:
                logger.info(
                    "Outcome CNN warmup complete after %d iterations. Activating gradient.",
                    self.warmup_iterations,
                )
            return xp.zeros(n, dtype=xp.float32)

        inputs = self.preprocessor.preprocess(dose_values, requires_grad=True)
        out = self.nn_model(*inputs)
        prob = self.preprocessor.postprocess(out)
        prob.backward()

        grad = self.preprocessor.gradient_to_dose_grid()
        if self.mode == "TCP":
            grad = -grad

        if self.debug_plots is not None:
            self._save_grad_plot(grad)

        return xp_utils.to_namespace(xp, grad)

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
    # ------------------------------------------------------------------

    def _check_ready(self) -> None:
        if self.preprocessor is None or self.nn_model is None:
            raise RuntimeError("OutcomeCNN not initialised — provide ct and cst at construction.")

    def _forward_no_grad(self, dose_values) -> float:
        """Run a forward pass without gradient tracking."""
        inputs = self.preprocessor.preprocess(dose_values, requires_grad=False)
        with torch.no_grad():
            out = self.nn_model(*inputs)
            return float(self.preprocessor.postprocess(out).item())

    def _save_grad_plot(self, grad) -> None:
        """Save a mid-axial slice of the dose-grid gradient to ``debug_plots``."""
        try:
            import matplotlib  # noqa: PLC0415

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError:
            logger.warning("matplotlib not available; skipping debug_plots.")
            return

        if self._dose_dims is None:
            logger.warning("Dose grid dimensions unknown; skipping debug plot.")
            return

        grad_np = xp_utils.to_numpy(grad).reshape(self._dose_dims, order="F")
        z = grad_np.shape[2] // 2

        out_dir = Path(self.debug_plots)
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots()
        im = ax.imshow(grad_np[:, :, z].T, origin="lower")
        fig.colorbar(im, ax=ax)
        ax.set_title(f"NTCP gradient (iter {self.iteration_count}, z={z})")
        fig.savefig(out_dir / f"grad_{self._debug_index:04d}.png", dpi=100)
        plt.close(fig)
        self._debug_index += 1
