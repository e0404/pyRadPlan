"""CNN-based outcome objective (NTCP / TCP).

Uses a pre-trained CNN loaded via the model importer to predict an outcome
probability and compute gradients w.r.t. dose for use in optimisation.
All model-specific parameters (dimensions, spacing, normalisation, input
ordering) are read from the JSON config shipped with the model weights.

Example
-------
>>> ntcp_obj = OutcomeCNN(
...     priority=10000.0,
...     model="densenet_example",
...     mode="NTCP",
...     warmup_iterations=5,
...     ct=ct,
...     cst=cst,
...     stf=stf,
...     cst_masks=["parotid_lt", "parotid_rt"],
... )
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Literal, Optional
import logging

import numpy as np
from pydantic import Field, model_validator

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from pyRadPlan.core import Grid
from pyRadPlan.optimization.objectives._objective import Objective
from pyRadPlan.data.ml_models.pytorch_model_importer import load_model
from .preprocessor._outcome_cnn_preprocessor import OutcomeCnnPreprocessor

logger = logging.getLogger(__name__)


class OutcomeCNN(Objective):
    """CNN-based outcome objective (NTCP or TCP).

    Parameters
    ----------
    model : str
        Name of the model folder (e.g. ``"densenet_example"``).
    mode : ``"NTCP"`` or ``"TCP"``
        ``"NTCP"`` minimises the predicted probability;
        ``"TCP"`` maximises it (i.e. minimises the negative).
    monitor_only : bool
        When *True* the objective returns 0 to the optimiser (and a zero
        gradient) but still logs the predicted probability.
    warmup_iterations : int
        Number of gradient evaluations before the CNN gradient is
        activated.  During warmup the probability is logged but a zero
        gradient is returned.
    cst_masks : list[str], optional
        VOI names for the mask channels.  If the config sets
        ``collaps: true`` the masks are collapsed into one channel;
        otherwise the ordering matters.
    """

    # -- Objective metadata --
    name: ClassVar[str] = "Outcome CNN"
    has_hessian: ClassVar[bool] = False

    # -- User-facing fields --
    model: str = Field(description="Model folder name")
    mode: Literal["NTCP", "TCP"] = Field(default="NTCP")
    monitor_only: bool = Field(default=False)
    warmup_iterations: int = Field(default=0, ge=0)
    cst_masks: Optional[list[str]] = Field(default=None)

    # -- Internal state (excluded from serialisation) --
    nn_model: Any = Field(default=None, exclude=True)
    loaded_config: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    preprocessor: Optional[OutcomeCnnPreprocessor] = Field(default=None, exclude=True)
    iteration_count: int = Field(default=0, exclude=True)
    ntcp_history: list[float] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _init_model_and_preprocessor(cls, data: Any) -> Any:
        """Load model + config and prepare the preprocessor."""
        if torch is None:
            raise RuntimeError("PyTorch is required for OutcomeCNN.")

        # Skip if already initialised (copy / re-validation)
        if data.get("preprocessor") is not None:
            return data

        # Pop non-field init arguments
        ct = data.pop("ct", None)
        cst = data.pop("cst", None)
        data.pop("stf", None)  # reserved for future use

        if ct is None or cst is None:
            return data  # partial init (e.g. deserialisation)

        model_name = data.get("model", "densenet_example")
        mode = data.get("mode", "NTCP")
        cst_masks = data.get("cst_masks")

        # TODO: Better NTCP as default.
        # Category derived from mode
        category = "ntcp" if mode == "NTCP" else "tcp"

        # Load CNN + JSON config via the model importer
        nn_model, config = load_model(model_name, category=category)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nn_model = nn_model.to(device)
        nn_model.eval()

        data["nn_model"] = nn_model
        data["loaded_config"] = config

        # Build preprocessor from config
        preprocessor = OutcomeCnnPreprocessor(config=config, device=device)
        preprocessor.prepare(ct, cst, cst_masks=cst_masks)
        data["preprocessor"] = preprocessor

        return data

    # ------------------------------------------------------------------
    # Objective interface
    # ------------------------------------------------------------------

    def compute_objective(self, dose_values: np.ndarray) -> float:
        """Compute the outcome probability.

        Returns
        -------
        float
            Probability (NTCP) or its negative (TCP).
            Returns ``0.0`` when *monitor_only* is active.
        """
        self._check_ready()

        prob = self._forward_no_grad(dose_values)

        # TODO: Why not above prob?
        if self.monitor_only:
            return 0.0

        return -prob if self.mode == "TCP" else prob

    def compute_gradient(self, dose_values: np.ndarray) -> np.ndarray:
        """Compute the gradient of the outcome probability w.r.t. dose.

        Returns
        -------
        np.ndarray
            1-D gradient array (Fortran order) matching ``dose_values``.
        """
        self._check_ready()

        self.iteration_count += 1

        # Warmup / monitor-only: log probability, return zero gradient
        if self.monitor_only or self.iteration_count <= self.warmup_iterations:
            prob = self._forward_no_grad(dose_values)
            self.ntcp_history.append(prob)
            if not self.monitor_only and self.iteration_count == self.warmup_iterations:
                logger.info(
                    "Outcome CNN warmup complete after %d iterations. Activating gradient.",
                    self.warmup_iterations,
                )
            return np.zeros(dose_values.size, dtype=np.float32)

        # Build input with gradient tracking on dose
        input_tensor, dose_leaf = self.preprocessor.build_input_tensor(
            dose_values, requires_grad=True
        )

        # Forward pass
        logit = self.nn_model(input_tensor)
        prob = torch.sigmoid(logit).sum()

        # Log
        prob_val = prob.detach().item()
        self.ntcp_history.append(prob_val)

        # Backward pass — gradient flows through grid_sample to dose_leaf
        prob.backward()

        # dose_leaf.grad is (Z, Y, X) → transpose to (X, Y, Z) → Fortran flatten
        grad_zyx = dose_leaf.grad.detach().cpu().numpy()
        grad_xyz = np.transpose(grad_zyx, (2, 1, 0))

        if self.mode == "TCP":
            grad_xyz = -grad_xyz

        return grad_xyz.ravel(order="F").astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def set_dose_grid(self, dose_grid: Grid) -> None:
        """Update the dose grid (recomputes sampling coordinates)."""
        self._check_ready()
        self.preprocessor.set_dose_grid(dose_grid)

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

    def _forward_no_grad(self, dose_values: np.ndarray) -> float:
        """Run a forward pass without gradient tracking."""
        input_tensor = self.preprocessor.build_input_tensor(dose_values, requires_grad=False)
        with torch.no_grad():
            logit = self.nn_model(input_tensor)
            return torch.sigmoid(logit.view(-1)[0]).item()
