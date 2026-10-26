"""Outcome objective backed by a differentiable torch model from the model hub."""

from pathlib import Path
from typing import Any, Literal, Optional

import array_api_compat
from pydantic import Field, model_validator

from pyRadPlan.core import xp_utils
from pyRadPlan.ai.modelhub import load_model

from ._outcome import OutcomeObjective


def _torch():
    torch = xp_utils.import_backend("torch")
    if torch is None:
        raise RuntimeError("PyTorch is required for OutcomeModel.")
    return torch


class OutcomeModel(OutcomeObjective):
    """Outcome objective using a torch model and its :class:`OutcomePreprocessor`.

    Any differentiable model works (CNN, MLP, ...); model-specific input handling lives
    in the preprocessor shipped with the model.

    Parameters
    ----------
    model : str or tuple
        HuggingFace model name, local model directory, or a preloaded
        ``(model, preprocessor)`` tuple.
    device : str, optional
        Device to run the model on. Defaults to following :attr:`XpSettings.prefer_gpu`.
    cst_masks : list[str], optional
        VOI names passed to the preprocessor as mask channels.
    """

    name: Literal["Outcome Model"] = "Outcome Model"

    model: str = Field(description="Model name, local directory, or preloaded objects.")
    device: Optional[str] = Field(
        default=None,
        description="Device to run the model on (e.g. 'cuda'); None follows xp.prefer_gpu.",
    )
    cst_masks: Optional[list[str]] = Field(default=None)

    nn_model: Any = Field(default=None, exclude=True)
    preprocessor: Any = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="before")
    @classmethod
    def _load(cls, data: Any) -> Any:
        """Load model and preprocessor."""
        if not isinstance(data, dict) or data.get("preprocessor") is not None:
            return data

        spec = data.get("model")
        try:
            nn_model, preprocessor = cls._resolve_model(spec, data.get("device"))
        except OSError as exc:  # a ValueError lets pydantic report it as a validation error
            raise ValueError(f"Cannot load outcome model {spec!r}: {exc}") from exc
        nn_model.requires_grad_(False)
        if not isinstance(spec, str):
            data["model"] = "<preloaded>"
        data["nn_model"] = nn_model
        data["preprocessor"] = preprocessor
        return data

    @staticmethod
    def _resolve_model(spec: Any, device: Optional[str] = None) -> tuple[Any, Any]:
        """Resolve ``model`` into an ``(nn_model, preprocessor)`` pair."""
        torch = _torch()
        if isinstance(spec, (tuple, list)) and len(spec) == 2 and hasattr(spec[0], "forward"):
            nn_model, preprocessor = spec
            if device is not None:
                nn_model.to(device)
            return nn_model, preprocessor
        if isinstance(spec, str):
            if device is None:
                device = str(
                    xp_utils.choose_device(array_api_compat.array_namespace(torch.empty(0)))
                )
            if Path(spec).is_dir():
                return load_model(local_dir=spec, device=device)
            return load_model(name=spec, device=device)
        raise TypeError(
            "OutcomeModel 'model' must be a model name, a local directory path, "
            "or a preloaded (model, preprocessor) tuple."
        )

    @property
    def _model_device(self):
        return next(self.nn_model.parameters()).device

    def setup(self, ct, cst) -> None:
        """Configure the preprocessor with the planning CT and structures."""
        self.preprocessor.configure(ct, cst, cst_masks=self.cst_masks, device=self._model_device)

    def _set_dose_grid(self, dose_grid) -> None:
        self.preprocessor.set_dose_grid(dose_grid)

    def _predict(self, dose) -> float:
        with _torch().no_grad():
            return float(self._forward(self._to_torch(dose)))

    def _predict_with_gradient(self, dose) -> tuple[float, Any]:
        torch = _torch()
        leaf = self._to_torch(dose).detach().requires_grad_(True)
        prob = self._forward(leaf)
        (grad,) = torch.autograd.grad(prob, leaf)
        return float(prob.detach()), xp_utils.to_namespace(
            array_api_compat.array_namespace(dose), grad
        )

    def _forward(self, dose_tensor):
        inputs = self.preprocessor.preprocess(dose_tensor)
        return self.preprocessor.postprocess(self.nn_model(*inputs))

    def _to_torch(self, dose):
        torch = _torch()
        return xp_utils.to_namespace(torch, dose).to(
            device=self._model_device, dtype=torch.float32
        )
