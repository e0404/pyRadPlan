"""Base class for model preprocessors.

A preprocessor is constructed by the loader with the ``model_preprocessing``
section of the model repository's ``model_config.json``.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BasePreprocessor(ABC):
    """Abstract base class for model preprocessors.

    Concrete preprocessors are shipped in a model repository's
    ``preprocessor.py`` and instantiated by the loader with the
    ``model_preprocessing`` section of ``model_config.json``. Implementations
    must override :meth:`preprocess`.

    How the model is then run is up to the code using it, so a task can require more.
    A dose-calculation preprocessor (``AIBeamletEngine``) must also define
    ``predict(model, model_input)``, which runs the model on the output of
    :meth:`preprocess` and returns ``{quantity: array}``. The base class has no
    default for it, since running a model is framework-specific (e.g. torch's
    ``inference_mode``).

    Parameters
    ----------
    config : dict, optional
        Preprocessing configuration, typically the ``model_preprocessing``
        section of ``model_config.json``. Stored on ``self.config``.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config: dict = config or {}

    @abstractmethod
    def preprocess(self, inputs: Any) -> Any:
        """Transform raw inputs into model-ready tensors.

        Parameters
        ----------
        inputs : Any
            Model-specific inputs. The shape/type contract is defined by the
            concrete preprocessor and the model it serves.

        Returns
        -------
        Any
            The preprocessed input(s) ready to be passed to the model.
        """

    def postprocess(self, outputs: Any) -> Any:
        """Optionally transform raw model outputs.

        The default implementation returns ``outputs`` unchanged. Override when
        a model's output needs to be mapped back to physical quantities.

        Parameters
        ----------
        outputs : Any
            Raw output produced by the model.

        Returns
        -------
        Any
            The post-processed output.
        """
        return outputs

    def __call__(self, inputs: Any) -> Any:
        """Alias for :meth:`preprocess`."""
        return self.preprocess(inputs)


class OutcomePreprocessor(BasePreprocessor):
    """Preprocessor contract for outcome models used as optimization objectives.

    The objective calls ``postprocess(model(*preprocess(dose)))`` and differentiates the
    result w.r.t. ``dose``, so :meth:`preprocess` must use differentiable torch ops.
    """

    @abstractmethod
    def configure(self, ct: Any, cst: Any, cst_masks: Optional[list[str]] = None, device=None):
        """Prepare the static inputs (e.g. CT and masks) on ``device``."""

    @abstractmethod
    def set_dose_grid(self, dose_grid: Any) -> None:
        """Set the grid the optimization dose lives on."""

    @abstractmethod
    def preprocess(self, inputs: Any) -> Any:
        """Map the dose on the dose grid to the model inputs.

        ``inputs`` is a flat float32 torch tensor on the model device with one value per
        voxel of the dose grid, in the order of the C-order flattening of the ``(z, y, x)``
        array (x runs fastest), i.e. of ``sitk.GetArrayFromImage(image).reshape(-1)`` and of
        ``VOI.scenario_indices(order="numpy")``. Returns the tuple of positional model inputs.
        """
