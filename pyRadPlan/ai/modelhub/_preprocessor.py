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
