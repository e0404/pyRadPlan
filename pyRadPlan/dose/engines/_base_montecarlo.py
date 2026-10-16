import logging
from typing import Annotated, Any, Optional

import numpy as np
from pydantic import Field

from ...biology.models import BiologicalModel, BioModelUnavailableError
from ._base import DoseEngineBase

logger = logging.getLogger(__name__)


class MonteCarloEngineAbstract(DoseEngineBase):
    """Abstract base for Monte Carlo dose calculation engines."""

    num_histories_per_beamlet: Annotated[float, Field(gt=0.0)] = 2e2
    num_histories_direct: Annotated[float, Field(gt=0.0)] = 1e6


class ScoredLETBioMixin:
    """
    Biological models of a Monte Carlo engine that scores the dose-averaged LET itself.

    The engine does not read the machine's pencil-beam kernels; the biological influence
    matrices are derived from the scored LET with
    :func:`~pyRadPlan.biology.models.bio_influence_from_let`, so only LET-based evaluators are
    supported. An engine using it implements :meth:`_request_let_scoring` and has
    ``calc_bio_dose`` / ``calc_let`` settings.
    """

    def _request_let_scoring(self) -> None:
        """Make the simulation score the LET."""
        raise NotImplementedError

    def _scores_let(self) -> bool:
        """Whether the engine scores LET for the machine's radiation mode."""
        modes = self.bio_scored_radiation_modes
        mode = getattr(getattr(self, "_machine", None), "radiation_mode", None)
        return modes is None or mode is None or mode in modes

    def _allocate_bio_matrices(self, dij: dict[str, Any], names: list[str]) -> dict[str, Any]:
        """Allocate the LET and biological influence containers."""
        return self._allocate_quantity_matrices(dij, names)

    def _unsupported_evaluator_reason(self, model, evaluator) -> Optional[str]:
        """Why the engine cannot drive ``evaluator``, or ``None`` if it can."""
        # bio_influence_from_let supplies alpha_x, beta_x, physical_dose and LET only; the
        # engine has no pencil-beam kernels to interpolate any further model input from.
        if evaluator.kernel_field_names:
            names = ", ".join(evaluator.kernel_field_names)
            return (
                f"the evaluator of {model!r} additionally requires the machine kernel inputs "
                f"{names}, which {self.name} cannot supply"
            )
        if not evaluator.influence_quantity_names:
            return f"the evaluator of {model!r} declares no biological influence quantities"
        return None

    def _unsupported_bio_reason(self, model, evaluator) -> Optional[str]:
        """Why the engine cannot evaluate ``model``, or ``None`` if it can."""
        if model is None:
            return None

        # Judge an existing evaluator by what it actually needs and produces rather than by
        # the model's intrinsic output declaration: an evaluator may build the additive
        # influence quantities directly, without declaring any intrinsic outputs.
        if evaluator is not None:
            return self._unsupported_evaluator_reason(model, evaluator)

        # No evaluator was built: either nothing biological was asked of the model, or it is
        # not LET-based and therefore outside what the scored LET can be turned into.
        if model.output_quantities and not model.requires("let"):
            return (
                f"{self.name} derives biological influence matrices from the scored LET and "
                f"therefore only supports LET-based evaluators, not {model!r}"
            )
        return None

    def _init_bio_model(self, dij: dict[str, Any]) -> dict[str, Any]:
        """Configure biological and LET outputs without constructing unsupported evaluators."""
        model = self.bio_model if isinstance(self.bio_model, BiologicalModel) else None
        model_requires_let = model is not None and model.requires("let")
        self._bio_evaluator = None
        self._bio_influence_names = ()

        evaluator = None
        if model_requires_let and self.calc_bio_dose is not False:
            try:
                evaluator = self._create_bio_evaluator(
                    model,
                    {"alpha_x": np.asarray(dij["alphax"]), "beta_x": np.asarray(dij["betax"])},
                )
            except BioModelUnavailableError as exc:
                if self.calc_bio_dose is True:
                    raise
                self._drop_bio_model(dij, model, str(exc))
                model, model_requires_let = None, False
        reason = self._unsupported_bio_reason(model, evaluator)
        supported = reason is None and evaluator is not None
        if supported:
            self._bio_evaluator = evaluator
            self._bio_influence_names = evaluator.influence_quantity_names

        if self.calc_bio_dose is True and not supported:
            raise NotImplementedError(
                reason
                or f"{self.name} cannot compute biological influence matrices for "
                f"{model!r} (no biological model requiring LET was selected)."
            )
        if self.calc_bio_dose == "auto" and reason is not None:
            logger.warning(
                "%s; influence matrices are skipped (set calc_bio_dose=False to silence this).",
                reason,
            )
        self._calc_bio_dose, self._calc_let, self._use_let_kernel = self._resolve_quantity_flags(
            self.calc_bio_dose if supported else False,
            self.calc_let,
            bio_influence_quantities=self._bio_influence_names,
            let_available=self._scores_let(),
            let_auto=model_requires_let,
        )
        if self._calc_bio_dose:
            self._validate_bio_influence_names(dij, self._bio_influence_names)
            dij = self._allocate_bio_matrices(dij, list(self._bio_influence_names))

        if self._use_let_kernel:
            self._request_let_scoring()
        if self._calc_let:
            dij = self._allocate_bio_matrices(dij, ["let_dose"])

        return dij
