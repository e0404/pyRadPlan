from typing import Annotated

from pydantic import Field

from ._base import DoseEngineBase


class MonteCarloEngineAbstract(DoseEngineBase):
    """Abstract base for Monte Carlo dose calculation engines."""

    num_histories_per_beamlet: Annotated[float, Field(gt=0.0)] = 2e2
    num_histories_direct: Annotated[float, Field(gt=0.0)] = 1e6
