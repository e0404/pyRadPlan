from typing import Annotated

from pydantic import Field

from ._base import DoseEngineBase


class MonteCarloEngineAbstract(DoseEngineBase):
    """Abstract base for Monte Carlo dose calculation engines."""

    num_histories_per_beamlet: Annotated[int, Field(gt=0)] = 10_000
    num_histories_direct: Annotated[int, Field(gt=0)] = 10_000_000
