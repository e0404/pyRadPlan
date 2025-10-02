import numpy as np
from ._base_tradeoff_strategies import TradeoffStrategyBase


class SinglePlan(TradeoffStrategyBase):
    name = "Single Plan Tradeoff Strategy"
    short_name = "single"
    scalarization_strategy = "WeightedSum"

    def _solve(self, x: np.ndarray[float]) -> list[np.ndarray[float]]:
        return self.scalarization_strategy.solve(x)
