from ._base import DoseEngineBase


class MonteCarloEngineAbstract(DoseEngineBase):
    def __init__(self, pln: dict):
        self.num_histories_per_beamlet: float = 2e2
        self.num_histories_direct: float = 1e6
        super().__init__(pln)
