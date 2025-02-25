from abc import ABC,abstractmethod
from typing import ClassVar, Union
import numpy as np
from ..scalarization_strategies import ScalarizationStrategyBase, get_scalarization_strategy
class TradeoffStrategyBase(ABC):
    """
    To be written later
    Abstract class for tradeoff exploration methods

    Parameters
    -----------

    Attributes
    ----------
    """

    short_name: ClassVar[str]
    name: ClassVar[str]
    scalarization_strategy: ScalarizationStrategyBase

    def __init__(self,callbacks: dict[str, callable],scalarization_desc: Union[str,dict]):
        self.callbacks = callbacks
        self.scalarization_strategy = scalarization_desc

    def solve(self,x: np.ndarray[float]) -> list[np.ndarray[float]]:
        return self._solve(x)
    
    def _initialize(self):
        self.scalarization_strategy = get_scalarization_strategy(self.scalarization_strategy,self.callbacks)#TODO: Pass options

    
    @abstractmethod
    def _solve(self,x: np.ndarray[float]) -> list[np.ndarray[float]]:
        pass