import os
from abc import ABC
from abc import abstractmethod


class MatRadEngine(ABC):
    # TODO: missing description
    @property
    def matRadPath(self):
        return self._matRadPath

    @matRadPath.setter
    def matRadPath(self, path):
        # Check if valid matrad path
        isdir = os.path.isdir(path)
        if not isdir:
            raise ValueError("matRad not found")

        self._matRadPath = path

    @property
    @abstractmethod
    def matRadVersionStr(self):
        pass

    @property
    @abstractmethod
    def engine(self):
        pass

    @property
    @abstractmethod
    def engineVersionStr(self):
        pass

    def __getattr__(self, name):
        return getattr(self.engine, name)
