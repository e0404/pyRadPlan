"""MatRad engine interfaces for MATLAB and Octave."""

from ._matlab import MatRadEngineMatlab
from ._octave import MatRadEngineOctave
from ._engine import MatRadEngine

engine = None


def setEngine(engine_=None):
    global engine
    if isinstance(engine, MatRadEngine) or engine is None:
        engine = engine_
    else:
        raise ValueError("Invalid matRad engine")

    print("Engine set to " + engine.engineVersionStr + " using " + engine.matRadVersionStr)


__all__ = ["engine", "setEngine", "MatRadEngineMatlab", "MatRadEngineOctave"]
