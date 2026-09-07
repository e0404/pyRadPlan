from ._engine import MatRadEngine
import os


class MatRadEngineOctave(MatRadEngine):
    # TODO: missing description
    @property
    def engine(self):
        return self.__engine

    @property
    def matRadVersionStr(self):
        return self.__matRadVersionStr

    @property
    def engineVersionStr(self):
        return self.__engineVersionStr

    def __init__(
        self,
        matRadPath="./matRad",
        octave_exec="C:/Program Files/GNU Octave/Octave-8.3.0/mingw64/bin/octave-cli.exe",
    ):
        import oct2py

        self.matRadPath = matRadPath
        self.octave_exec = octave_exec

        oct2pyInit = oct2py.Oct2Py()
        self.__engineVersionStr = oct2pyInit.version()

        # from oct2py import octave

        self.__engine = oct2pyInit
        self.__engine.addpath(self.matRadPath)
        self.__engine.matRad_rc
        self.__matRadVersionStr = self.engine.matRad_version()
        # print(self.matRadVersionStr)

    @property
    def octave_exec(self):
        return self._octave_exec

    @octave_exec.setter
    def octave_exec(self, path):
        # Check if executable
        executable = os.access(path, os.X_OK)
        if not executable:
            raise ValueError("Octave executable not found")

        os.environ["OCTAVE_EXECUTABLE"] = path
        self._octave_exec = path
