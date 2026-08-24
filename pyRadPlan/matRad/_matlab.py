from ._engine import MatRadEngine


class MatRadEngineMatlab(MatRadEngine):
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

    def __init__(self, matRadPath="./matRad"):
        import matlab.engine

        self.matRadPath = matRadPath
        self.__engine = matlab.engine.start_matlab()
        self.__engineVersionStr = self.engine.version()
        self.__engine.addpath(self.engine.genpath(self.matRadPath), nargout=0)
        self.__engine.matRad_rc
        self.__matRadVersionStr = self.engine.matRad_version()
        # print(self.matRadVersionStr)
