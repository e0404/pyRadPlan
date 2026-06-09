import array_api_strict as xp
from .lq_models import LQModel
from abc import abstractmethod


class LETBasedLQModel(LQModel):
    """
    Abstract base class for linear-quadratic (LQ) models, which are commonly used
    to describe the relationship between radiation dose and biological effect.

    """

    required_quantities = ["physical_dose", "LET"]  # Requires physical dose and LET information
    default_report_quantity = "RBExDose"  # Suggested quantity for display and planning


class RBEMinMax(LETBasedLQModel):
    """
    Abstract base class for RBE models that incorporate a minimum and maximum RBE value.

    """

    def calc_biological_quantities_for_bixel(self, bixel: dict, kernels: dict) -> dict:
        bixel["v_abr_x"] = bixel["v_alpha_x"] / bixel["v_beta_x"]
        bixel["v_abr_x"] = xp.reshape(xp.asarray(bixel["v_abr_x"]), (-1,))
        bixel = super().calc_biological_quantities_for_bixel(bixel, kernels)
        [RBEmin, RBEmax] = self._get_RBE_min_max(bixel, kernels)
        bixel["alpha"] = RBEmax * xp.reshape(xp.asarray(bixel["v_alpha_x"]), (-1,))
        bixel["beta"] = RBEmin**2 * xp.reshape(xp.asarray(bixel["v_beta_x"]), (-1,))
        return bixel

    @abstractmethod
    def _get_RBE_min_max(self, bixel: dict, kernels: dict) -> tuple[xp.asarray, xp.asarray]:
        """
        Return (RBEmin, RBEmax) arrays of shape (n_depths,).
        Must be implemented by concrete subclasses.
        """


class Wedenberg(RBEMinMax):
    """
    Wedenberg model, which is a specific implementation of the linear-quadratic
    model that incorporates LET dependence.
    (https://www.ncbi.nlm.nih.gov/pubmed/22909391) (accessed on 21/7/2023)
    """

    model = "WED"
    possible_radiation_modes = ["protons"]

    def __init__(self):
        self.p0_WED = 1
        self.p1_WED = 0.434
        self.p2_WED = 1
        super().__init__()

    def _get_RBE_min_max(self, bixel: dict, kernels: dict) -> tuple[float, float]:
        LET = xp.asarray(kernels["let"])
        RBEmax = self.p0_WED + (self.p1_WED * LET) / bixel["v_abr_x"]
        RBEmin = self.p2_WED
        return RBEmin, RBEmax


class MCNamara(RBEMinMax):
    """
    McNamara model, which is a specific implementation of the linear-quadratic
    model that incorporates LET dependence.
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4634882/) (accessed on 21/7/2023)
    """

    model = "MCN"
    possible_radiation_modes = ["protons"]

    def __init__(self):
        self.p0_MCN = 0.999064
        self.p1_MCN = 0.35605
        self.p2_MCN = 1.1012
        self.p3_MCN = -0.0038703
        super().__init__()

    def _get_RBE_min_max(self, bixel: dict, kernels: dict) -> tuple[float, float]:
        LET = xp.asarray(kernels["let"])
        RBEmax = self.p0_MCN + ((self.p1_MCN * LET) / bixel["v_abr_x"])
        RBEmin = self.p2_MCN + (self.p3_MCN * xp.sqrt(bixel["v_abr_x"]) * LET)
        return RBEmin, RBEmax


class Carabe(RBEMinMax):
    """
    Carabe model, which is a specific implementation of the linear-quadratic
    model that incorporates LET dependence.
    (https://www.tandfonline.com/doi/full/10.1080/09553000601087176?journalCode=irab20)% (accessed on 21/7/2023)
    """

    model = "CAR"
    possible_radiation_modes = ["protons"]

    def __init__(self):
        self.p0_CAR = 0.843
        self.p1_CAR = 0.154
        self.p2_CAR = 2.686
        self.p3_CAR = 1.09
        self.p4_CAR = 0.006
        super().__init__()

    def _get_RBE_min_max(self, bixel: dict, kernels: dict) -> tuple[float, float]:
        LET = xp.asarray(kernels["let"])
        RBEmax = self.p0_CAR + ((self.p1_CAR * self.p2_CAR) / bixel["v_abr_x"]) * LET
        RBEmin = self.p3_CAR + ((self.p4_CAR * self.p2_CAR) / bixel["v_abr_x"]) * LET
        return RBEmin, RBEmax


class HeliumMairani(RBEMinMax):
    """
    Mairani model for helium ions, which is a specific implementation of the linear-quadratic
    model that incorporates LET dependence.
    https://iopscience.iop.org/article/10.1088/0031-9155/61/2/888
    """

    model = "HEL"
    possible_radiation_modes = ["helium"]

    def __init__(self):
        self.p0_HEL = 1.36938e-1
        self.p1_HEL = 9.73154e-3
        self.p2_HEL = 1.51998e-2
        super().__init__()

    def _get_RBE_min_max(self, bixel: dict, kernels: dict) -> tuple[float, float]:
        LET = xp.asarray(kernels["let"])
        f_QE = (self.p1_HEL * LET**2) * xp.exp(-self.p2_HEL * LET)
        RBEmax_QE = 1 + ((self.p0_HEL + (bixel["vABratio"]) ** -1) * f_QE)

        # the linear quadratic fit yielded the best fitting result
        RBEmax = RBEmax_QE
        RBEmin = 1  # no gain in using fitted parameters over a constant value of 1
        return RBEmin, RBEmax


class LinearScaling(RBEMinMax):
    """
    This class implements the Linear Scaling Model
    according to Malte Frese https://www.ncbi.nlm.nih.gov/pubmed/20382482 (FITTED for head and neck patients !)
    """

    model = "LSM"
    possible_radiation_modes = ["protons", "helium", "carbon"]

    def __init__(self):
        self.p_lamda_1_1 = 0.008
        self.p_corrFacEntranceRBE = 0.5  # [kev/mum]
        self.p_upperLETThreshold = 30  # [kev/mum]
        self.p_lowerLETThreshold = 0.3  # [kev/mum]
        super().__init__()

    def _get_RBE_min_max(self, bixel: dict, kernels: dict) -> tuple[float, float]:
        LET = xp.asarray(kernels["let"])
        RBEmax = xp.nan * xp.ones(len(bixel["v_alpha_x"]))

        ix = self.p_lowerLETThreshold < LET < self.p_upperLETThreshold

        alpha_0 = xp.reshape(xp.asarray(bixel["v_alpha_x"]), (-1,)) - (
            self.p_lamda_1_1 * self.p_corrFacEntranceRBE
        )

        RBEmax[ix] = alpha_0[ix] + self.p_lamda_1_1 * LET[ix]

        if xp.sum(ix) < len(LET):
            RBEmax[LET > self.p_upperLETThreshold] = (
                alpha_0[LET > self.p_upperLETThreshold]
                + self.p_lamda_1_1 * self.p_upperLETThreshold
            )
            RBEmax[LET < self.p_lowerLETThreshold] = (
                alpha_0[LET < self.p_lowerLETThreshold]
                + self.p_lamda_1_1 * self.p_lowerLETThreshold
            )

        RBEmax = RBEmax / xp.asarray(bixel["v_alpha_x"])[ix]
        RBEmin = 1
        return RBEmin, RBEmax
