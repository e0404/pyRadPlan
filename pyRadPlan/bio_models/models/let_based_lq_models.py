import array_api_compat
from .lq_models import LQModel
from abc import abstractmethod
from typing import Any


class LETBasedLQModel(LQModel):
    """
    Abstract base class for LQ models whose RBE depends on linear energy transfer (LET).

    Extends :class:`LQModel` by requiring LET data from the dose engine in
    addition to physical dose. Concrete subclasses implement the specific
    relationship between LET and the radiobiological parameters alpha and beta.

    Class Attributes
    ----------------
    required_quantities : list[str]
        ``["physical_dose", "let"]``
    default_report_quantity : str
        ``"rbe_x_dose"``
    """

    required_quantities = ["physical_dose", "let"]  # Requires physical dose and LET information
    default_report_quantity = "rbe_x_dose"  # Suggested quantity for display and planning


class RBEMinMax(LETBasedLQModel):
    """
    Abstract base class for LET-based LQ models parameterised by RBEmin and RBEmax.

    alpha = RBE_max * alpha_x

    beta  = RBE_min^2 * beta_x

    where alpha_x and beta_x are the reference photon
    radiosensitivity coefficients for each voxel. Concrete subclasses provide
    the model-specific expressions for RBE_min and RBE_max as functions of LET and alpha_x/beta_x.
    """

    def calc_biological_quantities_for_bixel(self, bixel: dict, kernels: dict) -> dict:
        bixel["v_abr_x"] = bixel["v_alpha_x"] / bixel["v_beta_x"]
        bixel = super().calc_biological_quantities_for_bixel(bixel, kernels)
        [rbe_min, rbe_max] = self._get_rbe_min_max(bixel, kernels)
        bixel["alpha"] = rbe_max * bixel["v_alpha_x"]
        bixel["beta"] = rbe_min**2 * bixel["v_beta_x"]
        return bixel

    @abstractmethod
    def _get_rbe_min_max(self, bixel: dict, kernels: dict) -> tuple[Any, Any]:
        """
        Return (rbe_min, rbe_max) arrays of shape (n_depths,).
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

    def _get_rbe_min_max(self, bixel: dict, kernels: dict) -> tuple[float, float]:
        let = kernels["let"]
        rbe_max = self.p0_WED + (self.p1_WED * let) / bixel["v_abr_x"]
        rbe_min = self.p2_WED
        return rbe_min, rbe_max


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

    def _get_rbe_min_max(self, bixel: dict, kernels: dict) -> tuple[float, float]:
        xp = array_api_compat.array_namespace(kernels["let"])
        let = kernels["let"]
        rbe_max = self.p0_MCN + ((self.p1_MCN * let) / bixel["v_abr_x"])
        rbe_min = self.p2_MCN + (self.p3_MCN * xp.sqrt(bixel["v_abr_x"]) * let)
        return rbe_min, rbe_max


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

    def _get_rbe_min_max(self, bixel: dict, kernels: dict) -> tuple[float, float]:
        let = kernels["let"]
        rbe_max = self.p0_CAR + ((self.p1_CAR * self.p2_CAR) / bixel["v_abr_x"]) * let
        rbe_min = self.p3_CAR + ((self.p4_CAR * self.p2_CAR) / bixel["v_abr_x"]) * let
        return rbe_min, rbe_max


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

    def _get_rbe_min_max(self, bixel: dict, kernels: dict) -> tuple[float, float]:
        xp = array_api_compat.array_namespace(kernels["let"])
        let = kernels["let"]
        f_qe = (self.p1_HEL * let**2) * xp.exp(-self.p2_HEL * let)
        rbe_max_qe = 1 + (self.p0_HEL + 1 / bixel["v_abr_x"]) * f_qe

        # the linear quadratic fit yielded the best fitting result
        rbe_max = rbe_max_qe
        rbe_min = 1  # no gain in using fitted parameters over a constant value of 1
        return rbe_min, rbe_max


class LinearScaling(RBEMinMax):
    """
    The class implements the Linear Scaling Model.
    according to Malte Frese https://www.ncbi.nlm.nih.gov/pubmed/20382482 (FITTED for head and neck patients !)
    """

    model = "LSM"
    possible_radiation_modes = ["protons", "helium", "carbon"]

    def __init__(self):
        self.p_lamda_1_1 = 0.008
        self.p_corrFacEntrancerbe = 0.5  # [kev/mum]
        self.p_upperLETThreshold = 30  # [kev/mum]
        self.p_lowerLETThreshold = 0.3  # [kev/mum]
        super().__init__()

    def _get_rbe_min_max(self, bixel: dict, kernels: dict) -> tuple[float, float]:
        xp = array_api_compat.array_namespace(kernels["let"])
        let = kernels["let"]
        rbe_max = xp.full(bixel["v_alpha_x"].shape[0], 0.0)

        ix = (self.p_lowerLETThreshold < let) & (let < self.p_upperLETThreshold)

        alpha_0 = bixel["v_alpha_x"] - (self.p_lamda_1_1 * self.p_corrFacEntrancerbe)

        rbe_max[ix] = alpha_0[ix] + self.p_lamda_1_1 * let[ix]

        if int(xp.count_nonzero(ix)) < let.shape[0]:
            rbe_max[let > self.p_upperLETThreshold] = (
                alpha_0[let > self.p_upperLETThreshold]
                + self.p_lamda_1_1 * self.p_upperLETThreshold
            )
            rbe_max[let < self.p_lowerLETThreshold] = (
                alpha_0[let < self.p_lowerLETThreshold]
                + self.p_lamda_1_1 * self.p_lowerLETThreshold
            )
        rbe_max[ix] = rbe_max[ix] / bixel["v_alpha_x"][ix]
        rbe_min = 1
        return rbe_min, rbe_max
