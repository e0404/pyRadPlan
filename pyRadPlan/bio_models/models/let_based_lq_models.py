"""LET-based Linear-Quadratic (LQ) models."""

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

import array_api_compat

from pyRadPlan.bio_models._evaluator import BioModelEvaluator, ParametricEvaluator

from .lq_models import LQModel


class LETBasedLQModel(LQModel):
    """
    Abstract base class for LQ models whose RBE depends on linear energy transfer (LET).

    Requires LET kernels from the dose engine in addition to physical dose.
    """

    required_quantities = ("physical_dose", "let")

    def evaluator(self, machine: Any, voxel_params: dict[str, Any]) -> BioModelEvaluator:
        return ParametricEvaluator(self)


class RBEMinMax(LETBasedLQModel):
    """
    Abstract base class for LET-based LQ models parameterised by RBEmin and RBEmax:

    alpha = RBE_max * alpha_x,  beta = RBE_min^2 * beta_x

    Concrete subclasses provide :meth:`rbe_min_max` as a function of LET and the
    reference photon parameters.
    """

    def alpha_beta(self, alpha_x: Any, beta_x: Any, context: Mapping[str, Any]) -> tuple[Any, Any]:
        """Evaluate alpha and beta using the context's named LET input."""
        rbe_min, rbe_max = self.rbe_min_max(context["let"], alpha_x, beta_x)
        return rbe_max * alpha_x, rbe_min**2 * beta_x

    @abstractmethod
    def rbe_min_max(self, let: Any, alpha_x: Any, beta_x: Any) -> tuple[Any, Any]:
        """Return (rbe_min, rbe_max), each scalar or of shape ``(n_voxels,)``."""


class Wedenberg(RBEMinMax):
    """
    Wedenberg model (https://www.ncbi.nlm.nih.gov/pubmed/22909391).
    """

    model = "WED"
    possible_radiation_modes = ("protons",)

    def __init__(self, p0: float = 1.0, p1: float = 0.434, p2: float = 1.0):
        self.p0_WED = p0
        self.p1_WED = p1
        self.p2_WED = p2

    def rbe_min_max(self, let, alpha_x, beta_x):
        rbe_max = self.p0_WED + (self.p1_WED * let) / (alpha_x / beta_x)
        rbe_min = self.p2_WED
        return rbe_min, rbe_max


class MCNamara(RBEMinMax):
    """
    McNamara model (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4634882/).
    """

    model = "MCN"
    possible_radiation_modes = ("protons",)

    def __init__(
        self,
        p0: float = 0.999064,
        p1: float = 0.35605,
        p2: float = 1.1012,
        p3: float = -0.0038703,
    ):
        self.p0_MCN = p0
        self.p1_MCN = p1
        self.p2_MCN = p2
        self.p3_MCN = p3

    def rbe_min_max(self, let, alpha_x, beta_x):
        xp = array_api_compat.array_namespace(let)
        abr = alpha_x / beta_x
        rbe_max = self.p0_MCN + (self.p1_MCN * let) / abr
        rbe_min = self.p2_MCN + self.p3_MCN * xp.sqrt(abr) * let
        return rbe_min, rbe_max


class Carabe(RBEMinMax):
    """
    Carabe model
    (https://www.tandfonline.com/doi/full/10.1080/09553000601087176?journalCode=irab20).
    """

    model = "CAR"
    possible_radiation_modes = ("protons",)

    def __init__(
        self,
        p0: float = 0.843,
        p1: float = 0.154,
        p2: float = 2.686,
        p3: float = 1.09,
        p4: float = 0.006,
    ):
        self.p0_CAR = p0
        self.p1_CAR = p1
        self.p2_CAR = p2
        self.p3_CAR = p3
        self.p4_CAR = p4

    def rbe_min_max(self, let, alpha_x, beta_x):
        abr = alpha_x / beta_x
        rbe_max = self.p0_CAR + ((self.p1_CAR * self.p2_CAR) / abr) * let
        rbe_min = self.p3_CAR + ((self.p4_CAR * self.p2_CAR) / abr) * let
        return rbe_min, rbe_max


class HeliumMairani(RBEMinMax):
    """
    Mairani model for helium ions (https://iopscience.iop.org/article/10.1088/0031-9155/61/2/888).
    """

    model = "HEL"
    possible_radiation_modes = ("helium",)

    def __init__(self, p0: float = 1.36938e-1, p1: float = 9.73154e-3, p2: float = 1.51998e-2):
        self.p0_HEL = p0
        self.p1_HEL = p1
        self.p2_HEL = p2

    def rbe_min_max(self, let, alpha_x, beta_x):
        xp = array_api_compat.array_namespace(let)
        f_qe = (self.p1_HEL * let**2) * xp.exp(-self.p2_HEL * let)
        # the linear quadratic fit yielded the best fitting result
        rbe_max = 1 + (self.p0_HEL + beta_x / alpha_x) * f_qe
        rbe_min = 1  # no gain in using fitted parameters over a constant value of 1
        return rbe_min, rbe_max


class LinearScaling(RBEMinMax):
    """
    Linear Scaling Model according to Malte Frese
    (https://www.ncbi.nlm.nih.gov/pubmed/20382482, fitted for head and neck patients).
    """

    model = "LSM"
    possible_radiation_modes = ("protons", "helium", "carbon")

    def __init__(
        self,
        lambda_1_1: float = 0.008,
        corr_fac_entrance_rbe: float = 0.5,
        upper_let_threshold: float = 30.0,
        lower_let_threshold: float = 0.3,
    ):
        self.p_lamda_1_1 = lambda_1_1
        self.p_corrFacEntrancerbe = corr_fac_entrance_rbe  # [keV/um]
        self.p_upperLETThreshold = upper_let_threshold  # [keV/um]
        self.p_lowerLETThreshold = lower_let_threshold  # [keV/um]

    def rbe_min_max(self, let, alpha_x, beta_x):
        xp = array_api_compat.array_namespace(let)
        let = xp.clip(let, self.p_lowerLETThreshold, self.p_upperLETThreshold)
        alpha_0 = alpha_x - self.p_lamda_1_1 * self.p_corrFacEntrancerbe
        alpha = alpha_0 + self.p_lamda_1_1 * let
        rbe_max = xp.where(alpha_x > 0, alpha / xp.where(alpha_x > 0, alpha_x, 1.0), 0.0)
        rbe_min = 1
        return rbe_min, rbe_max
