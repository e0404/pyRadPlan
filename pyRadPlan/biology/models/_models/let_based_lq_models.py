"""LET-based Linear-Quadratic (LQ) models."""

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Optional

import array_api_compat

from pyRadPlan.biology.models._evaluator import BioModelEvaluator, ParametricEvaluator
from pyRadPlan.biology.models._resolution import TissueIssue

from .lq_models import LQModel


class LETBasedLQModel(LQModel):
    """
    Abstract base class for LQ models whose RBE depends on linear energy transfer (LET).

    Requires LET kernels from the dose engine in addition to physical dose. Machines with
    several LET kernel sets (e.g. dose- and track-averaged LET) are disambiguated by the
    ``quantity_variants`` parameter of the concrete models, e.g. ``{"let": "LETd"}``.
    """

    required_quantities = ("physical_dose", "let")
    kernel_set_kind = "let"

    def kernel_set_issue(self, kernel_set: Any) -> Optional[TissueIssue]:
        """Dose engines interpolate the LET kernel field ``let`` only."""
        if tuple(kernel_set.fields) == ("let",):
            return None
        return TissueIssue(
            "model_unavailable",
            f"Biological model '{self.model}' is configured for the LET kernel set "
            f"'{kernel_set.id}' ({kernel_set.label}), stored in the kernel field(s) "
            f"{list(kernel_set.fields)}; dose engines currently read the LET kernel field "
            "'let' only.",
            kernel_set=kernel_set.id,
        )

    def evaluator(self, machine: Any, voxel_params: dict[str, Any]) -> BioModelEvaluator:
        """Create the evaluator of this model for this machine."""
        return ParametricEvaluator(self)


class RBEMinMax(LETBasedLQModel):
    """
    Abstract base class for LET-based LQ models parameterised by RBEmin and RBEmax.

    alpha = RBE_max * alpha_x,  beta = RBE_min^2 * beta_x

    Concrete subclasses provide :meth:`rbe_min_max` as a function of LET and the
    reference photon parameters.

    These models are formulated in terms of the reference alpha/beta ratio and are therefore
    only defined for positive reference parameters; the exact requirement is declared in
    :attr:`~pyRadPlan.biology.models.BiologicalModel.requires_positive_reference` and validated
    by the dose engine before a calculation starts. Voxels outside every structure carry no
    reference parameters at all (``alpha_x == beta_x == 0``) and evaluate to
    ``alpha = beta = 0`` instead of a nonfinite value.
    """

    requires_positive_reference = ("alpha_x",)

    def alpha_beta(self, alpha_x: Any, beta_x: Any, context: Mapping[str, Any]) -> tuple[Any, Any]:
        """Evaluate alpha and beta using the context's named LET input."""
        let = context["let"]
        xp = array_api_compat.array_namespace(let)
        alpha_x = xp.asarray(alpha_x)
        beta_x = xp.asarray(beta_x)

        # Outside every structure no reference parameters exist. Evaluating the model there
        # would divide by zero, so it is evaluated on an in-domain placeholder and zeroed.
        defined = (alpha_x != 0.0) | (beta_x != 0.0)
        ones = xp.ones_like(alpha_x)
        rbe_min, rbe_max = self.rbe_min_max(
            let, xp.where(defined, alpha_x, ones), xp.where(defined, beta_x, ones)
        )
        alpha = rbe_max * alpha_x
        beta = rbe_min**2 * beta_x
        return (
            xp.where(defined, alpha, xp.zeros_like(alpha)),
            xp.where(defined, beta, xp.zeros_like(beta)),
        )

    @abstractmethod
    def rbe_min_max(self, let: Any, alpha_x: Any, beta_x: Any) -> tuple[Any, Any]:
        """Return (rbe_min, rbe_max), each scalar or of shape ``(n_voxels,)``."""


class Wedenberg(RBEMinMax):
    """Wedenberg model (https://www.ncbi.nlm.nih.gov/pubmed/22909391)."""

    model = "WED"
    name = "Wedenberg (LET)"
    description = "Phenomenological proton RBE from LET and α_x/β_x (Wedenberg 2013)."
    possible_radiation_modes = ("protons",)

    def __init__(
        self,
        p0: float = 1.0,
        p1: float = 0.434,
        p2: float = 1.0,
        quantity_variants: Optional[dict[str, str]] = None,
    ):
        self.quantity_variants = dict(quantity_variants) if quantity_variants else None
        self.p0_WED = p0
        self.p1_WED = p1
        self.p2_WED = p2

    def rbe_min_max(self, let, alpha_x, beta_x):
        """Return (rbe_min, rbe_max) from the Wedenberg model."""
        rbe_max = self.p0_WED + (self.p1_WED * let) / (alpha_x / beta_x)
        rbe_min = self.p2_WED
        return rbe_min, rbe_max


class MCNamara(RBEMinMax):
    """
    McNamara model (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4634882/).

    ``RBE_min`` contains ``sqrt(alpha_x / beta_x)``, which diverges for a vanishing reference
    beta, so unlike the other RBEmin/RBEmax models this one also needs ``beta_x > 0``.
    """

    model = "MCN"
    name = "McNamara (LET)"
    description = "Phenomenological proton RBE from LET and α_x/β_x (McNamara 2015)."
    possible_radiation_modes = ("protons",)
    requires_positive_reference = ("alpha_x", "beta_x")

    def __init__(
        self,
        p0: float = 0.999064,
        p1: float = 0.35605,
        p2: float = 1.1012,
        p3: float = -0.0038703,
        quantity_variants: Optional[dict[str, str]] = None,
    ):
        self.quantity_variants = dict(quantity_variants) if quantity_variants else None
        self.p0_MCN = p0
        self.p1_MCN = p1
        self.p2_MCN = p2
        self.p3_MCN = p3

    def rbe_min_max(self, let, alpha_x, beta_x):
        """Return (rbe_min, rbe_max) from the McNamara model."""
        xp = array_api_compat.array_namespace(let)
        abr = alpha_x / beta_x
        rbe_max = self.p0_MCN + (self.p1_MCN * let) / abr
        rbe_min = self.p2_MCN + self.p3_MCN * xp.sqrt(abr) * let
        return rbe_min, rbe_max


class Carabe(RBEMinMax):
    """
    Carabe model.

    https://www.tandfonline.com/doi/full/10.1080/09553000601087176?journalCode=irab20
    """

    model = "CAR"
    name = "Carabe (LET)"
    description = "Phenomenological proton RBE from LET and α_x/β_x (Carabe 2007)."
    possible_radiation_modes = ("protons",)

    def __init__(
        self,
        p0: float = 0.843,
        p1: float = 0.154,
        p2: float = 2.686,
        p3: float = 1.09,
        p4: float = 0.006,
        quantity_variants: Optional[dict[str, str]] = None,
    ):
        self.quantity_variants = dict(quantity_variants) if quantity_variants else None
        self.p0_CAR = p0
        self.p1_CAR = p1
        self.p2_CAR = p2
        self.p3_CAR = p3
        self.p4_CAR = p4

    def rbe_min_max(self, let, alpha_x, beta_x):
        """Return (rbe_min, rbe_max) from the Carabe model."""
        abr = alpha_x / beta_x
        rbe_max = self.p0_CAR + ((self.p1_CAR * self.p2_CAR) / abr) * let
        rbe_min = self.p3_CAR + ((self.p4_CAR * self.p2_CAR) / abr) * let
        return rbe_min, rbe_max


class HeliumMairani(RBEMinMax):
    """
    Mairani model for helium ions.

    https://iopscience.iop.org/article/10.1088/0031-9155/61/2/888
    """

    model = "HEL"
    name = "Mairani (LET)"
    description = "Phenomenological helium RBE from LET and α_x/β_x (Mairani 2016)."
    possible_radiation_modes = ("helium",)

    def __init__(
        self,
        p0: float = 1.36938e-1,
        p1: float = 9.73154e-3,
        p2: float = 1.51998e-2,
        quantity_variants: Optional[dict[str, str]] = None,
    ):
        self.quantity_variants = dict(quantity_variants) if quantity_variants else None
        self.p0_HEL = p0
        self.p1_HEL = p1
        self.p2_HEL = p2

    def rbe_min_max(self, let, alpha_x, beta_x):
        """Return (rbe_min, rbe_max) from the Mairani helium model."""
        xp = array_api_compat.array_namespace(let)
        f_qe = (self.p1_HEL * let**2) * xp.exp(-self.p2_HEL * let)
        # the linear quadratic fit yielded the best fitting result
        rbe_max = 1 + (self.p0_HEL + beta_x / alpha_x) * f_qe
        rbe_min = 1  # no gain in using fitted parameters over a constant value of 1
        return rbe_min, rbe_max


class LinearScaling(RBEMinMax):
    """
    Linear Scaling Model according to Malte Frese, fitted for head and neck patients.

    https://www.ncbi.nlm.nih.gov/pubmed/20382482
    """

    model = "LSM"
    name = "Linear scaling (LET)"
    description = "α scaled linearly with LET (Frese 2011)."
    possible_radiation_modes = ("protons", "helium", "carbon")

    def __init__(
        self,
        lambda_1_1: float = 0.008,
        corr_fac_entrance_rbe: float = 0.5,
        upper_let_threshold: float = 30.0,
        lower_let_threshold: float = 0.3,
        quantity_variants: Optional[dict[str, str]] = None,
    ):
        self.quantity_variants = dict(quantity_variants) if quantity_variants else None
        self.p_lamda_1_1 = lambda_1_1
        self.p_corrFacEntrancerbe = corr_fac_entrance_rbe  # [keV/um]
        self.p_upperLETThreshold = upper_let_threshold  # [keV/um]
        self.p_lowerLETThreshold = lower_let_threshold  # [keV/um]

    def rbe_min_max(self, let, alpha_x, beta_x):
        """Return (rbe_min, rbe_max) from the linear scaling model."""
        xp = array_api_compat.array_namespace(let)
        let = xp.clip(let, self.p_lowerLETThreshold, self.p_upperLETThreshold)
        alpha_0 = alpha_x - self.p_lamda_1_1 * self.p_corrFacEntrancerbe
        alpha = alpha_0 + self.p_lamda_1_1 * let
        rbe_max = xp.where(alpha_x > 0, alpha / xp.where(alpha_x > 0, alpha_x, 1.0), 0.0)
        rbe_min = 1
        return rbe_min, rbe_max
