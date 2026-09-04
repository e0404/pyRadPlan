"""Biological models and their per-calculation evaluators."""

from ._base import BiologicalModel, BiologicalModelBase, EmptyModel
from ._evaluator import (
    BioEvaluationContext,
    BioModelResult,
    BioModelEvaluator,
    ParametricEvaluator,
    KernelBasedEvaluator,
    TabulatedSpectrumEvaluator,
)
from ._tissue_lookup import TissueParameterLookup, ExactClassLookup, make_tissue_lookup
from ._influence import alpha_beta_influence_from_let
from .models.const_rbe import ConstantRBEModel
from .models.let_based_lq_models import (
    LETBasedLQModel,
    RBEMinMax,
    Wedenberg,
    MCNamara,
    Carabe,
    HeliumMairani,
    LinearScaling,
)
from .models.kernel_based_lq_model import KernelBasedLQModel
from .models.lq_models import LQModel
from .models.tabulated_rbe_models import TabulatedRBEModel, TabulatedAlphaBetaModel
from ._factory import (
    register_model,
    get_available_models,
    available_bio_models,
    get_bio_model,
    create_bio_model,
    bio_model_spec_from_matrad,
    BioModelSpec,
)

register_model(EmptyModel)
register_model(ConstantRBEModel)
register_model(Wedenberg)
register_model(MCNamara)
register_model(Carabe)
register_model(HeliumMairani)
register_model(KernelBasedLQModel)
register_model(LinearScaling)
register_model(TabulatedAlphaBetaModel)

__all__ = [
    "BiologicalModel",
    "BiologicalModelBase",
    "BioEvaluationContext",
    "BioModelResult",
    "BioModelEvaluator",
    "ParametricEvaluator",
    "KernelBasedEvaluator",
    "TabulatedSpectrumEvaluator",
    "TissueParameterLookup",
    "ExactClassLookup",
    "make_tissue_lookup",
    "alpha_beta_influence_from_let",
    "EmptyModel",
    "ConstantRBEModel",
    "LETBasedLQModel",
    "RBEMinMax",
    "Wedenberg",
    "MCNamara",
    "Carabe",
    "HeliumMairani",
    "LinearScaling",
    "KernelBasedLQModel",
    "LQModel",
    "TabulatedRBEModel",
    "TabulatedAlphaBetaModel",
    "register_model",
    "get_available_models",
    "available_bio_models",
    "get_bio_model",
    "create_bio_model",
    "bio_model_spec_from_matrad",
    "BioModelSpec",
]
