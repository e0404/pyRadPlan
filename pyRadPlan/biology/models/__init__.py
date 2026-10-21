"""Biological models and their per-calculation evaluators."""

from ._base import (
    BiologicalModel,
    BiologicalModelBase,
    BioModelUnavailableError,
    EmptyModel,
    raise_for_tissue_issues,
    validate_reference_coefficients,
)
from ._evaluator import (
    BioEvaluationContext,
    BioModelResult,
    BioModelEvaluator,
    ParametricEvaluator,
    KernelBasedEvaluator,
    PrecomputedAlphaBetaEvaluator,
    DerivedAlphaBetaEvaluator,
    TabulatedSpectrumEvaluator,
)
from ._kernel_sets import (
    KERNEL_SET_KINDS,
    BioMachineSummary,
    KernelSetInfo,
    TissueClassInfo,
    build_kernel_sets,
    kernel_reference_classes,
    machine_kernel_sets,
    parse_kernel_set_declarations,
)
from ._resolution import (
    ISSUE_KINDS,
    ISSUE_SEVERITIES,
    BioModelResolution,
    KernelSetSelectionError,
    StructureResolution,
    ParameterComparison,
    TissueComparison,
    TissueIssue,
    apply_provenance_fixes,
    compare_tissue_parameters,
    resolve_bio_model,
    select_kernel_set,
    structure_tissues,
)
from ._tissue_lookup import TissueParameterLookup, ExactClassLookup, make_tissue_lookup
from ._influence import bio_influence_from_let
from ._models.const_rbe import ConstantRBEModel
from ._models.let_based_lq_models import (
    LETBasedLQModel,
    RBEMinMax,
    Wedenberg,
    MCNamara,
    Carabe,
    HeliumMairani,
    LinearScaling,
)
from ._models.kernel_based_lq_model import KernelBasedLQModel
from ._models.zstar_based_lq_model import ZStarBasedLQModel, ZStarKernelModel, MKM
from ._models.lq_models import LQModel
from ._models.tabulated_rbe_models import (
    TabulatedRBEModel,
    TabulatedAlphaBetaModel,
    add_rbe_table,
    add_stopping_power_table,
)
from ._factory import (
    register_model,
    get_available_models,
    available_bio_models,
    get_bio_model,
    create_bio_model,
    bio_model_spec_from_matrad,
    BioModelSpec,
)
from ._options import BioModelOption, bio_model_options
from ._engine import EngineBioCapability, engine_bio_capability

#: The α/β kernel formalism (``alpha_beta_kernels``) by what it reads.
AlphaBetaKernelModel = KernelBasedLQModel
#: The α/β table formalism (``alpha_beta_tables``) by what it reads.
AlphaBetaTableModel = TabulatedAlphaBetaModel

register_model(EmptyModel)
register_model(ConstantRBEModel)
register_model(Wedenberg)
register_model(MCNamara)
register_model(Carabe)
register_model(HeliumMairani)
register_model(KernelBasedLQModel)
register_model(LinearScaling)
register_model(TabulatedAlphaBetaModel)
register_model(MKM)

__all__ = [
    "BiologicalModel",
    "BiologicalModelBase",
    "BioModelUnavailableError",
    "validate_reference_coefficients",
    "raise_for_tissue_issues",
    "BioEvaluationContext",
    "BioModelResult",
    "BioModelEvaluator",
    "ParametricEvaluator",
    "KernelBasedEvaluator",
    "PrecomputedAlphaBetaEvaluator",
    "DerivedAlphaBetaEvaluator",
    "TabulatedSpectrumEvaluator",
    "KERNEL_SET_KINDS",
    "BioMachineSummary",
    "KernelSetInfo",
    "TissueClassInfo",
    "build_kernel_sets",
    "kernel_reference_classes",
    "machine_kernel_sets",
    "parse_kernel_set_declarations",
    "ISSUE_KINDS",
    "ISSUE_SEVERITIES",
    "BioModelResolution",
    "KernelSetSelectionError",
    "StructureResolution",
    "TissueIssue",
    "ParameterComparison",
    "TissueComparison",
    "apply_provenance_fixes",
    "compare_tissue_parameters",
    "resolve_bio_model",
    "select_kernel_set",
    "structure_tissues",
    "BioModelOption",
    "bio_model_options",
    "EngineBioCapability",
    "engine_bio_capability",
    "TissueParameterLookup",
    "ExactClassLookup",
    "make_tissue_lookup",
    "bio_influence_from_let",
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
    "AlphaBetaKernelModel",
    "LQModel",
    "TabulatedRBEModel",
    "TabulatedAlphaBetaModel",
    "add_rbe_table",
    "add_stopping_power_table",
    "AlphaBetaTableModel",
    "ZStarBasedLQModel",
    "ZStarKernelModel",
    "MKM",
    "register_model",
    "get_available_models",
    "available_bio_models",
    "get_bio_model",
    "create_bio_model",
    "bio_model_spec_from_matrad",
    "BioModelSpec",
]
