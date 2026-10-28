biology.models
==============

.. currentmodule:: pyRadPlan.biology.models

See :ref:`custom_biological_models` for the extension contract and a complete registered model.

Models
------

.. autosummary::
   :toctree: generated/biology_models/

   BiologicalModel
   EmptyModel
   ConstantRBEModel
   LQModel
   LETBasedLQModel
   RBEMinMax
   Wedenberg
   MCNamara
   Carabe
   HeliumMairani
   LinearScaling
   KernelBasedLQModel
   ZStarBasedLQModel
   MKM
   TabulatedRBEModel
   TabulatedAlphaBetaModel

The RBE and stopping power tables of the tabulated models are base data files (see
:ref:`base_data_folders`); these functions add one to the user's folder:

.. autosummary::
   :toctree: generated/biology_models/

   add_rbe_table
   add_stopping_power_table

Formalisms are registered under identifiers named by what they read: ``alpha_beta_kernels``
(:class:`KernelBasedLQModel`), ``alpha_beta_tables`` (:class:`TabulatedAlphaBetaModel`) and
``zstar_kernels`` (:class:`MKM`). Their former identifiers (``kernel_based_lq`` / ``LEM``,
``dose_average_alpha_beta``, ``MKM`` / ``zs_based_lq``) remain registered; the classes are
also exported as ``AlphaBetaKernelModel``, ``AlphaBetaTableModel`` and ``ZStarKernelModel``.
See :ref:`biological_base_data`.


Evaluators and tissue lookups
-----------------------------

A lookup maps each voxel's tissue class onto the rows of the machine's per-class base
data, comparing the fields that identify a tissue class of the model (see
:mod:`pyRadPlan.biology.tissues`).

.. autosummary::
   :toctree: generated/biology_models/

   BioEvaluationContext
   BioModelResult
   BioModelEvaluator
   ParametricEvaluator
   KernelBasedEvaluator
   PrecomputedAlphaBetaEvaluator
   DerivedAlphaBetaEvaluator
   TabulatedSpectrumEvaluator
   TissueParameterLookup
   ExactClassLookup
   make_tissue_lookup


Options and tissue checks
-------------------------

:func:`bio_model_options` lists the formalisms a machine offers, one entry per kernel set or
RBE table, labelled "formalism · provenance". :meth:`BiologicalModel.check_tissues` reports
the structures whose tissues do not resolve to a tissue class of that base data; dose engines
act on the same check through :func:`raise_for_tissue_issues`, raising (or falling back to
the physical dose) for issues of ``severity="error"`` only. A tissue matching the base data
by its parameters but labelled with another provenance (MKM for mMKM, another LEM version)
is a ``provenance_differs`` warning: the model is evaluated with the base data, and the
optional relabel ``fix_provenance`` is applied by :func:`apply_provenance_fixes`.
:meth:`BiologicalModel.compare_tissues` gives the per-parameter view of the same check.

Both take the dose engine as ``engine`` (short name, class, instance or
:class:`EngineBioCapability`): a formalism is only available if the engine can compute it.
Pencil-beam engines evaluate the machine's kernel data; Monte Carlo engines do not use it and
supply only what they score themselves (FRED: LET), see
:meth:`~pyRadPlan.dose.engines.DoseEngineBase.bio_capability`.

.. autosummary::
   :toctree: generated/biology_models/

   bio_model_options
   BioModelOption
   EngineBioCapability
   engine_bio_capability
   TissueIssue
   ISSUE_KINDS
   ISSUE_SEVERITIES
   apply_provenance_fixes
   compare_tissue_parameters
   TissueComparison
   ParameterComparison
   resolve_bio_model
   BioModelResolution
   StructureResolution
   select_kernel_set
   KernelSetSelectionError
   structure_tissues
   raise_for_tissue_issues
   BioModelUnavailableError
   validate_reference_coefficients


Kernel sets
-----------

A kernel set is one family of per-energy biological kernel data of a machine (α/β kernels,
z* kernels, a LET), with the provenance that produced it and its tissue classes, as declared
in ``machine.meta.kernelSets`` / ``machine.meta.tissueParameters`` or inferred from legacy
data (see :ref:`base_data_metadata`).

.. autosummary::
   :toctree: generated/biology_models/

   KernelSetInfo
   TissueClassInfo
   BioMachineSummary
   KERNEL_SET_KINDS
   machine_kernel_sets
   build_kernel_sets
   parse_kernel_set_declarations
   kernel_reference_classes


Creating and registering models
-------------------------------

.. autosummary::
   :toctree: generated/biology_models/

   create_bio_model
   get_bio_model
   available_bio_models
   get_available_models
   register_model
   bio_model_spec_from_matrad
   bio_influence_from_let
