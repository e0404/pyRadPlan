.. _custom_biological_models:

Extending biological models
===========================

Biological models can be supplied by application code or another package without modifying
pyRadPlan. A model is a lightweight, serialisable parameter object. It creates a fresh evaluator
for each dose calculation; the evaluator owns machine- and patient-specific state and communicates
with the dose engine through named inputs and outputs.

Contract at a glance
--------------------

.. list-table::
   :header-rows: 1
   :widths: 24 31 45

   * - Owner
     - Declaration or method
     - Meaning
   * - Model
     - ``model`` and ``model_aliases``
     - Canonical registry name and optional lookup aliases.
   * - Model
     - ``possible_radiation_modes``
     - Modalities for which the model is meaningful.
   * - Model
     - ``required_quantities``
     - Capabilities required from the selected machine and dose engine, such as
       ``physical_dose``, ``let``, ``alpha``, ``beta`` or ``fluence``.
   * - Model
     - ``output_quantities``
     - Intrinsic names returned by ``evaluator.evaluate()``, for example ``alpha`` and ``beta``.
   * - Evaluator
     - ``kernel_field_names`` and ``kernel_quantities()``
     - Model-specific depth arrays taken from each machine-energy kernel and interpolated before
       evaluation. The default is no model-specific kernel fields.
   * - Evaluator
     - ``influence_quantity_names`` and ``evaluate_influence()``
     - Additive per-bixel outputs stored as influence matrices.

The particle pencil-beam path has the following data flow::

    Machine.provided_quantities() -> model availability check
    model.evaluator(machine, voxel_params) -> one validated evaluator
    machine kernel -> kernel_quantities() -> depth interpolation
    standard inputs + interpolated fields -> BioEvaluationContext
    evaluate_influence() -> additive Dij matrices

All name collections are immutable tuples of unique, non-empty strings. Registration validates
the model declarations. A dose engine validates the returned evaluator, its binding to the model
instance, and its kernel/influence declarations during setup. Custom evaluators implement the
protected ``_evaluate()`` and ``_evaluate_influence()`` hooks; their public counterparts validate
the returned names against ``output_quantities`` and ``influence_quantity_names``.

Complete LET-based example
--------------------------

This example implements the LQ parameters
``alpha = alpha_x * (1 + slope * LET)`` and ``beta = beta_x``. It uses only Array API operations,
so the evaluator follows the backend selected by the dose engine.

.. code-block:: python

    import array_api_compat

    from pyRadPlan.bio_models import (
        BioEvaluationContext,
        BiologicalModel,
        BioModelEvaluator,
        BioModelResult,
        register_model,
    )


    class LinearLETEvaluator(BioModelEvaluator):
        @property
        def influence_quantity_names(self) -> tuple[str, ...]:
            return ("alpha_dose", "sqrt_beta_dose")

        def _evaluate(self, context: BioEvaluationContext) -> BioModelResult:
            alpha_x = context.require("alpha_x")
            beta_x = context.require("beta_x")
            let = context.require("let")
            return BioModelResult(
                {
                    "alpha": alpha_x * (1.0 + self.model.slope * let),
                    "beta": beta_x,
                }
            )

        def _evaluate_influence(self, context: BioEvaluationContext):
            result = self.evaluate(context)
            dose = context.require("physical_dose")
            xp = array_api_compat.array_namespace(dose)
            return {
                "alpha_dose": dose * result.require("alpha"),
                "sqrt_beta_dose": dose * xp.sqrt(result.require("beta")),
            }


    @register_model
    class LinearLETModel(BiologicalModel):
        model = "linear_let"
        model_aliases = ("LLET",)
        required_quantities = ("physical_dose", "let")
        output_quantities = ("alpha", "beta")
        possible_radiation_modes = ("protons",)

        def __init__(self, slope: float = 0.02):
            self.slope = slope

        def evaluator(self, machine, voxel_params) -> BioModelEvaluator:
            return LinearLETEvaluator(self)


Import the module defining the class before validating a plan, then select it by canonical name,
alias, specification dictionary or instance:

.. code-block:: python

    pln.bio_model = {"model": "linear_let", "slope": 0.03}

The base class records explicitly supplied constructor arguments. Consequently
``pln.bio_model.to_dict()`` returns
``{"model": "linear_let", "slope": 0.03}``, which is the form used for native plan
serialisation. Constructor parameters should therefore be serialisable. Custom matRad translation
requires an explicit import/export adapter; a registry name alone does not establish equivalence
to a matRad model.

Registration behavior
---------------------

:func:`~pyRadPlan.bio_models.register_model` is both a normal function and a class decorator.
Registration is process-global and idempotent for the same class. Canonical names and aliases must
be distinct. If any name belongs to another class, registration raises before adding any names, so
an alias cannot be left partially registered.

Reloading the defining module creates a different class object and is deliberately treated as a
collision. Restart the process or notebook kernel before registering the replacement. This avoids
silently retaining the stale class in registry-based model lookup.

The registry validates model declarations, not numerical validity. Evaluators should use
:meth:`~pyRadPlan.bio_models.BioEvaluationContext.require` and
:meth:`~pyRadPlan.bio_models.BioModelResult.require` so missing inputs or outputs produce
model-facing diagnostics. Declaration validation also runs when a model instance is passed
directly without registry lookup.

Inputs and evaluation methodologies
-----------------------------------

The standard context vocabulary is ``alpha_x``, ``beta_x``, ``physical_dose`` and, when declared
as required by the model, ``let``. Engine geometry, raw bixel dictionaries and raw kernel objects
are intentionally not part of the contract.

For model-specific depth tables, override ``kernel_field_names`` and ``kernel_quantities()``. The
latter receives the namespaced kernel for one energy and must return exactly the declared keys for
every energy; the engine validates that equality before interpolating the first bixel at each
energy. Each value may have shape ``(n_depths,)`` or ``(n_classes, n_depths)``. After depth
interpolation, the evaluator reads the field under the same name from the context. Names colliding
with standard context inputs or dose-engine lateral/depth kernels are rejected during setup.

:class:`~pyRadPlan.bio_models.KernelBasedEvaluator` provides the existing exact tissue-class
lookup workflow for alpha/beta depth kernels. :class:`~pyRadPlan.bio_models.TabulatedSpectrumEvaluator`
provides the spectrum workflow: it dose-averages energy tables over fragment fluence spectra once
when the evaluator is created, then exposes the resulting depth kernels through the same context.
Their model-side protocols are documented on those classes and the built-in kernel and tabulated
models provide reference implementations.

Current storage boundary
------------------------

``BioModelResult`` itself can contain arbitrary intrinsic outputs. Influence outputs are more
restricted because ``Dij`` currently has fixed fields: a custom evaluator used by the dose engines
may declare ``alpha_dose`` and ``sqrt_beta_dose``. Other additive outputs fail during setup with a
clear ``NotImplementedError``. The proposed registry for dynamically stored influence quantities
is documented in ``docs/development/dynamic_dij_quantities.md`` and remains outside this API change.
