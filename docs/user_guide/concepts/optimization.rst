.. _concept_optimization:

Optimization
============

pyRadPlan separates the optimization workflow into three orthogonal concepts:

- **Problems** — define *what* is being optimized (variables, constraints, structure of the
  objective function).
- **Solvers** — define *how* the mathematical program is solved (algorithm, convergence
  criteria).
- **Objectives** — define *clinical goals* attached to individual structures (DVH goals, dose
  targets, dose limits).

This separation allows objectives to be reused across different problem formulations and solvers
to be swapped without touching clinical goal definitions.

Running optimization
--------------------

The high-level entry point is :func:`~pyRadPlan.fluence_optimization`:

.. code-block:: python

    from pyRadPlan import fluence_optimization

    fluence = fluence_optimization(ct, cst, stf, dij, pln)

Under the hood, this function:

1. Instantiates the planning problem configured in ``pln.prop_opt["problem"]``.
2. Reads objectives from each VOI in ``cst``.
3. Resolves required :ref:`quantities <concept_quantities>` from ``dij``.
4. Calls the solver and returns the optimal fluence vector.

Planning problems
-----------------

A *planning problem* defines the optimization variable and the structure of the objective
function. The problem class is selected via ``pln.prop_opt["problem"]``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Description
   * - ``"nonlin_fluence"``
     - Nonlinear beamlet fluence optimization. Variables are non-negative beamlet
       weights; the objective is the weighted sum of clinical-goal penalty functions.

.. code-block:: python

    pln.prop_opt = {"problem": "nonlin_fluence"}

Problems are registered at import time and can be extended by registering additional
``PlanningProblem`` subclasses with ``register_problem()``.

Solvers
-------

A *solver* implements the mathematical optimization algorithm. The solver is selected via the
``"solver"`` key inside ``pln.prop_opt`` or directly on the problem object:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Description
   * - ``"ipopt"``
     - (default) `IPOPT <https://coin-or.github.io/Ipopt/>`_ interior-point optimizer.
       Handles large-scale nonlinear programs efficiently. Suitable for the full fluence
       optimization problem.
   * - ``"ipax"``
     - `ipax <https://github.com/wahln/ipax>`_ interior-point optimizer. Array-API-native,
       so it can keep the fluence vector on the compute backend (e.g. a GPU) instead of
       round-tripping it through NumPy.
   * - ``"scipy"``
     - SciPy minimization (``scipy.optimize.minimize``). Multiple sub-methods available
       (L-BFGS-B, SLSQP, etc.). Lighter dependency, good for smaller problems.

.. code-block:: python

    pln.prop_opt = {
        "problem": "nonlin_fluence",
        "solver": "scipy",
    }

The currently registered fluence problem constrains fluence weights to be non-negative.

.. note::

   IPOPT is only offered when it can be used safely.  The ``ipyopt`` wheel bundles its own copy
   of the Intel OpenMP runtime, and if another copy is already loaded in the process -- PyTorch
   ships one too -- that runtime aborts the interpreter (``OMP: Error #15``) as soon as a solve
   starts.  pyRadPlan detects this at import time, logs a warning, leaves ``"ipopt"`` out of
   :func:`~pyRadPlan.optimization.solvers.get_available_solvers` (so a plan falls back to the next
   solver), and records the cause in
   ``pyRadPlan.optimization.solvers.IPOPT_DISABLED_REASON``.

   To use IPOPT anyway, set ``KMP_DUPLICATE_LIB_OK=TRUE`` *before* starting Python.  Intel
   documents this as unsafe -- the process may still crash or silently produce wrong results --
   so prefer an environment with a single OpenMP runtime where you can.

.. _concept_ipax:

The ipax solver
~~~~~~~~~~~~~~~

`ipax <https://github.com/wahln/ipax>`_ is a primal-dual interior-point solver for large-scale
nonlinear constrained optimization, written against the
`Python Array API standard <https://data-apis.org/array-api/latest/>`_.  It is developed by
Niklas Wahl as a separate project under the Apache-2.0 license, with its own
`documentation <https://ipax.readthedocs.io/>`_.  Its globalization (filter line search,
restoration) follows IPOPT (Wächter & Biegler, 2006), and its primal-dual formulation draws on
the radiotherapy-tuned interior-point method of Breedveld et al. (2017).  ipax is in beta: the
algorithm is feature-complete and tested, while its public API is still being hardened.

Because ipax never imports a concrete array library, it solves in whatever namespace the
starting point lives in.  pyRadPlan hands it the fluence vector on the compute backend chosen
for the quantities (see `Compute backend`_ below), so on a GPU backend the whole solve,
including the dose projections, can stay on the device.

ipax is installed with pyRadPlan.  pyRadPlan's interface,
:class:`~pyRadPlan.optimization.solvers.OptimizerIpax`, starts from defaults tuned for fluence
optimization: an L-BFGS Hessian, the adaptive barrier-parameter schedule, Gondzio corrections,
interpolated backtracking in the line search, and an acceptable-stopping band loosened for
radiotherapy objectives, which stops a solve once further iterations no longer improve the
plan noticeably.  The iteration and time limits come from the solver's ``max_iter`` and
``max_time``; everything else is taken from its ``options`` (an ``ipax.Options`` tree), which a
plan configures with a nested dictionary naming only the entries to change:

.. code-block:: python

    pln.prop_opt = {
        "solver": {
            "name": "ipax",
            "max_iter": 300,
            "options": {"lbfgs": {"memory": 20}, "acceptable": {"n_iter": 8}},
        },
    }

The built-in fluence problem supplies an objective and gradient, so keep a quasi-Newton
Hessian (``"lbfgs"`` or ``"auto"``).  ``"exact"`` requires an analytic Lagrangian Hessian,
which this interface does not currently provide.

In the GUI, the "…" button next to the planning problem in the plan widget offers the same
settings, with ipax's options as collapsible groups.

``force_cpu_inner_solve`` (default ``False``) runs ipax's own linear algebra in NumPy while the
objective and gradient evaluations stay on the compute backend.  Which linear-solve route ipax
takes (dense, matrix-free Krylov, or sparse-direct) depends on the size and structure of the
problem (``options.linsolve``, ``"auto"`` by default).  The sparse-direct route can use the
optional solvers described in :ref:`ipax-accelerations`; with ``force_cpu_inner_solve`` it
factors on the host, so it reaches the GPU solver cuDSS only without it.

Objectives
----------

*Objectives* are penalty or constraint functions that express clinical goals. They are attached
directly to :class:`~pyRadPlan.cst.VOI` objects inside the structure set and are collected
automatically during optimization.

Each objective targets a *quantity* (e.g. ``"physical_dose"``, ``"rbe_x_dose"``), has a
``priority`` (weight in the combined objective), and may reference a *dose level* or DVH
parameter.

Available objectives
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :class:`~pyRadPlan.optimization.objectives.SquaredDeviation`
     - Penalizes squared deviations from a reference dose. Good all-rounder for PTV coverage.
   * - :class:`~pyRadPlan.optimization.objectives.SquaredUnderdosing`
     - Penalizes only dose below the reference (one-sided). Use for target coverage without
       over-irradiation penalty.
   * - :class:`~pyRadPlan.optimization.objectives.SquaredOverdosing`
     - Penalizes only dose above the reference. Use for OAR sparing without coverage trade-off.
   * - :class:`~pyRadPlan.optimization.objectives.MeanDose`
     - Penalizes the mean dose in the structure.
   * - :class:`~pyRadPlan.optimization.objectives.MaxDVH`
     - Penalizes violation of a maximum DVH constraint (Dx < limit).
   * - :class:`~pyRadPlan.optimization.objectives.MinDVH`
     - Penalizes violation of a minimum DVH constraint (Dx > limit).
   * - :class:`~pyRadPlan.optimization.objectives.EUD`
     - Equivalent Uniform Dose penalty (generalized EUD formulation).
   * - :class:`~pyRadPlan.optimization.objectives.SquaredMimicking`
     - Penalizes deviation from a reference dose distribution (plan mimicking).

Attaching objectives to structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing

    ptv = next(v for v in cst.vois if v.voi_type == "TARGET")
    ptv.objectives = [
        SquaredDeviation(priority=1000, d_ref=60.0, quantity="physical_dose"),
    ]

    oar = next(v for v in cst.vois if v.name == "Spinal_Cord")
    oar.objectives = [
        SquaredOverdosing(priority=500, d_max=45.0, quantity="physical_dose"),
    ]

Objectives are pydantic models, so they can be serialized and shared as JSON.

Literal quantities and legacy conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current automatic conversion is a legacy, plan-wide mode and remains enabled by default for
compatibility with existing ion-planning workflows. When the dose influence data carries a
biological model (e.g. a constant RBE for protons or a variable RBE model for ions), it decides
on the ``"physical_dose"`` objectives as a whole:

- If no objective uses a biological quantity (``"rbe_x_dose"``, ``"effect"``, ...), all
  physical-dose objectives are converted to ``"rbe_x_dose"``.
- If some objectives already use a biological quantity, the mix is taken as intended and the
  physical-dose objectives stay on physical dose.

Both outcomes are logged as a warning, which the GUI also shows in a dialog. Objectives on any
other quantity, such as LET, are always used literally. The conversion does not convert
prescription values between physical dose, effect, or other biological semantics. To use every
objective quantity literally, disable the conversion explicitly:

.. code-block:: python

    pln.prop_opt["convert_dose_objectives"] = False

    ptv.objectives = [
        SquaredDeviation(quantity="effect", d_ref=4.0),
    ]
    oar.objectives = [
        SquaredOverdosing(quantity="physical_dose", d_max=2.0),
    ]

With conversion disabled, reference parameters are interpreted in the selected quantity's
semantics. The independent ``dose_convention`` normalization described below still applies.
Explicit automatic intent and biological prescription conversion are deferred to the future
``dose_auto`` strategy API.

.. _concept_dose_convention:

Dose convention
~~~~~~~~~~~~~~~

Dose influence matrices describe one fraction. ``pln.dose_convention`` states how dose values
in objectives (``d_ref``, ``d_max``, ...) and in reported results are to be read:

``"per_fraction"`` (default)
    Objective doses are fraction doses and results are reported per fraction; nothing is
    rescaled.

``"total"``
    Objective references (scalar values and reference images such as the ``d_ref`` of
    ``SquaredMimicking``) refer to the whole course and are converted to one fraction when the
    problem is set up (the objectives in the structure set are left untouched); results are
    converted to the whole course.

Both conversions follow how each quantity accumulates over ``n = pln.num_of_fractions``
identical fractions, declared by its ``fraction_exponent`` (total = ``n**fraction_exponent``
× per fraction):

.. list-table::
   :header-rows: 1

   * - Quantity
     - Total course
     - Objective reference used for optimization
   * - ``physical_dose``, ``rbe_x_dose``, ``effect``, ``alpha_dose``, ``let_dose``
     - ``n`` × per fraction
     - total / ``n``
   * - ``sqrt_beta_dose``
     - ``sqrt(n)`` × per fraction, so that the total effect remains
       ``alpha_dose + sqrt_beta_dose**2``
     - total / ``sqrt(n)``
   * - ``let``
     - unchanged
     - unchanged

A custom quantity has to declare its ``fraction_exponent`` to be used with ``"total"``;
otherwise setting up the problem raises an error rather than guessing.

The optimizer logs which interpretation it applies.

.. _concept_robust_optimization:

Robust optimization
-------------------

A plan that is optimized on the nominal scenario only can degrade when the patient is set up
with an offset, the particle range deviates or the anatomy moves. *Robust optimization* takes
the uncertainty into account by evaluating the objectives on a
:ref:`scenario set <concept_uncertainty>` instead of a single dose distribution. The scenario
set comes from the plan's scenario model and is attached to the influence data
(``dij.scenarios``); how the scenarios enter an objective is chosen per objective through its
``robustness`` option.

Concepts
~~~~~~~~

Let :math:`f_s` be the value of an objective on the quantity of scenario :math:`s`, and
:math:`w_s` the normalized weight of that scenario in the set. The robustness operator of an
objective combines the per-scenario values into the value the optimizer sees.

*Objective-wise* operators act on one objective at a time. The *expected value* averages
:math:`f_s` over the scenarios with the weights :math:`w_s`, i.e. it optimizes the average
objective. The *objective-wise worst case* takes the maximum of :math:`f_s` over the scenarios
for each objective separately, so every objective is protected against its own worst scenario.

*Composite* operators act on all objectives that carry the operator. The *composite worst case*
first sums the (priority-weighted) objectives of each scenario and then takes the maximum over
scenarios, so the optimizer minimizes the plan's worst scenario as a whole. This is the
classical formulation for target coverage together with sparing of organs at risk.

*Voxel-wise* operators take the worst case per voxel before the objective is evaluated: the
minimum dose over scenarios in a target and the maximum in an organ at risk, which mimics a
dose distribution that is as bad as the worst scenario in every voxel. The inverted variant
takes the maximum in a target and the minimum in an organ at risk. Voxel-wise operators are not
available for 4D scenario sets, whose CT phases differ in anatomy.

*Probabilistic* operators optimize the statistics of the dose over the scenarios. Besides the
expected value, *expected quantity* evaluates the objective on the expected dose
:math:`\bar q = \sum_s w_s q_s`, and the :class:`~pyRadPlan.optimization.objectives.Variance`
objective penalizes the spread of the dose over the scenarios (see below).

Setting ``robustness``
~~~~~~~~~~~~~~~~~~~~~~

Every objective has a ``robustness`` option, a string that defaults to ``"none"``:

.. code-block:: python

    from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing

    ptv.objectives = [
        SquaredDeviation(priority=1000, d_ref=60.0, robustness="composite_worst_case"),
    ]
    oar.objectives = [
        SquaredOverdosing(priority=500, d_max=45.0, robustness="expected_value"),
    ]

    SquaredDeviation.available_robustness()   # names accepted by the objective

The accepted names are listed below. The matRad names and short forms are accepted as well and
normalized to the canonical name (``"OWC"`` becomes ``"objective_worst_case"``). When an
objective is exported to matRad, the matRad name is written.

.. list-table::
   :header-rows: 1
   :widths: 24 22 14 40

   * - ``robustness``
     - matRad name / aliases
     - Scope
     - Value optimized
   * - ``"none"``
     -
     - objective
     - The nominal scenario only. In a 4D set, one nominal scenario per CT phase, summed.
   * - ``"expected_value"``
     - ``STOCH``, ``stochastic``, ``probabilistic``
     - objective
     - Weighted average of the objective over all scenarios.
   * - ``"expected_quantity"``
     - ``PROB``
     - objective
     - The objective evaluated on the weighted average of the quantity.
   * - ``"voxel_worst_case"``
     - ``VWWC``
     - objective
     - The objective evaluated on the voxel-wise worst quantity: minimum over scenarios in
       targets, maximum in organs at risk.
   * - ``"voxel_worst_case_inverted"``
     - ``VWWC_INV``
     - objective
     - As ``voxel_worst_case`` with maximum in targets and minimum in organs at risk.
   * - ``"objective_worst_case"``
     - ``OWC``
     - objective
     - Maximum over scenarios of the objective, per objective.
   * - ``"composite_worst_case"``
     - ``COWC``
     - composite
     - Maximum over scenarios of the sum of all objectives carrying this operator.

A few details differ from matRad:

- ``"expected_value"`` uses the set's normalized aggregation weights ``weights``, whereas
  matRad's ``STOCH`` multiplies by the raw scenario densities. The values differ by a constant
  factor, which does not change the optimum of a single operator but matters when several
  operators are mixed.
- matRad's ``PROB`` adds a variance term built from large covariance matrices of the influence
  data. pyRadPlan drops that term with a warning, so ``"expected_quantity"`` evaluates only the
  expected-quantity part. Use the :class:`~pyRadPlan.optimization.objectives.Variance`
  objective to penalize the spread of the dose.
- A structure type other than target or organ at risk (``EXTERNAL``, ``HELPER``) is treated
  like an organ at risk by the voxel-wise operators, with a warning.

Objectives with ``robustness="none"`` evaluate only the nominal scenario(s) of the set, even
when the influence data contains many scenarios, so a plan that mixes robust and non-robust
objectives optimizes the non-robust goals on the nominal dose.

Each quantity computes only the scenarios the objectives reading it need, so an optimization
with every objective at ``robustness="none"`` on a worst-case influence data set evaluates the
nominal scenario alone in every iteration. The planning problem logs the scenarios per quantity
when it initializes. ``pln.prop_opt["evaluate_all_scenarios"] = True`` makes every quantity
evaluate the whole set regardless of the objectives, for instance to inspect all scenario doses
of the result.

Smooth maximum
~~~~~~~~~~~~~~

The worst-case operators need a maximum over scenarios, which is not differentiable. By default
pyRadPlan uses a smooth approximation, selected on the planning problem through
``pln.prop_opt`` (matRad's ``useMaxApprox`` and ``p``):

.. code-block:: python

    pln.prop_opt = {
        "problem": "nonlin_fluence",
        "max_approximation": "logsumexp",   # "logsumexp" (default), "pnorm" or "none"
        "p_norm": 30.0,                     # exponent of the p-norm approximation
    }

``"logsumexp"`` is the log-sum-exp approximation, ``"pnorm"`` a normalized p-norm with the
exponent ``p_norm``, and ``"none"`` the exact maximum, whose gradient follows the currently
worst scenario only. The exact maximum is not smooth, so the approximations are usually the
better choice for gradient-based solvers.

The ``Variance`` objective
~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~pyRadPlan.optimization.objectives.Variance` (name ``"Variance"``) is a
*scenario-aware* objective: instead of being evaluated scenario by scenario, it receives the
quantity of the whole scenario set. It minimizes the mean over the structure's voxels of the
weighted variance of the quantity over the scenarios,

.. math::

    f = \frac{1}{n} \sum_{v} \sum_s w_s \left(q_{s,v} - \bar q_v\right)^2 ,

with :math:`\bar q_v = \sum_s w_s q_{s,v}`. It always evaluates the whole set and ignores the
``robustness`` option. It is evaluated matrix-free from the per-scenario quantities, so no
covariance matrices of the influence data are needed.

.. code-block:: python

    from pyRadPlan.optimization.objectives import Variance

    ptv.objectives = [
        SquaredDeviation(priority=1000, d_ref=60.0, robustness="expected_value"),
        Variance(priority=100),
    ]

Worked example
~~~~~~~~~~~~~~

The following sketch optimizes a plan on a worst-case scenario grid. The target is protected
by the composite worst case, the organ at risk by the expected value:

.. code-block:: python

    from pyRadPlan import IonPlan, calc_dose_influence, fluence_optimization
    from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing

    pln = IonPlan(
        radiation_mode="protons",
        scenarios={"model": "wcScen", "combinations": "all"},
    )

    dij = calc_dose_influence(ct, cst, stf, pln)    # one influence matrix per scenario

    ptv = next(v for v in cst.vois if v.voi_type == "TARGET")
    ptv.objectives = [
        SquaredDeviation(priority=1000, d_ref=60.0, robustness="composite_worst_case"),
    ]

    oar = next(v for v in cst.vois if v.name == "Spinal_Cord")
    oar.objectives = [
        SquaredOverdosing(priority=500, d_max=45.0, robustness="expected_value"),
    ]

    fluence = fluence_optimization(ct, cst, stf, dij, pln)

The optimization reads the scenario set from the influence data and evaluates each scenario's
dose through that scenario's influence matrix.

Per-scenario results
~~~~~~~~~~~~~~~~~~~~

To inspect how the optimized plan behaves in each scenario, ask for the scenario results when
computing the result images:

.. code-block:: python

    result = dij.compute_result_ct_grid(fluence, scenarios=True)

    nominal = result["physical_dose"]
    per_scenario = result["physical_dose_scenario"]   # one image per scenario
    third = per_scenario[3]                           # index = flat index in dij.scenarios

Next to the usual ``"<quantity>"`` and ``"<quantity>_beam"`` entries, the result holds a
``"<quantity>_scenario"`` list with one image per scenario. The position in the list is the
flat index of the scenario in the set, which is also the slot in the influence-matrix
containers of the Dij.

Robustness in the GUI
~~~~~~~~~~~~~~~~~~~~~

In the GUI, the objective editor has a *Robustness* column that offers the robustness names an
objective accepts, and the scenario model is chosen in the plan widget, where the "…" button
next to it opens the model's parameters (setup and range standard deviations, CT phases,
worst-case extent, grid points or sample count). While the dose influence is calculated, the
status line counts the computed scenarios ahead of the beams and rays. When the influence
data holds several scenarios, the result selector lists ``scenario i`` components next to the
beams, labelled by the scenario's realization, with the nominal scenario marked. The DVH widget
can overlay the DVH of every scenario behind the nominal one, which gives a band that shows
the plan's sensitivity to the uncertainties.

Adding a robustness operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Robustness operators are registered by name, like objectives and problems, so a custom operator
becomes available to every objective through ``robustness="<name>"``. An operator subclasses
:class:`~pyRadPlan.optimization.robustness.RobustnessOperator`, sets a ``name`` and a
``scope`` (``"objective"`` for one objective, ``"composite"`` for all objectives carrying the
name), and implements ``evaluate(entries, values)`` and ``gradient(entries, values)``. The
``entries`` are :class:`~pyRadPlan.optimization.robustness.RobustTermEntry` objects holding the
objective and its structure type; ``values[j]`` is the quantity of entry ``j`` as an array of
shape ``(S, n)`` over the operator's ``scenario_indices``. ``evaluate`` returns the scalar
value including priorities, ``gradient`` the derivative with respect to each entry's values,
one array per entry with the shape of ``values[j]``.

.. code-block:: python

    import array_api_compat
    from pyRadPlan.optimization.robustness import (
        RobustnessOperator,
        register_robustness_operator,
    )


    class FirstScenarioOnly(RobustnessOperator):
        name = "first_scenario_only"
        scope = "objective"

        def evaluate(self, entries, values):
            objective = entries[0].objective
            return objective.priority * objective.compute_objective(values[0][0])

        def gradient(self, entries, values):
            objective = entries[0].objective
            xp = array_api_compat.array_namespace(values[0])
            grad = xp.zeros_like(values[0])
            grad[0, ...] = objective.priority * objective.compute_gradient(values[0][0])
            return [grad]


    register_robustness_operator(FirstScenarioOnly)

Operators receive the values in the compute backend of the optimization and should use
``array_api_compat.array_namespace`` rather than NumPy directly.

Compute backend
---------------

The optimization problem runs internally against the
`Python Array API standard <https://data-apis.org/array-api/latest/>`_, so the compute backend
can be switched without modifying any algorithm code:

.. code-block:: python

    from pyRadPlan import settings

    settings.xp.prefer_gpu = False
    settings.xp.preferred_cpu_array_backend = "numpy"
    settings.xp.prefer_gpu = True
    settings.xp.preferred_gpu_array_backend = "cupy"

The quantity resolver chooses the current preferred namespace through
:func:`pyRadPlan.core.xp_utils.choose_array_api_namespace` and converts ``Dij`` matrices into
that namespace when quantities are resolved.
