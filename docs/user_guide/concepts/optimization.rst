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
    Objective doses refer to the whole course and are divided by ``pln.num_of_fractions``
    when the problem is set up (the objectives in the structure set are left untouched);
    result doses are multiplied by ``num_of_fractions``.

The optimizer logs which interpretation it applies.

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
