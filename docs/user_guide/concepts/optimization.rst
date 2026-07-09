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
   * - ``"scipy"``
     - SciPy minimization (``scipy.optimize.minimize``). Multiple sub-methods available
       (L-BFGS-B, SLSQP, etc.). Lighter dependency, good for smaller problems.

.. code-block:: python

    pln.prop_opt = {
        "problem": "nonlin_fluence",
        "solver": "scipy",
    }

The currently registered fluence problem constrains fluence weights to be non-negative.

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
