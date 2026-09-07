.. _concept_dose_calculation:

Dose Calculation
================

Dose calculation in pyRadPlan produces a *dose influence matrix* (``dij``) that encodes the
dose deposited in every voxel by every beamlet. The influence matrix is the central data
structure for optimization: the dose for any fluence vector ``w`` is simply the matrix–vector
product ``dij · w``.

Two top-level functions are provided:

- :func:`~pyRadPlan.calc_dose_influence` — compute the full dose influence matrix (batch mode).
- :func:`~pyRadPlan.calc_dose_forward` — compute only the forward dose for a given fluence
  vector (faster, used for dose reconstruction after optimization).

.. code-block:: python

    from pyRadPlan import calc_dose_influence, calc_dose_forward

    # Full influence matrix (used before optimization)
    dij = calc_dose_influence(ct, cst, stf, pln)

    # Forward dose after optimization
    dij_fwd = calc_dose_forward(ct, cst, stf, pln, weights=fluence)

Both functions select the appropriate engine automatically from ``pln.prop_dose_calc["engine"]``.

The Dij object
--------------

:class:`~pyRadPlan.dij.Dij` stores the dose influence matrix together with the grid metadata
needed to relate dose voxels back to CT voxels.

Dose matrices
~~~~~~~~~~~~~

The primary attribute is ``physical_dose``. Influence matrices are stored in object-array
containers so that single- and multi-scenario calculations use the same layout; each contained
matrix has shape ``(num_voxels, num_bixels)`` and is commonly stored in CSC sparse format.
Additional matrices are available for biological dose calculation in particle therapy:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Attribute
     - Description
   * - ``physical_dose``
     - Physical dose contribution per voxel per beamlet (Gy / primary particle).
   * - ``let_dose``
     - LET-weighted dose matrix for LET-based optimization.
   * - ``alpha_dose``
     - Per-beamlet α-dose for the linear-quadratic (LQ) biological model.
   * - ``sqrt_beta_dose``
     - Per-beamlet √β-dose for the LQ model.

The set of available matrices determines which :ref:`quantities <concept_quantities>` can be
resolved during optimization.

Grid information
~~~~~~~~~~~~~~~~

Dose is typically computed on a coarser *dose grid* (for speed) and then interpolated back to
the *CT grid* for display:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Attribute
     - Description
   * - ``dose_grid``
     - :class:`~pyRadPlan.core.Grid` describing the voxel grid on which dose was computed.
   * - ``ct_grid``
     - :class:`~pyRadPlan.core.Grid` of the original CT image.
   * - ``num_of_voxels``
     - Number of voxels in the dose grid (= ``dose_grid.num_voxels``).
   * - ``total_num_of_bixels``
     - Number of beamlets (= columns of the dose matrix).

Reconstructing dose on the CT grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Compute dose for optimized fluence and resample to CT grid
    result = dij.compute_result_ct_grid(fluence)
    physical_dose_ct = result["physical_dose"]  # SimpleITK image on the CT grid

    # Inspect which quantities are available
    print(dij.quantities)   # e.g. ['physical_dose', 'alpha_dose', 'sqrt_beta_dose']

Dose engines
------------

The dose engine is selected via ``pln.prop_dose_calc["engine"]``. All engines inherit from a
common abstract base and implement the same interface, so switching between them requires only
one line:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Engine key
     - Description
   * - ``"SVDPB"``
     - SVD-accelerated pencil beam (photons). Improved lateral scatter modelling.
   * - ``"HongPB"``
     - Hong's particle pencil-beam model. Reference implementation for protons/ions.
   * - ``"FRED"``
     - `FRED <https://www.fred-mc.org>`_ Monte Carlo engine (requires FRED installation).

.. code-block:: python

    pln.prop_dose_calc = {"engine": "HongPB"}
    dij = calc_dose_influence(ct, cst, stf, pln)

Backend-agnostic computation
-----------------------------

Dose engine and quantity code use the
`Python Array API standard <https://data-apis.org/array-api/latest/>`_ where practical. The
preferred compute backend is the ``xp`` sub-configuration of the global
:data:`pyRadPlan.settings` and can be set via environment variables
(``PYRADPLAN_XP_PREFER_GPU``, ``PYRADPLAN_XP_PREFERRED_GPU_ARRAY_BACKEND``), a ``.env`` file, or
at runtime:

.. code-block:: python

    from pyRadPlan import settings

    settings.xp.prefer_gpu = True
    settings.xp.preferred_gpu_array_backend = "cupy"
    dij = calc_dose_influence(ct, cst, stf, pln)

The ``Dij`` object can also convert its influence matrices explicitly with
``dij.to_namespace("cupy")`` or another supported namespace.

.. _jit-compiled-kernels:

Jit-compiled kernels
~~~~~~~~~~~~~~~~~~~~

A handful of hot-path computations (currently in the Siddon ray tracer) are written once
as plain Array API code and additionally wrapped in
:func:`pyRadPlan.core.xp_utils.jittable`. This decorator gives such a kernel jit-compiled
execution paths per backend — a `numba <https://numba.pydata.org>`_ implementation for
NumPy, or the backend's own tracer (``jax.jit``, ``torch.compile``) for backends where the
kernel is written in a jit-traceable, static-shape style — while still falling back to the
plain implementation everywhere else, and whenever a compiled path is unavailable or fails.

Which backends actually take the jit-compiled path is controlled by
``settings.xp.jit_backends``, a comma-separated list (``PYRADPLAN_XP_JIT_BACKENDS``,
default ``"numpy,jax"``):

.. code-block:: python

    from pyRadPlan import settings

    settings.xp.jit_backends = "numpy,jax,torch"   # also let torch.compile try
    settings.xp.jit_backends = ""                  # disable jitting everywhere

The NumPy path requires the ``[numba]`` extra (``pip install "pyRadPlan[numba]"``); without
it, NumPy silently runs the plain implementation. If a backend's jit fails to compile or
raises on its first execution (this can happen with ``torch.compile`` when the platform's
Triton install is missing or incompatible), pyRadPlan emits a ``RuntimeWarning`` once and
continues with the plain implementation for the remainder of the run — jitting is always a
performance option, never a correctness requirement.

Multi-scenario support
-----------------------

When the plan carries a robustness scenario model (``pln.mult_scen``), ``calc_dose_influence``
returns a ``Dij`` whose quantity matrices are stored as object-array containers with one
influence matrix per scenario. The optimization problem is then responsible for aggregating over
the resolved scenario quantities.
