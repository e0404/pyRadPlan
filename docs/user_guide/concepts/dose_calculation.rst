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
resolved during optimization. Every engine, photon engines included, stores the biological model it used as
``dij.bio_model``; for a constant-RBE model no matrices are needed and the constant is
available as ``dij.rbe``. An engine that cannot compute a model's declared outputs (for
example a photon engine with an LQ model) still records the model but logs a warning that
only physical dose is available. The model decides how ``rbe_x_dose`` is derived (LQ inversion of
the effect, or ``rbe * physical_dose``).

Biological models and LET
-------------------------

Which of the additional matrices a particle engine computes follows from ``pln.bio_model``
(see :ref:`concept_bio_model`) and the machine data:

* Supported biological influence matrices are computed when the evaluator declares them. The
  current ``Dij`` schema supports the ``alpha_dose`` / ``sqrt_beta_dose`` declared by the in-tree
  LQ evaluators.
* ``let_dose`` is computed when the machine provides LET kernels (pencil beam) or the model
  requires LET (FRED, which scores LET only on request). Because FRED scores LET itself, an
  LET-based model can be used with a machine that carries no LET kernels; the pencil-beam
  engines interpolate the machine's LET kernels and therefore need them.

Both can be overridden through the engine parameters ``calc_bio_dose`` and ``calc_let``,
which accept ``"auto"`` (default), ``True`` or ``False``:

.. code-block:: python

    pln.bio_model = "MCN"
    # keep only physical dose and LET; evaluate the LET-based model later
    pln.prop_dose_calc = {"engine": "HongPB", "calc_bio_dose": False}

Setting ``calc_bio_dose=True`` without an evaluator providing influence quantities raises an
error. An evaluator declaring a quantity that ``Dij`` cannot currently store also fails during
setup. Switching ``calc_let`` off does not affect LET-based models, which still receive the LET
kernels internally. The FRED engine derives biological matrices from its scored LET and therefore
supports LET-based models only.

During the calculation the model is bound to the machine and the tissue properties of the
voxels (from the structure set, with ``dij.alphax`` / ``dij.betax`` holding their reference
parameters) through a :class:`~pyRadPlan.biology.models.BioModelEvaluator`. Kernel- and
table-based models carry base data for a finite set of tissue classes, and select the class
of each voxel by exact match against them
(:class:`~pyRadPlan.biology.models.ExactClassLookup`). What is compared are the fields
identifying a tissue class of the model, not always the reference parameters: LEM base data
is tabulated per ``(alpha_x, beta_x, d_t, r_nucleus, version)``, while the MKM z\* kernels
only depend on the ``(r_nucleus, r_domain)`` geometry — its ``alpha_0`` / ``beta_0`` are taken
from the tissue per voxel. Only the tissue classes of the kernel set a model resolves to are
considered, and a tissue of another provenance (e.g. MKM parameters for LEM I kernels) is
rejected, so a machine carrying classes for several models cannot serve the base data of
one through the tissue class of another. The engine runs
:meth:`~pyRadPlan.biology.models.BiologicalModel.check_tissues` before the calculation. LEM
kernels identify their rows by the reference ``alpha_x`` / ``beta_x`` stored with them; the
machine's declared tissue classes are matched onto these rows by value, and a machine without
declared classes uses the kernels' own. See :ref:`voi_tissue_properties` for setting the
structures up and :ref:`biological_base_data` for the kernel sets and their metadata.

A model that cannot be applied to a calculation — the machine lacks its kernel quantities, a
voxel's tissue matches no class of the base data, or the reference parameters are outside the
model's domain — only fails the calculation when biological dose is requested explicitly
(``calc_bio_dose=True``). With ``"auto"`` or ``False`` a warning names the reason and the
physical dose influence is computed alone: the dij then records no model
(``dij.bio_model is None``) and carries no biological matrices, so ``rbe_x_dose`` is reported
as unavailable and the legacy objective conversion keeps physical-dose objectives on physical
dose. This keeps physical-dose workflows working with the per-modality default models.
Invalid reference coefficients (negative, infinite or NaN ``alpha_x`` / ``beta_x``) always
raise.

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
    physical_dose_ct = result.physical_dose.values  # SimpleITK image on the CT grid

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
       Validated for protons only; ion calculations warn that their doses do not yet
       reproduce the reference.
   * - ``"AIBeamlet"``
     - AI beamlet engine: a trained model predicts each beamlet's dose (see
       :ref:`concept_ai_dose_calculation`). Research feature.

.. code-block:: python

    pln.prop_dose_calc = {"engine": "HongPB"}
    dij = calc_dose_influence(ct, cst, stf, pln)

.. _concept_ai_dose_calculation:

AI-based dose calculation
-------------------------

The ``"AIBeamlet"`` engine replaces the analytical kernel with a trained model: for every
beamlet it samples the CT on a beamlet-local grid, lets the model predict the dose cuboid and
interpolates that cuboid onto the dose grid. Everything around it -- steering, the ``Dij``, the
optimization -- is unchanged, so switching engines is one line plus the model to use:

.. code-block:: python

    pln.prop_dose_calc = {
        "engine": "AIBeamlet",
        "model": "DKFZ-RadOpt/pyRadPlan-dosecalc-Det-proton-lung",
    }
    dij = calc_dose_influence(ct, cst, stf, pln)

.. warning::

   AI dose calculation is a **research feature and not intended for clinical use**. A model is
   valid only for the anatomy, radiation mode and machine it was trained on; used outside that
   domain it may be confidently wrong, and pyRadPlan cannot detect this for you. Verify
   predicted dose against an established engine before drawing conclusions from it.

Models and how they are found
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models are addressed as ``"<org>/<model>"`` and resolved by
:ref:`concept_ai_modelhub`: a copy on disk (or in the HuggingFace cache) is used as-is,
otherwise the model is downloaded on first use.
:func:`~pyRadPlan.ai.modelhub.list_local_models` reports what is already available offline.
Loading a model runs the Python code shipped with it, so the engine applies the same rule as
:func:`~pyRadPlan.ai.modelhub.load_model`: the calculation is refused unless you opted in with
``settings.ai.modelhub_trust_remote_code = True`` (environment variable
``PYRADPLAN_AI_MODELHUB_TRUST_REMOTE_CODE=1``). A plan cannot opt in by itself; enable it only
for model sources you trust.

The engine reads the repository's ``model_config.json`` and adapts to what it declares, so
pyRadPlan carries no per-model code:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Declaration
     - Effect
   * - ``metadata.dose_calc.grid_parameters``
     - Spacing, lateral width and depth range of the beamlet-local sampling grid.
   * - ``metadata.dose_calc.quantities``
     - Which matrices the model fills, e.g. ``physical_dose``, ``physical_dose_var``,
       ``let_dose``. A multi-output model returns them from its preprocessor as a dict.
   * - ``metadata.training.modality``
     - The radiation mode the model was trained for. A plan using another mode is rejected.
   * - ``metadata.training.energy_range_mev`` / ``trained_machine``
     - Checked against the plan; a mismatch is a warning, since the calculation can still run.
   * - ``metadata.requirements``
     - Packages the model's code needs (``torch``, ``onnxruntime``, ...). The engine itself
       requires no particular ML framework and only checks what the model declares.

Availability
~~~~~~~~~~~~

The engine is always offered in :func:`~pyRadPlan.dose.engines.get_available_engines` and in
the GUI's engine list, so a model can be picked at all. It reports itself unavailable only when
the plan names a model that is already on disk and cannot be used -- trained for another
modality, or its declared packages are not installed -- or names a model that is not an
``"<org>/<model>"`` id. The reason is part of the answer:

.. code-block:: python

    from pyRadPlan.dose.engines import AIBeamletEngine

    available, reason = AIBeamletEngine.is_available(pln, machine=None)
    # e.g. (False, "Model '<org>/<model>' was trained for 'VHEE', but the plan uses 'protons'.")

This check only consults the file system and never contacts the hub. A model that is not on
disk is resolved -- and downloaded, unless ``PYRADPLAN_AI_MODELHUB_OFFLINE=1`` -- only when a
calculation runs; if it cannot be found, that calculation fails with an error naming the model.
The suggestions in the GUI's engine configuration dialog do list the models of the configured
hub organization.

Devices and backends
~~~~~~~~~~~~~~~~~~~~

The engine's arrays follow ``settings.xp`` like every other engine, and the model is loaded onto
the device those arrays live on, so no data crosses devices per beamlet. Choosing the array
backend the model's framework uses (``settings.xp.preferred_gpu_array_backend = "torch"`` for a
torch model) additionally avoids a conversion per ray. Without a device of its own -- NumPy, for
instance -- the model is placed on ``PYRADPLAN_AI_MODELHUB_DEVICE`` (default ``cpu``).

A runnable end-to-end plan is in the
:doc:`AI dose calculation tutorial </tutorials/examples/ai_dosecalc>`.

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

When the plan carries an uncertainty scenario model (``pln.scenarios``, see
:ref:`concept_uncertainty`), ``calc_dose_influence`` resolves it into a
:class:`~pyRadPlan.scenarios.ScenarioSet`, checks the set against the engine's
``supported_uncertainties`` and returns a ``Dij`` whose quantity matrices are flat object-array
containers with one influence matrix per computed scenario, addressed by the scenario's flat
index (``dij.scenarios`` records the set). Scenarios sharing CT phase and setup shift share the
ray tracing; the range perturbation is applied per scenario. The quantities evaluate every
scenario of the set and the optimization problem aggregates over them.
