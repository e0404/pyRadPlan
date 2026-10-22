.. _concept_uncertainty:

Uncertainty and Scenarios
=========================

Treatment plans are evaluated and optimized under uncertainty: the patient may be set up with
an offset, the particle range may deviate from the planned one, and the anatomy may be in a
different phase of a 4D CT. pyRadPlan describes these uncertainties with three building blocks
in :mod:`pyRadPlan.scenarios`:

- **Uncertainty dimensions** say *what* is uncertain and with which parameters.
- A **scenario set** is a concrete list of realizations, with a probability and an aggregation
  weight each. It is what dose calculation, the quantities and the :class:`~pyRadPlan.dij.Dij`
  consume.
- A **scenario model** is the configuration that generates a scenario set from the dimensions,
  e.g. the nominal scenario, a worst-case grid or random samples.

This mirrors matRad's ``multScen`` models (``nomScen``, ``wcScen``, ``impScen``, ``rndScen``)
but does not hard-code the three matRad axes (CT phase, setup, range): a set is queried by
dimension name, and the axis layout is a view that can be re-tied at will.

Uncertainty dimensions
----------------------

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Dimension
     - Components
     - Parameters
   * - :class:`~pyRadPlan.scenarios.SetupUncertainty` (``"setup"``)
     - ``x``, ``y``, ``z`` in mm
     - ``sd`` per direction (default 2.25 mm), optional ``shift_mean``.
   * - :class:`~pyRadPlan.scenarios.RangeUncertainty` (``"range"``)
     - ``abs`` in mm, ``rel`` as fraction
     - ``abs_sd`` (1 mm), ``rel_sd`` in percent (3.5 %), ``combined`` (grid models move both
       components together, matRad's ``combineRange``).
   * - :class:`~pyRadPlan.scenarios.CTPhaseUncertainty` (``"ct_phase"``)
     - 0-based phase ``index``
     - ``phases`` and their ``probabilities`` (matRad's ``ctScenProb``).

Every dimension declares what it *affects*: ``"geometry"`` (ray tracing, isocenter),
``"range"`` (radiological depths) or ``"voxel_params"`` (per-voxel parameters such as
:math:`\alpha_x/\beta_x` or the tissue class). Dose engines use this to group scenarios that
share expensive inputs, and they refuse a set with a dimension they cannot apply, so an
unsupported uncertainty never silently turns into the nominal case. New dimensions (for example
a biological uncertainty of the tissue parameters) register themselves with
:func:`~pyRadPlan.scenarios.register_dimension` and need no change to the consumers.

The joint distribution of the continuous components is an
:class:`~pyRadPlan.scenarios.IndependentGaussian` by default (matRad's probability model); a
:class:`~pyRadPlan.scenarios.MultivariateGaussian` with a full covariance can be passed to any
model through its ``distribution`` field.

Scenario models
---------------

.. code-block:: python

    from pyRadPlan.scenarios import NominalScenarioModel, SetupUncertainty

    model = NominalScenarioModel(setup=SetupUncertainty(sd=(3.0, 3.0, 3.0)))
    scen_set = model.scenarios()          # generated lazily, cached until the model changes

Models are pydantic objects and accept the flat matRad parameter names as well
(``rangeRelSD``, ``rangeAbsSD``, ``shiftSD``, ``ctScenProb``, ``wcSigma``), so a matRad
``multScen`` struct validates directly. ``model.to_matrad()`` returns that flat struct again.
The model goes into the plan:

.. code-block:: python

    pln = IonPlan(radiation_mode="protons", scenarios={"model": "wcScen", "combinations": "all"})

``mult_scen`` (matRad's name) is still accepted as an alias of ``scenarios``. The dose
calculation also takes a set directly: ``calc_dose_influence(ct, cst, stf, pln, scenarios=...)``.

A set can also be built without a model from explicit points, which is the hook for adaptive
schemes (e.g. quadrature points of a polynomial chaos expansion) and for single hand-picked
scenarios:

.. code-block:: python

    from pyRadPlan.scenarios import ScenarioSet, CTPhaseUncertainty, RangeUncertainty

    scen_set = ScenarioSet.from_points(
        (CTPhaseUncertainty(), SetupUncertainty(), RangeUncertainty()),
        {"setup": [[0, 0, 0], [3, 0, 0]], "range": [[0, 0], [0, 0]]},
    )
    more = scen_set.extend(ScenarioSet.from_points(..., {"range": [[2.0, 0.0]]}))

Dimensions missing from ``points`` take their nominal value; probabilities come from the
distribution and the weights are the normalized probabilities unless given explicitly.

Scenario sets
-------------

A :class:`~pyRadPlan.scenarios.ScenarioSet` is immutable. Each
:class:`~pyRadPlan.scenarios.Scenario` carries its realization per dimension, ``probability``,
``weight``, ``tags`` (``"nominal"`` marks the unperturbed scenario) and a flat ``index``. The
flat index is the only stable handle: it is the slot of the scenario in every influence-matrix
container of the Dij (``dij.physical_dose.flat[index]``) and in the quantity caches.

.. code-block:: python

    len(scen_set), scen_set[1], scen_set.nominal_index()
    scen_set.values("setup")                # (n, 3) shifts
    scen_set.extent("range")                # largest |abs|, |rel| over the set (margins)
    scen_set.group_by("ct_phase", "setup")  # scenarios sharing ray tracing
    scen_set.voxel_param_set(i)             # column of alphax/betax used by scenario i

Scenarios can be *derived* instead of computed: a scenario with a
:class:`~pyRadPlan.scenarios.ScenarioBasis` is a linear combination of computed scenarios
(``coefficients`` map flat indices to factors). Dose engines skip it and the quantities evaluate
the combination; a basis with a single unit coefficient shares the referenced scenario's
matrices, which is how a scenario that differs only in per-voxel parameters avoids a second
influence matrix.

Dimensional view
~~~~~~~~~~~~~~~~

For legible access the set offers a grid view (:class:`~pyRadPlan.scenarios.ScenarioLayout`).
Its axes tie component keys; matRad's layout ties ``ct_phase``, ``setup`` (x, y, z together)
and ``range`` (abs, rel together):

.. code-block:: python

    layout = scen_set.layout             # default axes from the model
    layout.shape, layout.mask            # grid shape, occupied cells (matRad scenMask)
    scen_set.isel(setup=1)               # scenarios at grid position 1 of the setup axis
    scen_set.sel(setup=(3, 0, 0))        # scenarios with that shift (tolerance match)
    scen_set.linear_index(ct_phase=0, setup=1, range=0)
    scen_set.grid_view(dij.physical_dose)  # object array of matrices on the grid, no copy
    scen_set.with_axes("setup.x", "setup.y", "setup.z")   # re-tie, e.g. one axis per direction

Any tying is valid for any set. A tying that does not match how the model varied the
dimensions only yields a sparse mask (random samples on matRad's three axes are a diagonal),
never an error. Grid positions change when a set is extended, so persisted references use the
flat index. The optional :meth:`~pyRadPlan.scenarios.ScenarioSet.to_xarray` exports the index
grid as a labelled ``xarray.DataArray`` when xarray is installed.

Scenarios in dose calculation and quantities
--------------------------------------------

``calc_dose_influence`` resolves the plan's model into a set, checks it against the engine's
``supported_uncertainties`` and computes one influence matrix per computed scenario, grouping
the scenarios that share ray tracing (same CT phase and setup shift) and applying the range
perturbation inside the group. The resulting Dij records the set as ``dij.scenarios``. The
per-voxel parameter arrays (``alphax``, ``betax``, ``tissue_class_ix``) carry one column per
*voxel parameter set*, i.e. per unique combination of the dimensions affecting them (today the
CT phase); ``scen_set.voxel_param_set(i)`` maps a scenario to its column.

Quantities (:ref:`concept_quantities`) evaluate all computed scenarios of ``dij.scenarios`` by
default and expose them in the same flat layout; the optimization problem aggregates over them.
Robust objectives (expected value, worst case) build on the set's ``weights`` and ``tags`` and
are not part of the first release of the scenario system.

matRad interoperability
-----------------------

``pln.multScen`` structs validate into the matching model (1-based ``ctScenProb`` indices are
converted). matRad Dij cell arrays shaped ``(ct, shift, range)`` are flattened on import, and a
Dij with a scenario set exports its matrices on matRad's grid through the layout view.
