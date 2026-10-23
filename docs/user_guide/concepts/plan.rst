.. _concept_plan:

The Plan Object
===============

The *plan* object (``pln``) is the central configuration object for a treatment plan. It carries
all settings that control how the beam geometry is generated, how the dose is calculated, and how
the optimization problem is set up.

Class hierarchy
---------------

All plan objects derive from :class:`~pyRadPlan.plan.Plan`, which itself derives from
:class:`~pyRadPlan.core.datamodel.PyRadPlanBaseModel` (the pydantic base for all pyRadPlan
data structures). Two concrete subclasses exist:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Supported radiation modes
   * - :class:`~pyRadPlan.plan.IonPlan`
     - ``"protons"``, ``"helium"``, ``"carbon"``, ``"oxygen"``, ``"VHEE"``
   * - :class:`~pyRadPlan.plan.PhotonPlan`
     - ``"photons"``

Creating a plan
---------------

.. code-block:: python

    from pyRadPlan import IonPlan, PhotonPlan

    # Minimal proton plan
    pln = IonPlan(radiation_mode="protons", machine="Generic")

    # Photon plan with custom fractionation
    pln = PhotonPlan(
        radiation_mode="photons",
        machine="Generic",
        num_of_fractions=25,
        prescribed_dose=50.0,  # Gy
    )

Key fields
----------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - Description
   * - ``radiation_mode``
     - —
     - Particle type. Determines which dose engine and biological model are available.
   * - ``machine``
     - ``"Generic"``
     - Machine identifier. Can be a string key (resolved from the machine library) or a
       full :class:`~pyRadPlan.machines.Machine` dict.
   * - ``num_of_fractions``
     - ``30``
     - Number of fractions.
   * - ``dose_convention``
     - ``"per_fraction"``
     - Whether objective dose parameters and reported result doses refer to one fraction
       (``"per_fraction"``) or to the whole course (``"total"``, i.e. divided/multiplied by
       ``num_of_fractions``). See :ref:`concept_dose_convention`.
   * - ``prescribed_dose``
     - ``60.0``
     - Total prescribed dose in Gy. Used as the reference dose for normalization.
   * - ``bio_model``
     - per mode
     - Biological model (see :ref:`concept_bio_model`). Defaults to ``"constant_rbe"`` for
       protons, ``"HEL"`` for helium, ``"alpha_beta_kernels"`` for carbon and oxygen, and
       ``"none"`` for photons and VHEE.
   * - ``mult_scen``
     - nominal
     - Uncertainty / robustness scenario model
       (:class:`~pyRadPlan.scenarios.ScenarioModel`). Defaults to the nominal scenario.

.. _concept_bio_model:

Biological model
----------------

``pln.bio_model`` selects the model that turns physical dose into RBE-weighted dose (see
:ref:`concept_dose_calculation` for what the dose engines compute from it). It accepts a model
name, a ``{"model": name, **parameters}`` dict, or a model instance from
:mod:`pyRadPlan.biology.models`; the plan validates it against the radiation mode and stores the
model instance:

.. code-block:: python

    from pyRadPlan.biology.models import Wedenberg, available_bio_models

    pln.bio_model = "MCN"                                  # McNamara, default parameters
    pln.bio_model = {"model": "constant_rbe", "rbe": 1.0}  # parametrised
    pln.bio_model = Wedenberg(p1=0.5)                      # instance

    [cls.model for cls in available_bio_models("protons")]

Each model declares the named data it needs through ``required_quantities`` and the evaluator
quantities it can produce through ``output_quantities``. Availability checks compare those
requirements against the quantities the selected machine tabulates plus the ones the dose engine
produces itself (FRED, for instance, scores LET without needing LET kernels); the output
declaration describes the intrinsic result returned by the model evaluator.

Before a calculation starts, the dose engine checks the reference photon parameters of all
structures: ``alpha_x`` and ``beta_x`` must be finite and non-negative for every model. Models
formulated in terms of the reference α/β ratio additionally declare which of them must be
strictly positive (``requires_positive_reference``), so a structure with ``beta_x = 0`` selected
together with ``"MCN"`` fails with a clear message instead of producing non-finite influence
matrix entries. Voxels outside every structure carry no reference parameters
(``alpha_x == beta_x == 0``) and evaluate to zero.

An evaluator separately declares its additive matrix outputs through
``influence_quantity_names`` and produces them with ``evaluate_influence()``. For LQ models these
are ``alpha_dose`` and ``sqrt_beta_dose``. The evaluator boundary can describe different
sufficient statistics, but the current fixed ``Dij`` schema cannot store additional names; the
dose engine reports those during setup. A future generic quantity registry is outlined in
``docs/development/dynamic_dij_quantities.md``. See :ref:`custom_biological_models` for the
complete extension contract and a registered external model example.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Name
     - Radiation modes
     - Description
   * - ``"none"``
     - all
     - Physical dose only.
   * - ``"constant_rbe"``
     - all
     - Constant RBE (``rbe=1.1``); ``rbe_x_dose = rbe * physical_dose``.
   * - ``"WED"``, ``"MCN"``, ``"CAR"``
     - protons
     - LET-based LQ models (Wedenberg, McNamara, Carabé) yielding per-voxel α/β.
   * - ``"LSM"``
     - protons, helium, carbon
     - Linear scaling of α with LET between two thresholds.
   * - ``"HEL"``
     - helium
     - Mairani helium model.
   * - ``"alpha_beta_kernels"`` (``"kernel_based_lq"``, ``"LEM"``)
     - protons, helium, carbon, oxygen
     - α/β depth kernels per tissue class from the machine data, of whatever provenance
       (e.g. LEM I) the machine declares.
   * - ``"alpha_beta_tables"`` (``"dose_average_alpha_beta"``)
     - protons, helium, carbon, oxygen
     - α/β dose-averaged from RBE tables over the machine's fragment fluence spectra.
   * - ``"zstar_kernels"`` (``"MKM"``, ``"zs_based_lq"``)
     - protons, helium, carbon, oxygen
     - α = α₀ + β₀·z\* from the machine's z\* kernels, α₀/β₀ from the tissue.

Models are lightweight parameter objects: ``pln.bio_model.parameters`` and
``pln.bio_model.to_dict()`` give the constructor arguments, so a plan with a model serialises
and round-trips like any other field. matRad names (``constRBE``, ``LEM``) are understood on
import and written by :meth:`~pyRadPlan.plan.Plan.to_matrad`. The former identifiers in
parentheses keep loading plans and workspaces.

Three models read per-tissue-class base data: ``"alpha_beta_kernels"`` and
``"zstar_kernels"`` read kernels from the machine file, while ``"alpha_beta_tables"`` reads an
RBE table (see :ref:`base_data_folders`) and averages it over the fragment fluence spectra
of the machine. Each structure's tissue properties must match a tissue class of that data;
see :ref:`voi_tissue_properties`. A machine may carry several sets of one kind of base data
(e.g. α/β kernels computed with LEM I and with mMKM); ``data_model`` (the provenance, e.g.
``"LEM_I"``) or ``kernel_set`` (the set's id) selects one, ``quantity_variants`` a LET or z\*
definition, e.g. ``{"model": "WED", "quantity_variants": {"let": "LETd"}}``. If none is
given, the set the structures' tissues resolve to, the machine's default set or its only set
is used. :func:`~pyRadPlan.biology.models.bio_model_options` lists what a machine offers, and
:meth:`~pyRadPlan.biology.models.BiologicalModel.check_tissues` reports structures that do not
resolve. :ref:`biological_base_data` explains formalisms, provenances and the metadata the
base data carries.

Algorithm configuration dictionaries
-------------------------------------

Three ``prop_*`` dictionaries configure the sub-components that are selected automatically
during the workflow. They are passed down to the generators, engines, and optimizers:

``prop_stf``
    Controls beam geometry generation (see :ref:`concept_steering`).
    Common keys: ``"gantry_angles"``, ``"couch_angles"``, ``"bixel_width"``,
    ``"generator"`` (e.g. ``"IMPT"``).

``prop_dose_calc``
    Selects and configures the dose calculation engine (see :ref:`concept_dose_calculation`).
    Common key: ``"engine"`` (e.g. ``"HongPB"`` for particles or ``"SVDPB"`` for photons).

``prop_opt``
    Selects and configures the optimization problem (see :ref:`concept_optimization`).
    Common key: ``"problem"`` (currently ``"nonlin_fluence"``).

.. code-block:: python

    pln = IonPlan(radiation_mode="protons", machine="Generic")
    pln.prop_stf = {"gantry_angles": [0, 90, 270], "bixel_width": 5}
    pln.prop_dose_calc = {"engine": "HongPB"}
    pln.prop_opt = {"problem": "nonlin_fluence", "solver": "scipy"}

Pydantic validation and serialization
--------------------------------------

Because :class:`~pyRadPlan.plan.Plan` is a pydantic model, all field values are validated on
assignment, invalid configurations raise informative errors immediately, and the entire object
can be serialized to JSON:

.. code-block:: python

    import json
    print(json.dumps(pln.model_dump(), indent=2))

This also makes plan objects straightforwardly embeddable in LLM prompts for AI-assisted
treatment planning research.

matRad interoperability
-----------------------

.. code-block:: python

    matrad_pln = pln.to_matrad()  # dict compatible with matRad's pln struct
