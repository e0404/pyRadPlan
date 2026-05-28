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
     - Number of fractions. Used for biological dose calculations.
   * - ``prescribed_dose``
     - ``60.0``
     - Total prescribed dose in Gy. Used as the reference dose for normalization.
   * - ``mult_scen``
     - nominal
     - Uncertainty / robustness scenario model
       (:class:`~pyRadPlan.scenarios.ScenarioModel`). Defaults to the nominal scenario.

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
