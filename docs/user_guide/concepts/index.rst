.. _concepts:

Concepts
========

This section explains the main building blocks of a pyRadPlan treatment planning workflow.
Each concept corresponds to one or more Python classes and to one step in the workflow shown
in :ref:`quickstart`.

.. toctree::
    :maxdepth: 2
    :caption: Concepts:

    plan
    patient_data
    steering
    dose_calculation
    optimization
    quantities
    visualization
    ai_agents

.. rubric:: Workflow at a glance

.. code-block:: text

    load_patient()          →  CT  +  StructureSet  (patient_data)
    IonPlan / PhotonPlan    →  Plan                  (plan)
    generate_stf()          →  SteeringInformation   (steering)
    calc_dose_influence()   →  Dij                   (dose_calculation)
    fluence_optimization()  →  fluence array         (optimization)
    dij.compute_result_*()  →  dose result dict      (quantities)
    plot_slice()            →  figure                (visualization)

    # optional AI-assisted steps
    ai_agents.generate_beam_angles()    →  Plan with gantry angles  (ai_agents)
    ai_agents.generate_voi_objectives() →  StructureSet with objectives
