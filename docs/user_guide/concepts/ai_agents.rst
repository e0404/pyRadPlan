.. _concept_ai_agents:

AI Agents
=========

pyRadPlan ships an optional ``ai_agents`` module that uses
`pydantic-ai <https://ai.pydantic.dev/>`_ to integrate large-language-model (LLM) reasoning
into the treatment planning workflow.  The agents follow the same pydantic data-model
conventions as the rest of pyRadPlan and return fully-validated plan and structure-set
objects, so their output can be fed directly into the rest of the pipeline.

.. note::

   The ``ai_agents`` module is an optional feature.  Before using it you need to install
   the optional dependencies and set an API key for your chosen LLM provider:

   .. code-block:: bash

       pip install pydantic-ai pydantic-settings python-dotenv

   Then export your provider API key, for example:

   .. code-block:: bash

       export ANTHROPIC_API_KEY=sk-ant-...   # Anthropic / Claude
       export OPENAI_API_KEY=sk-...          # OpenAI
       export GOOGLE_API_KEY=...             # Google Gemini

   API keys can also be stored in a ``.env`` file in the working directory.

Overview
--------

The module exposes two high-level helper functions and a settings class:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Symbol
     - Purpose
   * - :func:`~pyRadPlan.ai_agents.generate_beam_angles`
     - Ask the LLM to suggest gantry (and couch) angles for a given treatment site and
       radiation mode, then write them back into the :class:`~pyRadPlan.plan.Plan`.
   * - :func:`~pyRadPlan.ai_agents.generate_voi_objectives`
     - Ask the LLM to propose optimization objectives for each VOI in a
       :class:`~pyRadPlan.cst.StructureSet` based on the treatment site and prescribed dose.
   * - :class:`~pyRadPlan.ai_agents.AiSettings`
     - Pydantic-settings class that reads the default model name from the environment
       variable ``PYRADPLAN_AI_MODEL`` or a ``.env`` file.

Model selection
---------------

The default model is ``claude-sonnet-4-5``.  You can override it globally via the
environment variable ``PYRADPLAN_AI_MODEL``, or per-call via the ``model=`` keyword:

.. code-block:: python

    import os
    os.environ["PYRADPLAN_AI_MODEL"] = "openai:gpt-4o-mini"

    from pyRadPlan import ai_agents

    # Check effective settings
    print(ai_agents.AiSettings().model)   # openai:gpt-4o-mini

    # Override for a single call
    pln = ai_agents.generate_beam_angles(pln, "prostate", model="gemini-2.0-flash")

pydantic-ai uses the provider prefix (``openai:``, ``anthropic:``, ``google-gla:``, …)
to select the backend automatically.  When no prefix is given the model string is passed
to the Anthropic backend.

Generating beam angles
----------------------

:func:`~pyRadPlan.ai_agents.generate_beam_angles` populates ``pln.prop_stf`` with
``gantry_angles`` and matching ``couch_angles`` (zeros unless the model suggests
non-coplanar beams):

.. code-block:: python

    from pyRadPlan import IonPlan, ai_agents

    pln = IonPlan(
        radiation_mode="protons",
        machine="Generic",
        num_of_fractions=30,
        prescribed_dose=60,
    )

    pln = ai_agents.generate_beam_angles(pln, treatment_site="prostate")

    print(pln.prop_stf["gantry_angles"])   # e.g. [0.0, 90.0, 180.0]

An optional ``additional_context`` string lets you pass extra clinical constraints:

.. code-block:: python

    pln = ai_agents.generate_beam_angles(
        pln,
        treatment_site="head and neck",
        additional_context="Patient has a hip replacement on the right side.",
    )

Generating optimization objectives
-----------------------------------

:func:`~pyRadPlan.ai_agents.generate_voi_objectives` iterates over the VOIs in a
:class:`~pyRadPlan.cst.StructureSet` and attaches LLM-suggested
:ref:`optimization objectives <concept_optimization>` to each structure.  By default any
previously assigned objectives are cleared first (``clear_existing=True``):

.. code-block:: python

    from pyRadPlan import load_tg119, IonPlan, ai_agents

    ct, cst = load_tg119()
    pln = IonPlan(
        radiation_mode="protons",
        machine="Generic",
        num_of_fractions=30,
        prescribed_dose=60,
    )

    cst = ai_agents.generate_voi_objectives(pln, cst, treatment_site="prostate")

    for voi in cst.vois:
        if voi.objectives:
            print(voi.name, [type(o).__name__ for o in voi.objectives])

The agent works directly on pyRadPlan's pydantic
:class:`~pyRadPlan.optimization.objectives.Objective` models: all registered objective
types (except image-based ones such as ``SquaredMimicking``) are offered to the LLM as a
discriminated union, so parameter names, bounds, defaults and descriptions are taken
straight from the objective definitions and the validated output consists of ready-to-use
objective instances. Objectives registered via
:func:`~pyRadPlan.optimization.objectives.register_objective` become available to the
agent automatically.

End-to-end example
------------------

The script ``examples/utils_ai_agents.py`` shows a complete workflow:

1. Load the TG119 phantom.
2. Use ``generate_beam_angles`` to obtain gantry angles for a prostate plan.
3. Use ``generate_voi_objectives`` to populate VOI objectives.
4. Run the standard dose calculation and fluence optimization with the AI-generated
   settings.
5. Visualize the result.

.. code-block:: python

    from dotenv import load_dotenv
    from pyRadPlan import (
        IonPlan, load_tg119, ai_agents,
        generate_stf, calc_dose_influence, fluence_optimization,
        plot_slice, DVHCollection,
    )

    load_dotenv()   # load API key from .env

    ct, cst = load_tg119()
    pln = IonPlan(radiation_mode="protons", machine="Generic",
                  num_of_fractions=30, prescribed_dose=60)
    pln.prop_opt = {"solver": "scipy"}
    pln.prop_dose_calc = {"dose_grid": ct.grid}

    pln = ai_agents.generate_beam_angles(pln, treatment_site="prostate")
    cst = ai_agents.generate_voi_objectives(pln, cst, treatment_site="prostate")

    stf = generate_stf(ct, cst, pln)
    dij = calc_dose_influence(ct, cst, stf, pln)
    fluence = fluence_optimization(ct, cst, stf, dij, pln)

    result = dij.compute_result_ct_grid(fluence)
    dvhs = DVHCollection.from_structure_set(cst, result["physical_dose"])
    plot_slice(ct=ct, cst=cst, overlay=result["physical_dose"])

Settings reference
------------------

.. autoclass:: pyRadPlan.ai_agents.AiSettings
   :members:

API reference
-------------

.. autofunction:: pyRadPlan.ai_agents.generate_beam_angles

.. autofunction:: pyRadPlan.ai_agents.generate_voi_objectives
