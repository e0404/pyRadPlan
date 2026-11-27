ai.agents
=========

.. currentmodule:: pyRadPlan.ai.agents

.. autosummary::
   :toctree: generated/ai/agents/

   AiSettings


Agents
------

.. autosummary::
   :toctree: generated/ai/agents/

   generate_beam_angles
   beam_angles_system_prompt
   generate_voi_objectives


Planning Agent
--------------

The tool-using planning agent, see :ref:`concept_planning_agent`. It is a research feature and
not for clinical use.

Running
~~~~~~~

.. autosummary::
   :toctree: generated/ai/agents/

   run_planning_agent
   continue_planning_agent
   PlanningState
   PlanningReport
   create_planning_agent
   planning_tools

Goals and constraint tables
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/ai/agents/

   Goal
   Protocol
   GoalFile
   load_goals
   save_goals
   available_constraint_tables
   load_constraint_table
   site_goals
   eqd2
   physical_from_eqd2
   convert_fractionation

Plans and selection
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/ai/agents/

   PlanRecord
   evaluate_protocol
   plan_score
   select_distinct_plans

Helper structure recipes
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/ai/agents/

   RingRecipe
   MarginRecipe
   BooleanRecipe
   IsodoseRecipe

Errors
~~~~~~

.. autosummary::
   :toctree: generated/ai/agents/

   PlanningRunError
   PlanningError
   UnknownVoiError


Other Functions
---------------

.. autosummary::
   :toctree: generated/ai/agents/

   load_ai_env
   available_models
   resolve_model
   cst_context_summary
