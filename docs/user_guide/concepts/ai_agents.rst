.. _concept_ai_agents:

AI Agents
=========

pyRadPlan ships an optional ``ai.agents`` module that uses
`pydantic-ai <https://ai.pydantic.dev/>`_ to integrate large-language-model (LLM) reasoning
into the treatment planning workflow.  The agents follow the same pydantic data-model
conventions as the rest of pyRadPlan and return fully-validated plan and structure-set
objects, so their output can be fed directly into the rest of the pipeline.

.. note::

   The ``ai.agents`` module is an optional feature.  Before using it you need to install
   the optional dependencies and set an API key for your chosen LLM provider:

   .. code-block:: bash

       pip install pydantic-ai pydantic-settings python-dotenv

   Then export your provider API key, for example:

   .. code-block:: bash

       export ANTHROPIC_API_KEY=sk-ant-...   # Anthropic / Claude
       export OPENAI_API_KEY=sk-...          # OpenAI
       export GOOGLE_API_KEY=...             # Google Gemini

   API keys can also be stored in a ``.env`` file in the working directory or,
   e.g. for the installed desktop application, in the per-user
   ``~/.pyradplan/.env`` (see :doc:`/api/settings`).

   Instead of a cloud provider, the agents can use a local
   `Ollama <https://ollama.com/>`_ server, which needs no API key: set
   ``OLLAMA_BASE_URL`` to its OpenAI-compatible address (e.g.
   ``http://localhost:11434/v1``) and select a model as ``ollama:<model name>``.

Overview
--------

The module exposes two high-level helper functions, a planning agent and a settings class:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Symbol
     - Purpose
   * - :func:`~pyRadPlan.ai.agents.generate_beam_angles`
     - Ask the LLM to suggest gantry (and couch) angles for a given treatment site and
       radiation mode, then write them back into the :class:`~pyRadPlan.plan.Plan`.
   * - :func:`~pyRadPlan.ai.agents.generate_voi_objectives`
     - Ask the LLM to propose optimization objectives for each VOI in a
       :class:`~pyRadPlan.cst.StructureSet` based on the treatment site and prescribed dose.
   * - :func:`~pyRadPlan.ai.agents.run_planning_agent`
     - Let the LLM work through a whole plan with pyRadPlan's tools: it sets beams and
       objectives, optimizes and checks the dose against clinical goals
       (see :ref:`concept_planning_agent`).
   * - :class:`~pyRadPlan.ai.agents.AiSettings`
     - The unified AI settings class (the ``ai`` sub-configuration of
       ``pyRadPlan.settings``); its :attr:`agents_model` field selects the default LLM,
       read from the environment variable ``PYRADPLAN_AI_AGENTS_MODEL`` or a ``.env``
       file.

Model selection
---------------

The default model is ``claude-sonnet-4-5``.  You can override it globally via the
environment variable ``PYRADPLAN_AI_AGENTS_MODEL`` (before importing pyRadPlan) or the
settings singleton, or per-call via the ``model=`` keyword:

.. code-block:: python

    import pyRadPlan
    from pyRadPlan.ai import agents as ai_agents

    pyRadPlan.settings.ai.agents_model = "openai:gpt-4o-mini"

    # Check effective settings
    print(pyRadPlan.settings.ai.agents_model)   # openai:gpt-4o-mini

    # Override for a single call
    pln = ai_agents.generate_beam_angles(pln, "prostate", model="gemini-2.0-flash")

pydantic-ai uses the provider prefix (``openai:``, ``anthropic:``, ``google-gla:``, …)
to select the backend automatically.  When no prefix is given the model string is passed
to the Anthropic backend.

Generating beam angles
----------------------

:func:`~pyRadPlan.ai.agents.generate_beam_angles` populates ``pln.prop_stf`` with
``gantry_angles`` and matching ``couch_angles`` (zeros unless the model suggests
non-coplanar beams):

.. code-block:: python

    from pyRadPlan import IonPlan
    from pyRadPlan.ai import agents as ai_agents

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

:func:`~pyRadPlan.ai.agents.generate_voi_objectives` iterates over the VOIs in a
:class:`~pyRadPlan.cst.StructureSet` and attaches LLM-suggested
:ref:`optimization objectives <concept_optimization>` to each structure.  By default any
previously assigned objectives are cleared first (``clear_existing=True``):

.. code-block:: python

    from pyRadPlan import load_tg119, IonPlan
    from pyRadPlan.ai import agents as ai_agents

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
        IonPlan, load_tg119,
        generate_stf, calc_dose_influence, fluence_optimization,
        plot_slice, DVHCollection,
    )
    from pyRadPlan.ai import agents as ai_agents

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
    result = fluence_optimization(ct, cst, stf, dij, pln)

    dvhs = DVHCollection.from_structure_set(cst, result.physical_dose)
    plot_slice(ct=ct, cst=cst, overlay=result.physical_dose)

.. _concept_planning_agent:

Planning agent
--------------

The two helpers above ask the model one question each. The *planning agent* instead works
through a whole plan: the model chooses a pyRadPlan *tool*, pyRadPlan runs it and returns a
short result, and the model decides what to do next. This repeats until the model writes a
final report or a budget is used up. With its tools the agent

1. maps the structure names of the goals to the structures of the patient (names differ
   between clinics and languages) and corrects structure types,
2. creates helper structures for the optimizer, e.g. a ring around the target or a margin
   around an organ,
3. sets the beam angles and the optimization objectives,
4. optimizes the fluence, and
5. evaluates the dose against the clinical goals, then changes objectives, helpers or beams
   and optimizes again.

.. warning::

   The planning agent is a research feature and **not for clinical use**. Its results depend
   on the language model: two runs of the same case can give different plans, and the final
   report is written by the model and can be wrong. Judge a plan by its goal evaluation and
   its dose, not by the report.

A few terms are used throughout this section:

Goal
   One clinical requirement on the dose of a structure, e.g. "PTV D95 of at least 57 Gy" or
   "spinal cord maximum of at most 45 Gy" (:class:`~pyRadPlan.ai.agents.Goal`). A *hard*
   goal must be met; a *soft* goal should be met as well as possible. Goals are what a plan
   is judged by. They are not the optimization objectives: the agent chooses and tunes the
   objectives to reach the goals.

Protocol
   The list of goals of one run (:class:`~pyRadPlan.ai.agents.Protocol`).

Constraint table
   A JSON file with goals per treatment site, stored independently of a fractionation and
   converted to the fractionation of the plan when it is used.

Plan record
   One plan the agent produced, kept with its dose, fluence, objectives and goal evaluation
   (:class:`~pyRadPlan.ai.agents.PlanRecord`).

Review pause
   A point at which the run stops and waits for you before the next optimization or beam
   change, so you can look at the plans and give feedback.

What it needs
~~~~~~~~~~~~~

The agent needs a language model that supports tool calls. Use either

* a model of a cloud provider with its API key, as described at the top of this page, or
* a local `Ollama <https://ollama.com/>`_ server: set ``OLLAMA_BASE_URL`` to its
  OpenAI-compatible address and pass the model as ``"ollama:<model name>"``.

.. code-block:: bash

    export OLLAMA_BASE_URL=http://localhost:11434/v1

.. code-block:: python

    result = run_planning_agent(state, task, model="ollama:gpt-oss:20b")

Without the ``model`` argument the agent uses ``pyRadPlan.settings.ai.agents_model``, like
the other agents. Small local models often fail to call the tools correctly; expect better
plans from the larger models of the cloud providers.

.. _concept_planning_agent_data:

**What is sent to the model provider.** The dose calculation and the optimization run on
your computer. The model receives text only: your task, the goals, the structure names, and
the results of the tools. These results are summaries in numbers: structure volumes,
positions, extents and overlaps, beam angles, objectives, and dose metrics such as mean
doses and goal margins. The CT, the contours and the dose distributions are not sent, and
neither is a patient name or patient ID. Structure names and the text you write are sent as
they are, so do not put identifying information into them. With a local Ollama server
nothing is sent to a cloud provider.

A minimal run
~~~~~~~~~~~~~

A run needs a :class:`~pyRadPlan.ai.agents.PlanningState`, which holds the patient, the plan
and the protocol, and a task in plain language:

.. code-block:: python

    from pyRadPlan import PhotonPlan, load_tg119
    from pyRadPlan.ai.agents import Goal, PlanningState, Protocol, run_planning_agent

    ct, cst = load_tg119()
    pln = PhotonPlan()  # 30 fractions, doses per fraction (the default dose convention)

    protocol = Protocol(
        name="TG119 C-shape",
        goals=[
            Goal(voi="PTV", metric="D95", op=">=", value=1.9),
            Goal(voi="PTV", metric="D5", op="<=", value=2.2),
            Goal(voi="Core", metric="max", op="<=", value=1.0),
            Goal(voi="Core", metric="D5", op="<=", value=0.8, hard=False),
        ],
    )

    state = PlanningState(ct=ct, cst=cst, pln=pln, protocol=protocol)
    result = run_planning_agent(
        state, "Plan this C-shaped target with photon IMRT, at most 9 coplanar beams."
    )

    print(result.output.summary)           # the model's report
    print(state.last_evaluation["goals"])  # the goals as evaluated by pyRadPlan

``result`` is the result object of pydantic-ai. ``result.output`` is a
:class:`~pyRadPlan.ai.agents.PlanningReport`. Its fields ``all_hard_goals_met``,
``summary``, ``trade_offs`` and ``suggestions_for_planner`` are written by the model;
``goal_aliases`` is filled in by pyRadPlan and states which structure the goals of each
protocol name were evaluated on. ``result.usage`` holds the number of requests and tokens. Everything the agent worked on stays in ``state``:

* ``state.plans``: every plan as a plan record (see :ref:`below <concept_planning_agent_plans>`),
* ``state.cst``: the structures with the helper structures and the current objectives,
* ``state.pln``: the plan with the beam angles the agent set. The state works on its own
  copy of the plan you passed,
* ``state.last_evaluation``: the goal evaluation of the current dose,
* ``state.history``: one entry per step the agent took.

The task can state anything a planner would be told: the technique, the number of beams,
which trade-off matters most, or how many alternative plans to produce. It cannot lift the
budgets or the hard goals.

Goals
~~~~~

A :class:`~pyRadPlan.ai.agents.Goal` has these fields:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Field
     - Meaning
   * - ``voi``
     - Name of the structure. It does not have to match a structure of the patient: the
       agent maps ``SpinalCord`` to ``SPINAL_CORD`` or ``Myelon``.
   * - ``metric``
     - The dose metric, see the next table.
   * - ``op``
     - ``">="`` (the metric must reach at least the value) or ``"<="`` (at most).
   * - ``value``
     - The limit: a dose in Gy for ``mean``, ``max``, ``min`` and ``D<x>``, a volume in
       percent of the structure for ``V<x>Gy``.
   * - ``hard``
     - ``True`` (default) for a hard goal, ``False`` for a soft goal.
   * - ``source``
     - Optional text stating where the goal comes from. Goals from a constraint table
       carry the table, site and structure here.

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Metric
     - Meaning
   * - ``mean``, ``max``, ``min``
     - Mean, maximum and minimum dose of the structure.
   * - ``D95``, ``D2``, ``D0.5``
     - ``D<x>``: the dose that at least x percent of the volume receive. ``D95`` is used for
       target coverage, a small x such as ``D2`` for a near-maximum dose.
   * - ``V20Gy``, ``V0.5Gy``
     - ``V<d>Gy``: the percentage of the volume that receives at least the dose d in Gy.

Other spellings (``Dmean``, ``D95%``, ``V20``) are rejected when the goal is created, and so
are values that cannot occur: a volume above 100 % (``D120``, or a ``V<d>Gy`` goal with a
value above 100) and negative values.

**Units and dose convention.** All doses of a goal, including the dose in ``V<d>Gy``, are
read in the :ref:`dose convention <concept_dose_convention>` of the plan. With the default
``pln.dose_convention = "per_fraction"`` a value of ``1.9`` is 1.9 Gy per fraction; with
``"total"`` the goals are total doses of the course, e.g. ``66.5``. Set the convention on the
plan before the run and write the goals accordingly. The agent does not convert goals of a
protocol you give it.

**Who sets the goals.** A protocol you pass is *locked* by default
(``Protocol(locked=True)``): the agent plans for these goals and cannot replace them. With an
empty, unlocked protocol (``Protocol(locked=False)``) the agent picks the goals itself from a
constraint table, as described below.

In addition to your goals, every evaluation contains one soft goal on ``NormalTissue``, the
patient volume more than ``state.normal_tissue_margin_mm`` (default 5 mm) outside the
targets: its maximum dose should stay below the highest target dose level. It keeps the agent
from placing hot spots where no goal looks.

Goal files
~~~~~~~~~~

Goals can be kept in a JSON file and shared between plans.
:func:`~pyRadPlan.ai.agents.save_goals` writes such a file,
:func:`~pyRadPlan.ai.agents.load_goals` reads it:

.. code-block:: json

    {
      "format": "pyRadPlan-goals",
      "version": 1,
      "name": "HN SIB 70/63 Gy",
      "dose": "total",
      "fractions": 35,
      "goals": [
        {"voi": "PTV70", "metric": "D95", "op": ">=", "value": 66.5},
        {"voi": "SpinalCord", "metric": "max", "op": "<=", "value": 45.0},
        {"voi": "Parotid_L", "metric": "mean", "op": "<=", "value": 26.0, "hard": false}
      ]
    }

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Key
     - Meaning
   * - ``format``
     - Always ``"pyRadPlan-goals"``. Marks the file as a goal file.
   * - ``version``
     - Version of the file format, currently ``1``.
   * - ``name``
     - Optional name of the goal set.
   * - ``dose``
     - ``"total"`` or ``"per_fraction"``: how the doses in the file are to be read. Required.
   * - ``fractions``
     - Optional number of fractions the goals were written for.
   * - ``goals``
     - The goals, each with the fields of :class:`~pyRadPlan.ai.agents.Goal`. ``hard`` and
       ``source`` may be left out.

A file with keys other than these (e.g. a misspelled ``"hard"``), or with a goal that is
rejected for the reasons above, is not loaded; the error names the problem.

When loading, state the dose convention and the number of fractions of your plan. The doses
are converted to that convention:

.. code-block:: python

    from pyRadPlan.ai.agents import Protocol, load_goals, save_goals

    goals, notes = load_goals(
        "hn_goals.json", dose=pln.dose_convention, fractions=pln.num_of_fractions
    )
    for note in notes:
        print(note)
    protocol = Protocol(name="HN SIB", goals=goals)

    save_goals("hn_goals.json", protocol.goals, dose="total", fractions=35, name="HN SIB")

Doses that are already in the plan's convention are used as written. Otherwise total and
per-fraction doses are converted with the number of fractions stated in the file, because
that is the fractionation the doses were written for: 1.9 Gy per fraction for 30 fractions
are 57 Gy in total, whatever the number of fractions of the plan. If the file states no
``fractions``, the plan's number is used.

This is plain arithmetic and not a radiobiological conversion: goals written for 35
fractions are not adapted to the different biological effect of a plan with 20 fractions.
``load_goals`` reports such a mismatch in the returned ``notes``. Use a constraint table
when goals have to follow the fractionation.

Constraint tables
~~~~~~~~~~~~~~~~~

A constraint table stores goals per treatment site in a form that does not depend on one
fractionation. pyRadPlan converts the doses to the fractionation of the plan with the
linear-quadratic model, so the numbers the agent works with come from the table and a fixed
formula, never from the model's memory.

pyRadPlan bundles one table, ``demo``, with the sites ``prostate``, ``head_neck`` and
``thorax``. Its values are QUANTEC/RTOG-style: simplified, for demonstration only, not
validated and not for clinical use. For your own work write a table with the values of your
clinic and load it from its file.

.. code-block:: python

    from pyRadPlan.ai.agents import Protocol, load_constraint_table, site_goals

    table = load_constraint_table("demo")  # or the path of your own JSON file
    goals, notes = site_goals(
        table,
        "prostate",
        num_fractions=20,
        prescription_total_gy=60.0,
        per_fraction=True,  # doses per fraction, as in the plan's dose convention
    )
    protocol = Protocol(name="demo/prostate", goals=goals)

:func:`~pyRadPlan.ai.agents.site_goals` returns the converted goals and notes for the
planner, e.g. the table's disclaimer or a warning about the conversion.
:func:`~pyRadPlan.ai.agents.available_constraint_tables` lists the bundled tables.

A table is a JSON file of this form (a shortened excerpt of the ``demo`` table):

.. code-block:: json

    {
      "id": "demo",
      "name": "Demo constraints (QUANTEC/RTOG-style, not validated)",
      "version": "1",
      "disclaimer": "DEMONSTRATION VALUES - NOT FOR CLINICAL USE.",
      "source": "QUANTEC/RTOG-style values, simplified for demonstration; not validated.",
      "dose_reference": {"type": "EQD2", "note": "Organ doses are EQD2."},
      "sites": {
        "prostate": {
          "description": "Pelvis, prostate",
          "targets": {
            "PTV": {
              "constraints": [
                {"metric": "D95", "op": ">=", "value": 95, "unit": "%Rx"},
                {"metric": "max", "op": "<=", "value": 107, "unit": "%Rx"}
              ]
            }
          },
          "structures": {
            "Rectum": {
              "alpha_beta": 3,
              "constraints": [
                {"metric": "V50Gy", "op": "<=", "value": 50},
                {"metric": "V70Gy", "op": "<=", "value": 20}
              ]
            },
            "Femur_Head_L": {
              "alpha_beta": 3,
              "constraints": [
                {"metric": "V50Gy", "op": "<=", "value": 5, "hard": false}
              ]
            }
          }
        }
      }
    }

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Meaning
   * - ``id``, ``name``, ``version``
     - Identify the table. Goals taken from it name the id and version in their ``source``.
   * - ``disclaimer``, ``source``
     - Free text. The disclaimer is passed on to the planner and to the agent with the
       goals.
   * - ``dose_reference.type``
     - ``"EQD2"``: the doses of the table are equivalent doses in 2 Gy fractions.
       ``"fractionation"``: they are physical total doses for the ``prescription`` of the
       site.
   * - ``sites.<site>``
     - One treatment site with a ``description``, an optional ``prescription``
       (``total_dose_gy``, ``num_fractions``; required for the ``"fractionation"``
       reference), its ``targets`` and its other ``structures``.
   * - ``targets`` / ``structures``
     - Map a structure name to its ``alpha_beta`` in Gy and its ``constraints``. Without
       ``alpha_beta``, 10 Gy is used for targets and 3 Gy for other structures.
   * - ``constraints``
     - Each has ``metric``, ``op``, ``value`` and ``hard`` as in a goal, and optionally a
       ``unit``, an ``endpoint`` and a ``note``. Doses are total doses in Gy, volumes in
       percent. With ``"unit": "%Rx"`` the value of a dose metric is a percentage of the
       plan's prescribed dose.

As in a goal file, unknown keys and impossible values are rejected when the table is loaded.

**Conversion to the plan's fractionation.** A total dose :math:`D` given in :math:`n`
fractions of :math:`d = D/n` has the equivalent dose in 2 Gy fractions

.. math::

   \mathrm{EQD2} = D \, \frac{d + \alpha/\beta}{2\,\mathrm{Gy} + \alpha/\beta} .

For a table in EQD2, pyRadPlan solves this equation for the total dose in the plan's number of
fractions, using the :math:`\alpha/\beta` of the structure. A table for a fixed fractionation
is first converted to EQD2 and then to the plan. The result is written per fraction or as
total dose, depending on ``per_fraction``. Three cases are worth knowing:

* For ``V<d>Gy`` the dose threshold is converted and the metric renamed (``V50Gy`` in EQD2
  becomes ``V2.341Gy`` per fraction for 20 fractions and an :math:`lpha/eta` of 3 Gy), while the volume stays.
* ``%Rx`` values refer to the prescription of the plan and are not converted.
* The conversion of a mean dose is approximate, and above 6 Gy per fraction the
  linear-quadratic model is unreliable. Both are stated in the notes.

The functions :func:`~pyRadPlan.ai.agents.eqd2`,
:func:`~pyRadPlan.ai.agents.physical_from_eqd2` and
:func:`~pyRadPlan.ai.agents.convert_fractionation` do the same for single doses.

**Letting the agent choose.** With an empty, unlocked protocol the agent looks up the tables
itself, picks the site that fits your task, maps the table's structure names to the patient's
structures and applies the goals. Goals on structures the patient does not have are dropped.
The chosen goals are in ``state.protocol`` afterwards.

.. code-block:: python

    pln = PhotonPlan(num_of_fractions=20, prescribed_dose=60.0)
    state = PlanningState(ct=ct, cst=cst, pln=pln, protocol=Protocol(locked=False))
    result = run_planning_agent(
        state, "Prostate, 60 Gy in 20 fractions. Use the prostate goals of the demo table."
    )
    print([g.label for g in state.protocol.goals])

.. _concept_planning_agent_plans:

Plan records and scores
~~~~~~~~~~~~~~~~~~~~~~~

Every optimization adds a :class:`~pyRadPlan.ai.agents.PlanRecord` to ``state.plans``, also
when a later plan is better and when the run ends early. A record holds

* ``index``: the plan number, counted from 1,
* ``result``: the :class:`~pyRadPlan.result.PlanResult` with the dose on the CT grid and the
  fluence (``record.dose`` and ``record.fluence`` are shortcuts),
* ``beams``: the gantry and couch angles,
* ``cst`` and ``objectives``: the structures and objectives the plan was optimized with,
* ``evaluation``: the goal evaluation.

The evaluation is a dictionary. ``evaluation["goals"]`` lists every goal with its ``value``,
its ``margin`` (the distance to the limit in the goal's unit; zero or more means the goal
is met), the ``rel_margin`` (the margin divided by the limit) and ``passed``.
``all_hard_passed`` and ``num_failed`` summarize them.

A goal cannot be evaluated if its structure was not found among the patient's structures
or if the structure has no voxels. Such a goal is listed under ``unresolved_goals`` and does
not count as passed or failed, nor in the score. Structures that were not found are listed
under ``unresolved_structures``, empty ones under ``empty_structures``. Check these lists
after a run: an unresolved hard goal is one nobody checked.

``evaluation["score"]`` condenses the goals into one number to rank plans by. It is the
weighted mean of the relative margins in percent, where a hard goal counts three times as
much as a soft goal. Each margin is limited to the range from -100 % to +20 %, so one goal
far inside its limit cannot hide another goal that fails. A higher score is better, and the
score is zero or more when all goals are met. The score is a ranking aid, not a clinical
measure of plan quality.

.. code-block:: python

    for record in state.plans:
        ev = record.evaluation
        print(record.index, ev["score"], ev["all_hard_passed"], record.beams["gantry_angles"])

    best = max(state.plans, key=lambda p: (p.evaluation["all_hard_passed"], p.evaluation["score"]))
    dose = best.result.physical_dose.values  # sitk.Image on the CT grid

Warm starts, blends and the Pareto front
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three tools let the agent explore trade-offs without spending its whole budget on
optimizations from scratch. Each is also a method of
:class:`~pyRadPlan.ai.agents.PlanningState`, so you can call it yourself after a run.

**Warm starts.** ``state.optimize(start_from=3)`` starts the optimizer from the fluence of
plan 3 instead of from zero (see :ref:`concept_initial_fluence`). After a small change of
the objectives the new plan then differs from plan 3 only where the change asks for it, and
it needs fewer iterations. The plan started from must have the same beams.

**Blends.** ``state.blend_plans(plans=[2, 5], weights=[0.6, 0.4])`` mixes the fluences of
earlier plans with the same beams and computes the dose of the mix. This costs no
optimization, and the blend is a complete plan with its own record. Physical dose, and
RBE-weighted dose with a constant RBE, is linear in the fluence. Maximum, minimum and mean
goals that all source plans meet are then also met by the blend, while ``D<x>`` and
``V<d>Gy`` goals can change either way. Every goal is evaluated again on the dose of the
blend. At most ``state.max_blends`` (default 20) blends are kept.

**Pareto sampling.** ``state.pareto_front(plans=[2, 5, 6])`` evaluates several hundred
random blends of the given plans and returns their *Pareto front*: the blends that meet all
hard goals and where no goal can improve without another getting worse. The result names the
trade-off axes, a few spread-out blends on the front with their weights, a compromise blend,
and the plans that add nothing to the front. No plan is changed; pass the weights of a blend
to ``blend_plans`` to turn it into a plan. The sampling relies on a dose that is linear in
the fluence, so it is refused for plans with a variable RBE (e.g. carbon ions); blending
works there too.

Selecting distinct plans
~~~~~~~~~~~~~~~~~~~~~~~~

A run usually leaves several good plans. :func:`~pyRadPlan.ai.agents.select_distinct_plans`
picks up to ``n`` of them that are good in different ways. It ranks the plans by whether all
hard goals pass and then by score, and takes the best one first. Each further plan must be
better than every plan picked so far on at least one goal, by at least ``min_gain`` (0.05,
i.e. 5 % of the goal's limit, by default). If no such alternative exists, fewer than ``n``
plans are returned.

.. code-block:: python

    from pyRadPlan.ai.agents import select_distinct_plans

    for record in select_distinct_plans(state.plans, n=2):
        print(record.index, record.evaluation["score"])

Budgets and limits
~~~~~~~~~~~~~~~~~~

A run is bounded on two sides: the computation on your machine and the requests to the
model, which cost money with a cloud provider.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Limit
     - Meaning
   * - ``max_optimizations``
     - Number of optimizations (default 8). Set it on the state
       (``state.max_optimizations``) or pass it to
       :func:`~pyRadPlan.ai.agents.run_planning_agent`. Blends and Pareto sampling do not
       count.
   * - ``state.max_dij_calcs``
     - Number of dose influence calculations (default 3). Every new beam setup needs one,
       so this limits how often the agent can change the beams.
   * - ``state.max_iter``
     - Most optimizer iterations of one optimization (default 5000).
   * - ``state.min_beams``, ``state.max_beams``, ``state.allow_couch``
     - Allowed number of beams (1 to 30) and whether couch angles other than 0 may be used.
   * - ``request_limit``
     - Number of requests to the model in one call of ``run_planning_agent`` or
       ``continue_planning_agent`` (default 40). One request is one answer of the model,
       which usually holds one tool call.
   * - ``tool_calls_limit``
     - Number of tool calls in one call. By default it is derived from the optimization
       budget (40 plus 20 per optimization); it stops a model that keeps calling tools
       without making progress.

When the optimization budget is used up, further optimizations are refused and the agent is
told to write its report. When the request limit or the tool-call limit is reached, the run
stops with a :class:`~pyRadPlan.ai.agents.PlanningRunError`, see
:ref:`concept_planning_agent_continue`.

The argument ``tools`` restricts the agent to some of its tools, e.g. to keep it from
changing the beams. :func:`~pyRadPlan.ai.agents.planning_tools` returns the name and the
description of every tool.

.. _concept_planning_agent_continue:

Reviewing and continuing a run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Review pauses.** With ``state.review_every = 3`` the run pauses for you before an
optimization

* once three optimizations have run since the last review,
* when a plan meets all hard goals for the first time, and
* when the optimization budget is used up.

A paused run returns normally, but ``result.output`` is not a
:class:`~pyRadPlan.ai.agents.PlanningReport`. Look at ``state.plans``, then resume with
:func:`~pyRadPlan.ai.agents.continue_planning_agent`:

.. code-block:: python

    from pyRadPlan.ai.agents import PlanningReport, continue_planning_agent

    state.review_every = 3
    result = run_planning_agent(state, task)

    while not isinstance(result.output, PlanningReport):
        print("Score of the last plan:", state.plans[-1].evaluation["score"])
        result = continue_planning_agent(
            state, result, prompt="Good coverage. Spare the parotids more.", action="continue"
        )

``action`` decides what happens to the step the agent was about to take:

* ``"continue"`` runs it. A ``prompt`` is passed to the agent as your feedback together with
  the result.
* ``"redirect"`` skips it and gives the agent your ``prompt`` instead.
* ``"finish"`` skips it and asks the agent for its final report.

Before continuing you can raise a budget (e.g. ``state.max_optimizations``) or change the
goals with the argument ``protocol``: every plan is then evaluated again against the new
goals, and the agent is told what changed.

**After the report.** A finished run can be continued in the same way with a new task as
``prompt``, e.g. ``"Now try a plan with 5 beams."`` The agent keeps its conversation and all
plans.

**After an interruption.** A run that reaches the request or tool-call limit, or that fails
because of the model or the provider (e.g. a network error), raises a
:class:`~pyRadPlan.ai.agents.PlanningRunError`. Nothing is lost: the state keeps all plans,
and the error carries the conversation so far (``err.messages``) and the usage
(``err.usage``). Pass the error to ``continue_planning_agent`` to go on where the run
stopped. Each call has its own request limit.

.. code-block:: python

    from pyRadPlan.ai.agents import PlanningRunError

    try:
        result = run_planning_agent(state, task, request_limit=40)
    except PlanningRunError as err:
        print(err)  # names the cause
        result = continue_planning_agent(state, err, request_limit=40)

``err.limit_exceeded`` tells whether a limit was the cause. For other causes, check the
message before continuing, since the same error may occur again.

History compaction
~~~~~~~~~~~~~~~~~~

Every request sends the whole conversation to the model, so long runs become slow and
expensive. With ``compact_after=3`` the conversation is shortened after every third
optimization: older tool results are replaced by a short note, older optimization results
keep their score and goal values, and results about the anatomy stay. The agent's own
messages and tool calls are kept, so it still knows what it tried. The default ``0`` never
compacts. Providers that cache the conversation (e.g. Anthropic) have to rebuild their cache
after each compaction, so very frequent compaction can cost more than it saves.

Reproducibility
~~~~~~~~~~~~~~~

The optimizations of the agent run with the SciPy solver and NumPy on the CPU, whatever
compute backend and solver are configured otherwise. The same beams, objectives and starting
point therefore give the same plan, and the agent can tell the effect of its changes from
numerical noise. ``PlanningState(solver=None, cpu_only=False)`` uses the solver of the plan
and the configured backend instead. The model itself is not deterministic: a repeated run
takes different decisions.

Limitations
~~~~~~~~~~~

* The agent is a research feature and not for clinical use. A plan it produces has to be
  checked like any other plan.
* The results depend on the model, its version and the wording of the task. A weaker model
  may map structures wrongly, stop early or spend its budget on small variations.
* Goals are checked on the dose pyRadPlan computed for the nominal scenario. Robustness and
  deliverability (e.g. after sequencing) are not checked.
* Converting constraints between fractionations uses the linear-quadratic model with the
  :math:`\alpha/\beta` values of the table, which is an approximation.
* The agent offers the objectives registered in pyRadPlan. Objectives that are set up in
  code, such as the :ref:`outcome objectives <concept_outcome_objectives>`, and objectives
  that need a reference image are not offered.

The tutorials ``utils_ai_planning_agent`` (TG119 phantom) and ``utils_ai_planning_agent_hn``
(head-and-neck phantom with a simultaneous integrated boost) show complete runs
(:ref:`Tutorials`).

.. _concept_agentic_ui_mode:

Agentic UI mode of the GUI
~~~~~~~~~~~~~~~~~~~~~~~~~~

The desktop GUI runs the planning agent in its *agentic* UI mode. Select it under
**File > UI Mode**, or set ``pyRadPlan.settings.gui.ui_mode = "agentic"`` (environment
variable ``PYRADPLAN_GUI_UI_MODE``) before starting the GUI. The left side of the main window
then shows an **Agent** tab:

* **Plan** holds the usual plan editor (radiation mode, fractionation, dose convention).
* **Agentic Plan** holds the settings of the run: the model, the task, the budgets described
  above and how many plans to select at the end. Its **Goals** table lists one goal per row
  (structure, metric, operator, value, hard). Goals can be added and removed by hand, taken
  from a site of a constraint table (**From table**), or read from and written to a goal file
  (**Import…**, **Export…**). The **Tools** list switches single tools of the agent off.
* **Progress** plots the margin of every goal over the optimizations, together with the
  score. A goal above the zero line is met.
* **Agentic Result** shows the agent's report and a table of all optimizations with their
  score and the number of missed hard and soft goals.

**Run agent** starts the run. The **Agent log** below lists every tool call of the agent with
its arguments as it happens. **Pause** makes the agent stop before its next optimization or
beam change; a run also pauses by itself if *Review every* is set. While it is paused you can
type feedback and choose **Continue**, **Redirect** or **Finish**, which correspond to the
actions of :func:`~pyRadPlan.ai.agents.continue_planning_agent`. You can also change the
goals and budgets before continuing.

At the end of a run, the plans picked by
:func:`~pyRadPlan.ai.agents.select_distinct_plans` are opened in their own tabs next to the
Agent tab. A plan tab shows the usual workflow, so the plan can be inspected, changed and
optimized further by hand. **Load plan** in the result table opens any other optimization in
the same way.

The agentic mode is available in the desktop GUI only, not in the web version.

API reference
-------------

The full API for the ``ai.agents`` module, including the settings class and helper
functions, is documented in the :doc:`API reference </api/ai_agents>`:

* :class:`~pyRadPlan.ai.agents.AiSettings`
* :func:`~pyRadPlan.ai.agents.generate_beam_angles`
* :func:`~pyRadPlan.ai.agents.generate_voi_objectives`
* :func:`~pyRadPlan.ai.agents.run_planning_agent`,
  :func:`~pyRadPlan.ai.agents.continue_planning_agent` and
  :class:`~pyRadPlan.ai.agents.PlanningState`
