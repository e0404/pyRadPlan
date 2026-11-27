optimization.objectives
=======================

.. currentmodule:: pyRadPlan.optimization.objectives

.. autosummary::
   :toctree: generated/optimization/objectives/

    DoseUniformity
    EUD
    MaxDVH
    MeanDose
    MinDVH
    SquaredDeviation
    SquaredMimicking
    SquaredOverdosing
    SquaredUnderdosing
    Variance


POLO Objectives
---------------

Objectives on the probability of lesion origin (POLO) model for proton plans, see
:class:`~pyRadPlan.quantities.POLO`. They are a research feature.

.. autosummary::
   :toctree: generated/optimization/objectives/

    NTCPPOLO
    HellingerPOLO
    LogSumExpConvPOLO
    HellingerConvPOLO


Outcome Objectives
------------------

Objectives on a probability predicted by an outcome model (e.g. NTCP), see
:ref:`concept_outcome_objectives`. They are a research feature and created in code only.

.. autosummary::
   :toctree: generated/optimization/objectives/

    OutcomeModel
    OutcomeLR
    OutcomeSklearn
    OutcomeObjective
    FeatureOutcomeObjective


Base Classes
------------

.. autosummary::
   :toctree: generated/optimization/objectives/

    Objective


Other Functions
---------------

.. autosummary::
   :toctree: generated/optimization/objectives/

    get_available_objectives
    get_objective
    register_objective
