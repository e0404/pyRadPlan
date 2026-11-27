ai.modelhub
===========

.. currentmodule:: pyRadPlan.ai.modelhub

Loading of AI models and their preprocessors from the HuggingFace Hub or a
local directory. See :ref:`concept_ai_modelhub` in the user guide for the
configuration and the model-repository contract (``model.py``,
``preprocessor.py``, ``weights.safetensors``, ``model_config.json``).

.. note::

   Loading a model executes Python shipped inside the repository. A model
   folder you placed yourself runs as-is; downloaded code runs only once
   trusted, with :func:`trust_model` or the setting
   ``pyRadPlan.settings.ai.modelhub_trust_remote_code = True``
   (``PYRADPLAN_AI_MODELHUB_TRUST_REMOTE_CODE=1``). See :ref:`modelhub_trust`.

.. autosummary::
   :toctree: generated/ai/modelhub/

   AiSettings


Loading
-------

.. autosummary::
   :toctree: generated/ai/modelhub/

   load_model
   load_model_with_dir
   BasePreprocessor
   OutcomePreprocessor


pyanno4rt outcome models
------------------------

Outcome models trained with pyanno4rt are scikit-learn estimators in a local folder, not
model repositories of the hub; see :ref:`modelhub_pyanno4rt`.

.. autosummary::
   :toctree: generated/ai/modelhub/

   load_pyanno4rt_model
   OutcomeModelSpec
   OutcomeFeature


Trust
-----

.. autosummary::
   :toctree: generated/ai/modelhub/

   trust_model
   untrust_model
   is_trusted
   TrustedModel
   UntrustedModelError


Discovery
---------

.. autosummary::
   :toctree: generated/ai/modelhub/

   list_local_models
   ModelTask
   task_from_dir
   task_from_name


Inspection
----------

.. autosummary::
   :toctree: generated/ai/modelhub/

   model_facts
   local_model_dir
   hub_config_dir
   list_hub_models


Hub resolution
--------------

.. autosummary::
   :toctree: generated/ai/modelhub/

   resolve_model_dir
   is_valid_model_dir
   repo_subpath
