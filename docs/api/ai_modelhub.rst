ai.modelhub
===========

.. currentmodule:: pyRadPlan.ai.modelhub

Loading of AI models and their preprocessors from the HuggingFace Hub or a
local directory. See :ref:`concept_ai_modelhub` in the user guide for the
configuration and the model-repository contract (``model.py``,
``preprocessor.py``, ``weights.safetensors``, ``model_config.json``).

.. note::

   Loading a model fetched from the Hub executes Python shipped inside the
   repository and therefore requires an explicit ``trust_remote_code=True``
   (or ``PYRADPLAN_AI_TRUST_REMOTE_CODE=1``). A directory passed as
   ``local_dir`` is exempt.

.. autosummary::
   :toctree: generated/ai/modelhub/

   AiSettings


Loading
-------

.. autosummary::
   :toctree: generated/ai/modelhub/

   load_model
   BasePreprocessor


Discovery
---------

.. autosummary::
   :toctree: generated/ai/modelhub/

   list_local_models
   ModelTask
   task_from_dir
   task_from_name


Hub resolution
--------------

.. autosummary::
   :toctree: generated/ai/modelhub/

   resolve_model_dir
   is_valid_model_dir
   repo_subpath
