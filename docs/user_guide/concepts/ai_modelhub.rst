.. _concept_ai_modelhub:

AI Model Hub
============

pyRadPlan ships an optional ``ai.modelhub`` module that loads trained AI models -- and the
preprocessor that prepares their inputs -- either from the
`HuggingFace Hub <https://huggingface.co>`_ or from a local directory, with version dedup
and offline support.  A model repository is self-contained: it carries its network
definition, its preprocessor and a declarative ``model_config.json`` that drives
instantiation, so pyRadPlan needs no per-model code.

.. note::

   The Hub client (``huggingface_hub``, ``safetensors``) is installed with pyRadPlan;
   only a ``torch`` build matching your platform must be installed separately, e.g.:

   .. code-block:: bash

       pip install torch

   See `pytorch.org <https://pytorch.org/get-started/locally/>`_ for the right ``torch``
   command for your CUDA/CPU setup.

Quick start
-----------

.. code-block:: python

    from pyRadPlan.ai.modelhub import load_model, list_local_models

    # From a local directory (works fully offline):
    model, preprocessor = load_model(local_dir="path/to/model_folder")

    # By bare name on HuggingFace (resolves to <hf_org>/<name>):
    model, preprocessor = load_model(
        "outcome-ORPDenseNet-tg119", revision="v1.0.0", trust_remote_code=True
    )

    # By full id -- e.g. a fork -- used as-is:
    model, preprocessor = load_model("my-org/my-repo", trust_remote_code=True)

    # Same thing, explicitly:
    model, preprocessor = load_model(repo_id="my-org/my-repo", trust_remote_code=True)

    print(list_local_models())  # models available on disk (no network)

:func:`~pyRadPlan.ai.modelhub.load_model` returns the model with its weights loaded, moved
to the configured device and set to eval mode, together with the
:class:`~pyRadPlan.ai.modelhub.BasePreprocessor` instance configured for it.  The
preprocessor brackets the model call:

.. code-block:: python

    inputs = preprocessor.preprocess(raw_inputs)   # or: preprocessor(raw_inputs)
    outputs = model(inputs)
    result = preprocessor.postprocess(outputs)

What ``raw_inputs`` and ``result`` are is defined by the model repository, not by
pyRadPlan -- record it in the model's ``metadata`` (see
:ref:`model_repository_contract`).

.. warning::

   :func:`~pyRadPlan.ai.modelhub.load_model` executes the ``model.py`` and
   ``preprocessor.py`` shipped inside the model repository.  For anything fetched from
   the Hub this is gated behind an explicit ``trust_remote_code=True`` (or
   ``PYRADPLAN_AI_MODELHUB_TRUST_REMOTE_CODE=1``), which defaults to ``False``.  Enable it
   only for sources you trust, and prefer pinning a ``revision`` so you know which code
   you are running.  A directory you pass yourself as ``local_dir`` is exempt -- that
   code is already under your control.

Discovering local models
------------------------

:func:`~pyRadPlan.ai.modelhub.list_local_models` lists the models present under the
configured ``local_models_dir`` and in the HuggingFace cache.  It performs no network
access.  Models are reported as ``<org>/<repo>`` -- which
:func:`~pyRadPlan.ai.modelhub.load_model` accepts directly, so the output feeds straight
back in, and a private fork is never confused with its upstream namesake.

A model's *task* -- what it does -- comes from ``metadata.task`` in its
``model_config.json``, falling back to the repository-name prefix (``dosecalc-*`` /
``outcome-*``), so the list can be filtered:

.. code-block:: python

    from pyRadPlan.ai.modelhub import ModelTask, list_local_models, load_model

    for name in list_local_models(ModelTask.DOSE_CALC):
        print(name)                     # e.g. "DKFZ-RadOpt/dosecalc-..."

    model, preprocessor = load_model(list_local_models(ModelTask.OUTCOME)[0])

The task (:class:`~pyRadPlan.ai.modelhub.ModelTask`) is deliberately distinct from the
*modality* in the pyRadPlan sense (the radiation type, ``radiation_mode`` on a
:class:`~pyRadPlan.plan.Plan`), which a repository records under
``metadata.training.modality``.

Configuration
-------------

The model hub is configured through :class:`~pyRadPlan.ai.modelhub.AiSettings`, the
unified ``ai`` sub-configuration of the global ``pyRadPlan.settings`` singleton (see
:doc:`/api/settings`), which also carries the AI agent settings.  It reads environment
variables prefixed with ``PYRADPLAN_AI_`` (or a ``.env`` file in the working
directory); the model-hub-related fields are:

.. list-table::
   :header-rows: 1
   :widths: 20 32 18 30

   * - Setting
     - Environment variable
     - Default
     - Meaning
   * - ``modelhub_hf_org``
     - ``PYRADPLAN_AI_MODELHUB_HF_ORG``
     - ``DKFZ-RadOpt``
     - HuggingFace org/namespace a bare model name is resolved against
   * - ``modelhub_local_models_dir``
     - ``PYRADPLAN_AI_MODELHUB_LOCAL_MODELS_DIR``
     - ``<data_dir>/ai_models``
     - Base directory for local models; the default ``local_dir`` is
       ``<local_models_dir>/<org>/<repo>``.  An empty string uses the HuggingFace cache
       only
   * - ``modelhub_cache_dir``
     - ``PYRADPLAN_AI_MODELHUB_CACHE_DIR``
     - HuggingFace default
     - HuggingFace cache directory
   * - ``modelhub_offline``
     - ``PYRADPLAN_AI_MODELHUB_OFFLINE``
     - ``False``
     - Force ``local_files_only``
   * - ``modelhub_trust_remote_code``
     - ``PYRADPLAN_AI_MODELHUB_TRUST_REMOTE_CODE``
     - ``False``
     - Allow executing the code shipped with a model fetched from the Hub
   * - ``modelhub_device``
     - ``PYRADPLAN_AI_MODELHUB_DEVICE``
     - ``cpu``
     - Device the model is moved to

The settings can also be inspected and changed at runtime:

.. code-block:: python

    import pyRadPlan

    print(pyRadPlan.settings.ai.modelhub_local_models_dir)
    pyRadPlan.settings.ai.modelhub_device = "cuda"

The data root (``<data_dir>``) defaults to ``~/.pyradplan`` and can be relocated with the
``PYRADPLAN_DATA_DIR`` environment variable (see
:func:`~pyRadPlan.core.get_data_dir`).  Downloaded models are laid out as
``<local_models_dir>/<org>/<repo>``.

Versions and updates
--------------------

Pin a ``revision`` (tag, branch or commit) for reproducible loads: a local copy recorded
as that revision is used directly, without touching the network.  Without a pinned
revision the Hub is asked whether the local copy is still current, and a stale one is
refreshed; if the Hub cannot be reached the local copy is used anyway, with a warning.

.. _model_repository_contract:

Model-repository contract
-------------------------

Each model lives in its own HuggingFace repository (or a local directory) containing
exactly these files:

.. code-block:: text

    <model folder>/
      model.py             # nn.Module class definition(s)
      preprocessor.py      # subclass of pyRadPlan.ai.modelhub.BasePreprocessor
      weights.safetensors  # state_dict saved with safetensors
      model_config.json    # declarative config (see model_config.template.json)

Instantiation is driven entirely by ``model_config.json`` -- there is no extra init file.
Any model type (dose calculation, outcome, …) uses the same handful of loader-consumed
keys; everything else is free-form metadata the loader ignores:

* ``model_name`` selects the class in ``model.py``; ``model_params`` are passed to its
  constructor as keyword arguments (so keep only real constructor kwargs there).
* The :class:`~pyRadPlan.ai.modelhub.BasePreprocessor` subclass in ``preprocessor.py`` is
  constructed with ``model_preprocessing``, which configures **both** ``preprocess()``
  and ``postprocess()``.  If ``preprocessor.py`` defines more than one subclass, set
  ``preprocessor_name`` in ``model_config.json`` to disambiguate.
* ``metadata`` is otherwise free-form and not read by the loader; use it to record how
  the model was trained and any assumptions it makes.  Only ``metadata.task`` is
  interpreted, by :func:`~pyRadPlan.ai.modelhub.list_local_models`.

.. code-block:: json

    {
        "model_name": "ExampleNet",
        "model_params": {"in_channels": 3, "hidden_channels": 4, "out_features": 1},
        "model_preprocessing": {"type_order": ["dose", "ct", "mask"]},
        "metadata": {
            "task": "outcome",
            "description": "…",
            "training": {"modality": "protons"}
        }
    }

A model folder's ``model.py`` and ``preprocessor.py`` are imported under a package name
unique to the folder, so several models can be loaded side by side and the classes they
define stay picklable.  Import sibling files relatively
(``from .preprocessor import …``).

A working reference implementation lives in ``test/data/ai_models/dummy_model/`` and is
exercised by the test suite.  The annotated config template is
``pyRadPlan/data/ai_models/model_config.template.json``.

API reference
-------------

The full API for the ``ai.modelhub`` module is documented in the
:doc:`API reference </api/ai_modelhub>`:

* :func:`~pyRadPlan.ai.modelhub.load_model`
* :class:`~pyRadPlan.ai.modelhub.BasePreprocessor`
* :func:`~pyRadPlan.ai.modelhub.list_local_models`
* :class:`~pyRadPlan.ai.modelhub.ModelTask`
* :class:`~pyRadPlan.ai.modelhub.AiSettings`
