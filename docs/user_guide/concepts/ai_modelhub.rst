.. _concept_ai_modelhub:

AI Model Hub
============

pyRadPlan ships an optional ``ai.modelhub`` module that loads trained AI models -- and the
preprocessor that prepares their inputs -- either from the
`HuggingFace Hub <https://huggingface.co>`_ or from a local directory, with version dedup
and offline support.  A model repository is self-contained: it carries its network
definition, its preprocessor and a declarative ``model_config.json`` that drives
instantiation, so pyRadPlan needs no per-model code.

.. seealso::

   The module is the model source of the ``"AIBeamlet"`` dose engine; how a dose-calculation
   model is selected, checked and used is described under
   :ref:`concept_ai_dose_calculation`.

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

    from pyRadPlan.ai.modelhub import list_local_models, load_model, trust_model

    # From a local directory you placed yourself (works fully offline):
    model, preprocessor = load_model(local_dir="path/to/model_folder")

    # From HuggingFace by bare name (resolves to <hf_org>/<name>). The model's code is
    # downloaded but runs only once you trust it (see "Which model code runs"):
    trust_model("outcome-ORPDenseNet-tg119", revision="v1.0.0")
    model, preprocessor = load_model("outcome-ORPDenseNet-tg119", revision="v1.0.0")

    # By full id -- e.g. a fork -- used as-is:
    trust_model("my-org/my-repo")
    model, preprocessor = load_model("my-org/my-repo")

    # Same thing, explicitly:
    model, preprocessor = load_model(repo_id="my-org/my-repo")

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

.. _modelhub_trust:

Which model code runs
---------------------

.. warning::

   :func:`~pyRadPlan.ai.modelhub.load_model` (and the ``"AIBeamlet"`` dose engine)
   executes the ``model.py`` and ``preprocessor.py`` shipped inside the model
   repository -- arbitrary Python with your permissions.

Whether a model's code may run depends on where it came from, not on how you name it:

* **A model folder you placed yourself** runs as-is, whether you pass it as
  ``local_dir`` or load it by id from ``<local_models_dir>/<org>/<repo>``.  pyRadPlan
  never refreshes such a folder from the Hub, not even for an unpinned load.
* **A model pyRadPlan downloaded** -- into ``<local_models_dir>/<org>/<repo>``, into a
  ``local_dir`` or into the HuggingFace cache -- runs only once you trust it, also when
  you later pass its folder as ``local_dir``.  pyRadPlan tells the two apart by the
  ``.pyradplan_model.json`` marker it writes into a folder *before* downloading into
  it, and by the HuggingFace cache layout; it downloads only into an empty folder or
  one carrying the marker.  A folder holding neither a complete model nor the marker
  is refused rather than downloaded over.

Downloaded code is trusted in one of two ways:

* :func:`~pyRadPlan.ai.modelhub.trust_model` downloads the model if needed, without
  running anything, and records a SHA-256 hash of its code (every file but the
  ``*.safetensors`` weights, which hold tensors only).  Inspect the code in the
  returned directory first:

  .. code-block:: python

      from pyRadPlan.ai.modelhub import trust_model

      model_dir, code_hash = trust_model("DKFZ-RadOpt/dosecalc-x", revision="v1.0.0")

* The setting ``pyRadPlan.settings.ai.modelhub_trust_remote_code = True`` (environment
  variable ``PYRADPLAN_AI_MODELHUB_TRUST_REMOTE_CODE=1``), or
  ``load_model(..., trust_remote_code=True)``, trusts whatever is downloaded.  The code
  loaded this way is recorded too, so it keeps running after the setting is switched
  off.

The records live in ``<data_dir>/ai_trusted_models.json`` (``~/.pyradplan`` by default),
keyed by repository id -- never inside the model folder, so nothing a repository ships
can make it trusted.  :func:`~pyRadPlan.ai.modelhub.is_trusted` tells whether a model on
disk runs without the setting, and :func:`~pyRadPlan.ai.modelhub.untrust_model` forgets a
model's records.

**When the code changes upstream**, e.g. an unpinned load pulls an update, its hash no
longer matches and loading is refused until you inspect the new code and call
:func:`~pyRadPlan.ai.modelhub.trust_model` again (or enable the setting).  Pin a
``revision`` so the code you trusted is the code you keep running.

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
directory or the per-user ``~/.pyradplan/.env``); the model-hub-related fields are:

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
     - Trust the code of every downloaded model (see :ref:`modelhub_trust`)
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

Pin a ``revision`` (tag, branch or commit) for reproducible loads: a downloaded copy
recorded as that revision is used directly, without touching the network.  Without a
pinned revision the Hub is asked whether the downloaded copy is still current, and a
stale one is refreshed -- a refresh that changes the model's code needs your trust
again (see :ref:`modelhub_trust`); if the Hub cannot be reached the local copy is used
anyway, with a warning.  A model folder you placed yourself is never refreshed.

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

Using a model for dose calculation
----------------------------------

Dose-calculation models (``metadata.task = "dose_calc"``) are consumed by the ``"AIBeamlet"``
engine rather than loaded by hand; it reads the same repository contract and additionally
honours the ``metadata.dose_calc`` section (sampling grid, predicted quantities) and
``metadata.training.modality``:

.. code-block:: python

    pln.prop_dose_calc = {"engine": "AIBeamlet", "model": "<org>/<model>"}
    dij = calc_dose_influence(ct, cst, stf, pln)

See :ref:`concept_ai_dose_calculation` for model selection, availability and device handling,
and the :doc:`AI dose calculation tutorial </tutorials/examples/ai_dosecalc>` for a runnable
example. AI dose calculation is a research feature and not intended for clinical use.

API reference
-------------

The full API for the ``ai.modelhub`` module is documented in the
:doc:`API reference </api/ai_modelhub>`:

* :func:`~pyRadPlan.ai.modelhub.load_model`
* :class:`~pyRadPlan.ai.modelhub.BasePreprocessor`
* :func:`~pyRadPlan.ai.modelhub.list_local_models`
* :class:`~pyRadPlan.ai.modelhub.ModelTask`
* :class:`~pyRadPlan.ai.modelhub.AiSettings`
