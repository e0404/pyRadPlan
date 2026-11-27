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
* ``metadata`` is otherwise free-form: use it to record how the model was trained and any
  assumptions it makes. The loader itself ignores it. Other parts of pyRadPlan read these
  entries:

  * ``metadata.task`` -- by :func:`~pyRadPlan.ai.modelhub.list_local_models` (and the model
    suggestions of the GUI) to filter models by what they do;
  * ``metadata.training.modality`` and ``metadata.requirements`` -- by the ``"AIBeamlet"`` dose
    engine (via :func:`~pyRadPlan.ai.modelhub.model_facts`) to check the radiation mode and
    the installed packages of a model before offering it;
  * ``metadata.dose_calc`` -- by the ``"AIBeamlet"`` dose engine, see below.

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

Working reference implementations live in the test data and are exercised by the test suite:
``test/data/ai_models/dummy_model/`` is an outcome model, and
``test/data/ai_models/DKFZ-RadOpt/dummy_dosecalc_model/`` a dose-calculation model for the
``"AIBeamlet"`` engine.  The annotated config template is
``pyRadPlan/data/ai_models/model_config.template.json``.

Using a model as an optimization objective
------------------------------------------

Outcome models (``metadata.task = "outcome"``) predict a probability from the dose, e.g. an
NTCP, and are used through the
:class:`~pyRadPlan.optimization.objectives.OutcomeModel` objective, which loads the model like
:func:`~pyRadPlan.ai.modelhub.load_model` does (including the rule of :ref:`modelhub_trust`);
see :ref:`concept_outcome_objectives`. The preprocessor of such a model implements
:class:`~pyRadPlan.ai.modelhub.OutcomePreprocessor`:

``configure(ct, cst, cst_masks=None, device=None)``
    Called before the optimization with the planning CT and structure set, the VOI names given
    to the objective and the device of the model, to prepare the static inputs.
``set_dose_grid(dose_grid)``
    Called with the grid of the optimization dose.
``preprocess(dose)``
    Receives the dose as a flat float32 torch tensor on the device of the model, with one
    value per voxel of the dose grid in the order of the flattened ``(z, y, x)`` array (x
    runs fastest), and returns the tuple of model inputs. The objective differentiates
    ``postprocess(model(*preprocess(dose)))`` with respect to the dose, so both methods must
    use differentiable torch operations and ``postprocess`` must return a scalar probability.

.. _modelhub_pyanno4rt:

pyanno4rt outcome models
~~~~~~~~~~~~~~~~~~~~~~~~

Outcome models trained with `pyanno4rt <https://github.com/pyanno4rt/pyanno4rt>`_ are
scikit-learn estimators on dose and geometry features, not model repositories, and are loaded
from a local folder with :func:`~pyRadPlan.ai.modelhub.load_pyanno4rt_model`. It returns the
estimator and its features (:class:`~pyRadPlan.ai.modelhub.OutcomeModelSpec`) for the
:class:`~pyRadPlan.optimization.objectives.OutcomeLR` and
:class:`~pyRadPlan.optimization.objectives.OutcomeSklearn` objectives, which call it when given
the folder:

.. code-block:: python

    from pyRadPlan.ai.modelhub import load_pyanno4rt_model

    estimator, spec = load_pyanno4rt_model("path/to/pyanno4rt/snapshot")

* The folder holds pyanno4rt's ``input.json`` and the model in ``<label>/model.sav`` (as in the
  public pyanno4rt snapshots) or ``<label>/predictor.*``.
* The feature order is taken from ``feature_names`` in the ``configuration.json`` of the model
  folder when present, else from the estimator (``feature_names_in_``), else from the header of
  the training table.
* Models trained with a feature preprocessing other than identity are refused.
* ``scikit-learn`` is needed and not installed with pyRadPlan: ``pip install scikit-learn``.
* The model file is a pickle, and unpickling runs code from the file. The trust records of the
  hub do not apply to it, so only load models from a source you trust.

Using a model for dose calculation
----------------------------------

Dose-calculation models (``metadata.task = "dose_calc"``) are consumed by the ``"AIBeamlet"``
engine rather than loaded by hand:

.. code-block:: python

    pln.prop_dose_calc = {"engine": "AIBeamlet", "model": "<org>/<model>"}
    dij = calc_dose_influence(ct, cst, stf, pln)

The engine loads the model like :func:`~pyRadPlan.ai.modelhub.load_model` does (including the
rule of :ref:`modelhub_trust`) and additionally reads the ``metadata`` entries listed in
:ref:`concept_ai_dose_calculation`: the sampling grid and predicted quantities under
``metadata.dose_calc`` (the grid is required), the training assumptions it checks against the
plan, ``metadata.training.modality`` and ``metadata.requirements``.

.. _dose_calc_preprocessor_contract:

Preprocessor contract of a dose-calculation model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What ``raw_inputs`` and ``result`` of the generic quick start are is fixed for dose
calculation: for every ray the engine samples the CT on the model's grid and calls the
preprocessor of the model in two steps. A dose-calculation preprocessor therefore implements
``preprocess`` and ``predict(model, model_input)``, and usually ``postprocess``, which
``predict`` calls:

.. code-block:: python

    model_input = preprocessor.preprocess({"ct_cuboid": ct_cuboid, "energy": energy})
    outputs = preprocessor.predict(model, model_input)   # {quantity: array}

``ct_cuboid``
    The CT sampled on the model's grid (``metadata.dose_calc.grid_parameters``): Hounsfield
    units as a floating-point array of shape ``(depth, lateral, lateral)``, with ``-1000`` for
    points outside the CT. There is no batch axis: all beamlets of the ray, which differ only in
    their energy, share the cuboid.

``energy``
    The nominal energies in MeV of the beamlets to predict, a one-dimensional floating-point
    array of shape ``(batch,)``. Its length is at most the engine's ``max_batch_size`` (a ray
    is split into several calls beyond it) and may be 1.

Both arrive in the engine's array namespace and on its device, i.e. NumPy, CuPy, PyTorch or
JAX arrays depending on ``pyRadPlan.settings.xp``. A preprocessor therefore converts them to
the framework of its model first, e.g. with ``pyRadPlan.core.xp_utils.to_namespace("torch",
array)`` or with ``array_api_compat.array_namespace`` for plain array operations.

``preprocess`` turns them into whatever ``model_input`` the model needs (for example a
``(batch, channels, depth, lateral, lateral)`` tensor); ``model_input`` is passed on untouched.
``predict`` receives the loaded model, already in eval mode and on its device, and runs the
forward pass(es) -- a single one, or several for an ensemble, as the model's author sees fit --
and the scaling to physical units, typically by calling ``postprocess``. It must return a
dictionary mapping each quantity name to an array of shape ``(batch, depth, lateral, lateral)``
on the grid of the ``ct_cuboid``, with an entry for every quantity declared under
``metadata.dose_calc.quantities`` (``["physical_dose"]`` if none are declared, and
``physical_dose`` must be among them, listed first); a declared quantity that is missing makes
the calculation fail. The arrays may be returned in the model's own framework (and on its
device): the engine converts them to its array namespace. ``physical_dose`` decides which voxels a beamlet stores, and its unit becomes the
unit of the influence matrix.

A sketch of such a preprocessor; the dose-calculation reference model is a complete one:

.. code-block:: python

    import torch

    from pyRadPlan.ai.modelhub import BasePreprocessor
    from pyRadPlan.core.xp_utils import to_namespace


    class MyDoseCalcPreprocessor(BasePreprocessor):
        def preprocess(self, inputs: dict) -> torch.Tensor:
            ct = to_namespace("torch", inputs["ct_cuboid"]).float()
            energy = to_namespace("torch", inputs["energy"]).float().reshape(-1)
            ...  # normalize and stack to (batch, channels, depth, lateral, lateral)

        def postprocess(self, outputs: torch.Tensor) -> dict:
            return {"physical_dose": outputs[:, 0] * self.config["max_dose"]}

        def predict(self, model, model_input):
            with torch.inference_mode():
                return self.postprocess(model(model_input.to(next(model.parameters()).device)))

See :ref:`concept_ai_dose_calculation` for the geometry of the grid, the engine parameters and
device handling, and the :doc:`AI dose calculation tutorial </tutorials/examples/ai_dosecalc>`
for a runnable example. AI dose calculation is a research feature and not intended for clinical
use.

API reference
-------------

The full API for the ``ai.modelhub`` module is documented in the
:doc:`API reference </api/ai_modelhub>`:

* :func:`~pyRadPlan.ai.modelhub.load_model`
* :class:`~pyRadPlan.ai.modelhub.BasePreprocessor`
* :class:`~pyRadPlan.ai.modelhub.OutcomePreprocessor`
* :func:`~pyRadPlan.ai.modelhub.load_pyanno4rt_model`
* :func:`~pyRadPlan.ai.modelhub.list_local_models`
* :class:`~pyRadPlan.ai.modelhub.ModelTask`
* :class:`~pyRadPlan.ai.modelhub.AiSettings`
