# `pyRadPlan.ai_models` — AI model handling

Loads AI models and their preprocessors from the HuggingFace Hub or a local
directory, with version dedup and offline support.

## Quick start

```python
from pyRadPlan.ai_models import load_model, list_local_models

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
```

Loading a model from the Hub runs Python shipped inside the repository, so it
requires an explicit `trust_remote_code=True` (or
`PYRADPLAN_AI_MODELS_TRUST_REMOTE_CODE=1`). A directory you pass yourself as
`local_dir` is exempt — that code is already under your control.

`list_local_models` lists the models present under
`AiModelSettings.local_models_dir` and in the HuggingFace cache; it performs no
network access. Models are reported as `<org>/<repo>` -- which `load_model`
accepts directly, so the output feeds straight back in, and a private fork is
never confused with its upstream namesake.

A model's *task* -- what it does -- comes from `metadata.task` in its
`model_config.json`, falling back to the repository-name prefix (`dosecalc-*` /
`outcome-*`), so you can filter with `list_local_models(ModelTask.DOSE_CALC)`.
Task is deliberately distinct from *modality* in the pyRadPlan sense (the
radiation type), which a repository records under `metadata.training.modality`.

## Configuration

`AiModelSettings` reads environment variables prefixed with
`PYRADPLAN_AI_MODELS_` (or a `.env` file):

| Setting | Env var | Default | Meaning |
|---|---|---|---|
| `hf_org` | `PYRADPLAN_AI_MODELS_HF_ORG` | `DKFZ-RadOpt` | HuggingFace org/namespace for registered repos |
| `local_models_dir` | `PYRADPLAN_AI_MODELS_LOCAL_MODELS_DIR` | `<data_dir>/ai_models` | Base dir for local models; default `local_dir` is `<path>/<org>/<repo>`. Empty string uses the HF cache only |
| `cache_dir` | `PYRADPLAN_AI_MODELS_CACHE_DIR` | HF default | HuggingFace cache directory |
| `offline` | `PYRADPLAN_AI_MODELS_OFFLINE` | `False` | Force `local_files_only` |
| `trust_remote_code` | `PYRADPLAN_AI_MODELS_TRUST_REMOTE_CODE` | `False` | Allow executing code shipped with a model fetched from the Hub |
| `device` | `PYRADPLAN_AI_MODELS_DEVICE` | `cpu` | Device the model is moved to |

`PYRADPLAN_HUGGINGFACE_PATH` is still accepted as a legacy alias for
`PYRADPLAN_AI_MODELS_LOCAL_MODELS_DIR`; the prefixed name wins if both are set.

The data root (`<data_dir>`) defaults to `~/.pyradplan` and can be relocated with
the `PYRADPLAN_DATA_DIR` environment variable (see `pyRadPlan.core.get_data_dir`).

Downloaded models are laid out as `<local_models_dir>/<org>/<repo>`.

## Versions and updates

Pin a `revision` (tag, branch or commit) for reproducible loads: a local copy
recorded as that revision is used directly, without touching the network.
Without a pinned revision the Hub is asked whether the local copy is still
current, and a stale one is refreshed; if the Hub cannot be reached the local
copy is used anyway, with a warning.

## Model-repository contract

Each model lives in its own HuggingFace repo (or a local directory) containing
exactly these files:

```
<model folder>/
  model.py             # nn.Module class definition(s)
  preprocessor.py      # subclass of pyRadPlan.ai_models.BasePreprocessor
  weights.safetensors  # state_dict saved with safetensors
  model_config.json    # declarative config (see model_config.template.json)
```

Instantiation is driven entirely by `model_config.json` — there is no extra
init file. Any model type (dose calc, outcome, …) uses the same handful of
loader-consumed keys; everything else is free-form metadata the loader ignores:

- `model_name` selects the class in `model.py`; `model_params` are passed to its
  constructor as keyword arguments (so keep only real constructor kwargs there).
- the `BasePreprocessor` subclass in `preprocessor.py` is constructed with
  `model_preprocessing`, which configures **both** `preprocess()` and
  `postprocess()`. If `preprocessor.py` defines more than one subclass, set
  `preprocessor_name` in `model_config.json` to disambiguate.
- `metadata` is otherwise free-form and not read by the loader; use it to record
  how the model was trained and any assumptions it makes. Only `metadata.task`
  is interpreted, by `list_local_models`.

```json
{
    "model_name": "ExampleNet",
    "model_params": {"in_channels": 3, "hidden_channels": 4, "out_features": 1},
    "model_preprocessing": {"type_order": ["dose", "ct", "mask"]},
    "metadata": {"task": "outcome", "description": "…", "training": {"modality": "protons"}}
}
```

A model folder's `model.py` and `preprocessor.py` are imported under a package
name unique to the folder, so several models can be loaded side by side. Import
sibling files relatively (`from .preprocessor import …`).

A working reference implementation lives in `test/data/ai_models/dummy_model/`
and is exercised by the test suite. The annotated config template is
`pyRadPlan/data/ai_models/model_config.template.json`.

> **Security note:** `load_model` executes `model.py` and `preprocessor.py`
> from the model repository. For anything fetched from the Hub this is gated
> behind `trust_remote_code`, which defaults to `False`; enable it only for
> sources you trust, and prefer pinning a `revision` so you know which code you
> are running.
