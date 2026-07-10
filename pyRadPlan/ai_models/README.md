# `pyRadPlan.ai_models` — AI model handling

Loads AI models and their preprocessors from the HuggingFace Hub or a local
directory, with version dedup and offline support.

## Quick start

```python
from pyRadPlan.ai_models import load_model, list_local_models

# From a local directory (works fully offline):
model, preprocessor = load_model(local_dir="path/to/model_folder")

# By friendly name on HuggingFace (resolves to <hf_org>/<name>):
model, preprocessor = load_model("outcome-ORPDenseNet-tg119", revision="v1.0.0")

# Explicit repository:
model, preprocessor = load_model(repo_id="my-org/my-repo")

print(list_local_models())  # models available on disk (no network)
```

`list_local_models` lists the models present under
`AiModelSettings.local_models_dir`; it performs no network access. The modality
is derived from the repository name (`dosecalc-*` / `outcome-*`), so you can
filter with `list_local_models(Modality.DOSE_CALC)`. Each model lives in its own
repository and is listed under the repository name.

## Configuration

`AiModelSettings` reads environment variables prefixed with
`PYRADPLAN_AI_MODELS_` (or a `.env` file):

| Setting | Env var | Default | Meaning |
|---|---|---|---|
| `hf_org` | `PYRADPLAN_AI_MODELS_HF_ORG` | `DKFZ-RadOpt` | HuggingFace org/namespace for registered repos |
| `local_models_dir` | `PYRADPLAN_HUGGINGFACE_PATH` | `<data_dir>/ai_models` | Base dir for local models; default `local_dir` is `<path>/<repo>`. Empty string uses the HF cache only |
| `cache_dir` | `PYRADPLAN_AI_MODELS_CACHE_DIR` | HF default | HuggingFace cache directory |
| `offline` | `PYRADPLAN_AI_MODELS_OFFLINE` | `False` | Force `local_files_only` |
| `trust_remote_code` | `PYRADPLAN_AI_MODELS_TRUST_REMOTE_CODE` | `True` | Allow executing shipped model code |
| `device` | `PYRADPLAN_AI_MODELS_DEVICE` | `cpu` | Device the model is moved to |

The data root (`<data_dir>`) defaults to `~/.pyradplan` and can be relocated with
the `PYRADPLAN_DATA_DIR` environment variable (see `pyRadPlan.core.get_data_dir`).

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
- `metadata` is optional and never read by the loader; use it to record how the
  model was trained and any assumptions it makes.

```json
{
    "model_name": "ExampleNet",
    "model_params": {"in_channels": 3, "hidden_channels": 4, "out_features": 1},
    "model_preprocessing": {"type_order": ["dose", "ct", "mask"]},
    "metadata": {"description": "…", "training": {"modality": "protons"}}
}
```

A working reference implementation ships in
`pyRadPlan/data/ai_models/example_model/`. The annotated config template is
`pyRadPlan/data/ai_models/model_config.template.json`.

> **Security note:** `load_model` executes `model.py` and `preprocessor.py`
> from the model repository. This is gated behind `trust_remote_code`
> (default `True`); disable it for untrusted sources.
