# %% [markdown]
"""
# Loading ML Models in pyRadPlan.

This example shows how to load a model and its preprocessor with the
`pyRadPlan.ml` subsystem. Models follow a fixed repository contract
(`model.py`, `preprocessor.py`, `weights.safetensors`, `model_config.json`) and
can be loaded either from a local directory or from the HuggingFace Hub.

## Prerequisites

1. The optional ML dependencies installed: `pip install pyRadPlan[ml]`
   (plus a `torch` build matching your platform/CUDA).
2. Optionally, set a base directory for local models via the
   `PYRADPLAN_HUGGINGFACE_PATH` environment variable (or a `.env` file).

## Setup
"""

# %% tags=["active-ipynb"]
import logging
from importlib import resources

import numpy as np
import torch
from dotenv import load_dotenv

from pyRadPlan.ml import get_ml_settings, list_models, load_model, Modality

# %%
# Load environment variables (e.g. PYRADPLAN_HUGGINGFACE_PATH) from a .env file
load_dotenv()

# Configure logging so the loader reports what it resolves/downloads
logging.basicConfig(level=logging.INFO)
# Quiet the per-request HTTP chatter from the HuggingFace client
logging.getLogger("httpx").setLevel(logging.WARNING)

# Inspect effective settings
settings = get_ml_settings()
print(f"HuggingFace org:    {settings.hf_org}")
print(f"Local models dir:   {settings.local_models_dir}")
print(f"Offline:            {settings.offline}")
print(f"Device:             {settings.device}")

# %% [markdown]
"""
## Available models

`list_models` discovers models dynamically: locally downloaded models (with all
versions available on disk) and, when online, models published on the HuggingFace
Hub under `MlSettings.hf_org` (most recent version only). The modality is derived
from the repository name (`dosecalc-*` / `outcome-*`).
"""


# %%
def _fmt(spec):
    local = f"local={list(spec.local_versions)}" if spec.is_local else "local=-"
    remote = f"hub={spec.remote_version}" if spec.is_remote else "hub=-"
    return f"  {spec.name}  ({local}, {remote})"


print("Dose calculation models:")
for spec in list_models(Modality.DOSE_CALC):
    print(_fmt(spec))

print("\nOutcome models:")
for spec in list_models(Modality.OUTCOME):
    print(_fmt(spec))

# Offline: only models already on disk (no network access)
print("\nAvailable offline:")
for spec in list_models(offline=True):
    print(_fmt(spec))

# %% [markdown]
"""
## Load from a local directory

`pyRadPlan` ships a tiny example model that implements the repository contract.
We load it directly via `local_dir`, which works fully offline. `load_model`
returns the model (in eval mode) together with its preprocessor.
"""

# %%
example_dir = resources.files("pyRadPlan.data.ml_models").joinpath("example_model")

model, preprocessor = load_model(local_dir=str(example_dir))
print(f"Loaded model:        {type(model).__name__}")
print(f"Loaded preprocessor: {type(preprocessor).__name__}")

# %% [markdown]
"""
## Run the preprocessor and the model

The preprocessor turns a `{modality: volume}` mapping into a model-ready tensor;
the model then produces its prediction. Here we feed dummy volumes matching the
example model's expected input.
"""

# %%
rng = np.random.default_rng()
inputs = {name: rng.random((8, 8, 8)) for name in ("dose", "ct", "mask")}

x = preprocessor(inputs)
print(f"Preprocessed tensor shape: {tuple(x.shape)}")

with torch.no_grad():
    prediction = model(x)
print(f"Model output:              {prediction.squeeze().item():.4f}")

# %% [markdown]
"""
## Load from the HuggingFace Hub

Once the model repositories are published, you can load a registered model by its
friendly name. Set `PYRADPLAN_ML_HF_ORG` to the hosting organization and,
optionally, `PYRADPLAN_HUGGINGFACE_PATH` to keep a local copy. Pin a `revision`
(tag/branch/commit) for reproducible, version-checked downloads.
"""

# %%
model, preprocessor = load_model("outcome-ORPDenseNet-tg119")  # , revision="v1.0.0")
# Or fully explicit (bypassing the registry):
model, preprocessor = load_model(repo_id="my-org/dosecalc-bayes-protons", subfolder="lung")
pass
