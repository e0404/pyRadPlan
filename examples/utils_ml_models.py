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

from dotenv import load_dotenv

from pyRadPlan.ml import get_ml_settings, list_local_models, load_model, Modality

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
## Load from the HuggingFace Hub

Once the model repositories are published, you can load a model by its friendly
name, which resolves to `<hf_org>/<name>`. Set `PYRADPLAN_ML_HF_ORG` to the
hosting organization and, optionally, `PYRADPLAN_HUGGINGFACE_PATH` to keep a
local copy. Pin a `revision` (tag/branch/commit) for reproducible,
version-checked downloads.
"""

# %%
model, preprocessor = load_model(
    "pyRadPlan-outcome-ORPDenseNet-Xerostomia", revision="v0.1", device="cuda", offline=False
)
# Or with an explicit repository id:
model, preprocessor = load_model(
    repo_id="pyRadPlan/pyRadPlan-outcome-ORPDenseNet-Xerostomia",
    revision="v0.1",
    device="cuda",
    offline=False,
)

# %% [markdown]
"""
## Available models

`list_local_models` reports the models available on disk.
"""

# %%
print("Dose calculation models:")
for name in list_local_models(Modality.DOSE_CALC):
    print(f"  {name}")

print("\nOutcome models:")
for name in list_local_models(Modality.OUTCOME):
    print(f"  {name}")
