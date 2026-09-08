# %% [markdown]
"""
# Loading AI Models in pyRadPlan.

This example shows how to load a model and its preprocessor with the
`pyRadPlan.ai.modelhub` subsystem. Models follow a fixed repository contract
(`model.py`, `preprocessor.py`, `weights.safetensors`, `model_config.json`) and
can be loaded either from a local directory or from the HuggingFace Hub.

## Prerequisites

1. A `torch` build matching your platform/CUDA (the Hub client itself ships
   with pyRadPlan).
2. By default, local models live under `<data_dir>/ai_models` (the data root
   defaults to `~/.pyradplan`, relocatable via `PYRADPLAN_DATA_DIR`). Override
   just the models base with the `PYRADPLAN_AI_MODELHUB_LOCAL_MODELS_DIR`
   environment variable (or a `.env` file).
3. Loading a model from the Hub runs Python shipped inside the repository, so it
   needs an explicit opt-in: `trust_remote_code=True`, or
   `PYRADPLAN_AI_MODELHUB_TRUST_REMOTE_CODE=1`. Loading from a directory you point
   at yourself with `local_dir=` needs no opt-in.

## Setup
"""

# %% tags=["active-ipynb"]
import logging

from dotenv import load_dotenv

from pyRadPlan import settings
from pyRadPlan.ai.modelhub import list_local_models, load_model, ModelTask

# %%
# Load environment variables (e.g. PYRADPLAN_AI_MODELHUB_LOCAL_MODELS_DIR) from a .env file
load_dotenv()

# Configure logging so the loader reports what it resolves/downloads
logging.basicConfig(level=logging.INFO)
# Quiet the per-request HTTP chatter from the HuggingFace client
logging.getLogger("httpx").setLevel(logging.WARNING)

# Inspect effective settings (the modelhub_* fields of the `ai` sub-configuration)
ai_cfg = settings.ai
print(f"HuggingFace org:    {ai_cfg.modelhub_hf_org}")
print(f"Local models dir:   {ai_cfg.modelhub_local_models_dir}")
print(f"Offline:            {ai_cfg.modelhub_offline}")
print(f"Device:             {ai_cfg.modelhub_device}")
print(f"Trust remote code:  {ai_cfg.modelhub_trust_remote_code}")

# %% [markdown]
"""
## Load from the HuggingFace Hub

Once the model repositories are published, you can load a model by its friendly
name, which resolves to `<hf_org>/<name>`. Set `PYRADPLAN_AI_MODELHUB_HF_ORG` to
the hosting organization and, optionally,
`PYRADPLAN_AI_MODELHUB_LOCAL_MODELS_DIR` to keep a local copy.

Pin a `revision` (tag/branch/commit): it makes the download reproducible, tells
you which code `trust_remote_code` is about to run, and lets a matching local
copy be reused without contacting the Hub. Without one, the Hub is asked whether
the local copy is still current on every load.
"""

# %%
model, preprocessor = load_model(
    "pyRadPlan-outcome-ORPDenseNet-Xerostomia",
    revision="v0.4",
    trust_remote_code=True,
    device=ai_cfg.modelhub_device,
)
# Or with an explicit repository id:
model, preprocessor = load_model(
    repo_id="DKFZ-RadOpt/pyRadPlan-outcome-ORPDenseNet-tg119",
    revision="v0.6",
    trust_remote_code=True,
    device=ai_cfg.modelhub_device,
)

# %% [markdown]
"""
## Available models

`list_local_models` reports the models available on disk, without any network
access. Models are listed as `<org>/<repo>`, which `load_model` accepts
directly, so a private fork is never confused with its upstream namesake. A
model's task comes from `metadata.task` in its `model_config.json`, falling
back to the `dosecalc-*` / `outcome-*` name prefix.
"""

# %%
print("Dose calculation models:")
for name in list_local_models(ModelTask.DOSE_CALC):
    print(f"  {name}")

print("\nOutcome models:")
for name in list_local_models(ModelTask.OUTCOME):
    print(f"  {name}")
