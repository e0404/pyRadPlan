# %% [markdown]
"""
# Loading AI Models in pyRadPlan.

This example shows how to load a model and its preprocessor with the
`pyRadPlan.ai_models` subsystem. Models follow a fixed repository contract
(`model.py`, `preprocessor.py`, `weights.safetensors`, `model_config.json`) and
can be loaded either from a local directory or from the HuggingFace Hub.

## Prerequisites

1. The optional AI dependencies installed: `pip install pyRadPlan[ai]`
   (plus a `torch` build matching your platform/CUDA).
2. By default, local models live under `<data_dir>/ai_models` (the data root
   defaults to `~/.pyradplan`, relocatable via `PYRADPLAN_DATA_DIR`). Override
   just the models base with the `PYRADPLAN_HUGGINGFACE_PATH` environment
   variable (or a `.env` file).

## Setup
"""

# %% tags=["active-ipynb"]
import logging

from dotenv import load_dotenv

from pyRadPlan import settings
from pyRadPlan.ai_models import list_local_models, load_model, Modality

# %%
# Load environment variables (e.g. PYRADPLAN_HUGGINGFACE_PATH) from a .env file
load_dotenv()

# Configure logging so the loader reports what it resolves/downloads
logging.basicConfig(level=logging.INFO)
# Quiet the per-request HTTP chatter from the HuggingFace client
logging.getLogger("httpx").setLevel(logging.WARNING)

# Inspect effective settings (the `ai_models` sub-configuration of pyRadPlan.settings)
ai_models = settings.ai_models
print(f"HuggingFace org:    {ai_models.hf_org}")
print(f"Local models dir:   {ai_models.local_models_dir}")
print(f"Offline:            {ai_models.offline}")
print(f"Device:             {ai_models.device}")

# %% [markdown]
"""
## Load from the HuggingFace Hub

Once the model repositories are published, you can load a model by its friendly
name, which resolves to `<hf_org>/<name>`. Set `PYRADPLAN_AI_MODELS_HF_ORG` to the
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
    repo_id="DKFZ-RadOpt/pyRadPlan-outcome-ORPDenseNet-tg119",
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
