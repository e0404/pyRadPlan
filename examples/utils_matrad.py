# %% [markdown]
"""# Example on how export/import of data structs to/from matRad is handled."""
# %% [markdown]
# Note, that pyRadPlan can be used completely independent of matlab. Since, many functionalities
# are yet to be implemented in pyRadPlan, it might be useful to have compatibility with matRad.
# You can import and export every pydantic data structure. <br>
# The general usage for export is: `data.to_matrad()`

# and for import simply use the given `data.validate_x()` where x is e.g. pln (plan-object)

# To display this script in a Jupyter Notebook, you need to install jupytext via pip and run the following command.
# This will create a .ipynb file in the same directory:

# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/utils_matrad.py
# %%
# some imports
from importlib import resources

from pyRadPlan import (
    load_tg119,
    load_patient,
    save_data,
    validate_cst,
    validate_ct,
)
from pyRadPlan.io import MatlabHandler, validate_matrad_patient

# %% [markdown]
# CT-image (CT) and Structure Set (CST):
# %%
# You can e.g. load the in pyRadPlan provided TG119 Phantom:
ct, cst = load_tg119()

# %%
# Alternatively, you can load any patient data from matRad.
tg119_path = resources.files("pyRadPlan.data.phantoms").joinpath("TG119.mat")
ct, cst = load_patient(tg119_path)

# %%
# Of course you can load them separately too. `MatlabHandler` is the low-level handler
# bundling the matRad importer and exporter for one .mat file:
handler = MatlabHandler(tg119_path)

# Load CT data
ct = handler.load_ct()

# Load CST data (the CT provides the reference geometry for the VOI masks)
cst = handler.load_cst(ct)

# %%
# The handler also exposes the raw matRad dictionary, so you can validate the structures
# yourself or reach data that pyRadPlan does not model yet:
print("Contents of the .mat file:", [k for k in handler.mdict if not k.startswith("__")])

ct = validate_ct(handler.mdict["ct"])
cst = validate_cst(handler.mdict["cst"], ct)

# `validate_matrad_patient` does that for a whole matRad workspace at once (ct, cst, pln,
# stf, dij, resultGUI) and returns a dict of the validated objects:
patient = validate_matrad_patient(dict(handler.mdict))
print("Validated structures:", list(patient))

# %%
# Export the data to matRad format. Every pyRadPlan structure provides `to_matrad()`:
ct_mat = ct.to_matrad()
cst_mat = cst.to_matrad()

# %%
# To write a matRad-readable file, use `save_data()`. It calls `to_matrad()` for you and
# collects everything into a single .mat file:
save_data(ct=ct, cst=cst, file_name="patient.mat")

# The handler is bound to its own path and does the same:
MatlabHandler("patient_from_handler.mat").save(ct=ct, cst=cst)

# %% [markdown]
# The same holds for all the other pyRadPlan data structures. To pull them out of a
# matRad workspace, hand `load_patient` a dict via `extra_plan_data`:

# ```python
# extra = {}
# ct, cst = load_patient("path/to/matRad_workspace.mat", extra_plan_data=extra)
# pln, stf, dij = extra["pln"], extra["stf"], extra["dij"]
#
# # ... and back to matRad:
# pln_mat = pln.to_matrad()
# stf_mat = stf.to_matrad()
# ```

# %% [markdown]
# You get the idea :). Same can be applied to dij and result!
