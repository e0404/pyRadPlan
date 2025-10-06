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
# jupytext --to notebook path/to/this/file/matrad_import_export.py
# %%
# some imports
from importlib import resources
from scipy.io import savemat
import pymatreader

from pyRadPlan import (
    load_tg119,
    load_patient,
    validate_cst,
    validate_ct,
)

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
# Of course you can load them separately too:
tg119_path = resources.files("pyRadPlan.data.phantoms").joinpath("TG119.mat")

# Read .mat files via pymatreader:
ct_mat, cst_mat = pymatreader.read_mat(tg119_path)

# Load CT data
ct = validate_ct(ct_mat)

# Load CST data
cst = validate_cst(cst_mat)

# %%
# Export the data to matRad format:
ct_mat = ct.to_matrad()
cst_mat = cst.to_matrad()

# Save them as .mat files:
savemat("ct.mat", {"ct": ct_mat})
savemat("cst.mat", {"cst": cst_mat})

# %% [markdown]
# The latter works also for all the other pyRadPlan data structures:

# ```bash
# pln_mat = pymatreader.read_mat(path/to/file_pln)
# pln = validate_pln(pln_mat)
# pln_exp = pln.to_matrad()

# stf_mat = pymatreader.read_mat(path/to/file_stf)
# stf = validate_stf(stf_mat)
# stf_exp = stf.to_matrad()
# %% [markdown]
# You get the idea :). Same can be applied to dij and result!
