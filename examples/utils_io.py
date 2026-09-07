# %% [markdown]
"""# Example on how to import and export patient data in pyRadPlan."""

# %% [markdown]
# pyRadPlan ships a small, extensible import/export framework in `pyRadPlan.io`.
# It supports MATLAB (`.mat`, matRad-compatible), DICOM (CT, RTSTRUCT, SEG, RTDOSE),
# the SimpleITK-based image formats (NIfTI/NRRD/MetaImage), NumPy `.npz` and pickle.
#
# There are two layers:
#
# 1. A simple **top-level API**: `load_patient`, `load_data` and `save_data`.
# 2. **Low-level handlers** (`MatlabHandler`, `DicomHandler`) for fine-grained control.
#
# To display this script in a Jupyter Notebook, install jupytext via pip and run:
#
# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/utils_io.py
# ```
# %%
# some imports
import tempfile
from pathlib import Path

from pyRadPlan import (
    load_tg119,
    load_patient,
    load_data,
    save_data,
)
from pyRadPlan.io import MatlabHandler, DicomHandler

# %% [markdown]
# ## Top-level API
#
# The most convenient entry point. `load_patient` returns the CT image (CT) and the
# Structure Set (CST), choosing the right loader automatically from the path
# (a `.mat` file, or a folder/file of DICOM data).
# %%
# Load the TG119 phantom bundled with pyRadPlan ...
ct, cst = load_tg119()

# ... which is equivalent to loading the patient file directly:
from importlib import resources  # noqa: E402

tg119_path = resources.files("pyRadPlan.data.phantoms").joinpath("TG119.mat")
ct, cst = load_patient(tg119_path)

# %% [markdown]
# `load_data` loads *everything* it finds into a dictionary. For a matRad `.mat` file this
# may contain `ct`, `cst` and (if present) `dose`; for a DICOM folder it collects the CT
# series, structures and dose. Missing pieces are simply omitted.
# %%
data = load_data(tg119_path)
print("Loaded keys:", list(data))
ct = data["ct"]
cst = data["cst"]

# %% [markdown]
# ## Saving data
#
# `save_data` writes one or more objects. The format is chosen (in this order) from an
# explicit `format=` argument, the extension of `file_name`, or a fast default (`.mat`).
# %%
work_dir = Path(tempfile.mkdtemp())

# Save into a single .mat file (format taken from the extension):
save_data(ct=ct, cst=cst, file_name=str(work_dir / "patient.mat"))

# No extension? The default format is appended automatically (-> patient2.mat):
save_data(ct=ct, file_name=str(work_dir / "patient2"))

# Force a format explicitly, regardless of the file name:
save_data(ct=ct, file_name=str(work_dir / "patient3"), format="mat")

# A dict of named objects works too:
save_data({"ct": ct, "cst": cst}, file_name=str(work_dir / "patient4.mat"))

print("Wrote:", sorted(p.name for p in work_dir.glob("*.mat")))

# %% [markdown]
# ## Low-level handlers
#
# Each format also has a handler that bundles importing and exporting and lets you load
# individual objects. A `MatlabHandler` is bound to a single file.
# %%
handler = MatlabHandler(work_dir / "patient.mat")
ct = handler.load_ct()  # load just the CT
cst = handler.load_cst(ct)  # load just the StructureSet
# handler.load_patient() -> (ct, cst); handler.load_data() -> dict of everything

# Saving via the handler is equivalent to save_data with that format:
handler_out = MatlabHandler(work_dir / "from_handler.mat")
handler_out.save(ct=ct, cst=cst)

# %% [markdown]
# ## DICOM import / export
#
# DICOM is directory-based. Here we export the phantom to a folder as a CT series plus an
# RTSTRUCT, then import it back. CT geometry/HU and structure masks are preserved.
# %%
dicom_dir = work_dir / "dicom"
DicomHandler(dicom_dir).save(ct=ct, cst=cst)
print("DICOM files:", sorted(p.name for p in dicom_dir.glob("*.dcm")))

# Re-import the whole folder:
ct_dcm, cst_dcm = load_patient(dicom_dir)
print("Imported structures:", [voi.name for voi in cst_dcm.vois])

# %% [markdown]
# Structures are exported as RTSTRUCT by default. To export them as a DICOM SEG object
# instead (which stores the voxel masks directly), use the exporter's `structure_format`:
# %%
from pyRadPlan.io.dicom import DicomExporter  # noqa: E402

seg_dir = work_dir / "dicom_seg"
DicomExporter(seg_dir, structure_format="seg").save(ct=ct, cst=cst)
print("SEG export:", sorted(p.name for p in seg_dir.glob("*.dcm")))

# %% [markdown]
# That's it! The framework also accepts a dose distribution (a `SimpleITK.Image`) via the
# `dose=` argument of `save_data` / the handlers, which is written as DICOM RTDOSE or stored
# in the matRad `resultGUI`. See `utils_matrad.py` for matRad-specific (de)serialization.
