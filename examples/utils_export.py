# %% [markdown]
"""# Example on how to export a treatment plan and its results in pyRadPlan."""

# %% [markdown]
# This example plans a proton treatment on the TG119 phantom and saves it in the formats
# pyRadPlan can write:
#
# 1. **DICOM** for treatment planning systems and viewers: CT, RTSTRUCT, RTDOSE (with DVHs)
#    and the RT Ion Plan, which can also be imported again.
# 2. **HDF5 workspaces**, which store every object losslessly and load without running code
#    from the file (unlike pickle), and the lossless **JSON** mode of the data models.
# 3. **Image formats** (NIfTI, NRRD, MetaImage), including segmentations that keep
#    overlapping structures.
# 4. **DVH and QI tables** as CSV or JSON.
# 5. **Monte Carlo inputs** for TOPAS and FRED.
#
# `utils_io.py` introduces loading and saving of patient data (CT and structures) in general.
#
# To display this script in a Jupyter Notebook, install jupytext via pip and run:
#
# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/utils_export.py
# ```
# %%
import json
import logging
import tempfile
from pathlib import Path

from pyRadPlan import (
    IonPlan,
    calc_dose_influence,
    fluence_optimization,
    generate_stf,
    load_data,
    load_tg119,
    save_data,
)
from pyRadPlan.analysis import DVHCollection, QICollection
from pyRadPlan.io import ExportBundle
from pyRadPlan.io.dicom import DicomImporter
from pyRadPlan.stf import validate_stf

logging.basicConfig(level=logging.WARNING)

# %% [markdown]
# ## A plan to export
#
# A single proton beam on TG119 with the default objectives of the phantom.
# %%
ct, cst = load_tg119()

pln = IonPlan(radiation_mode="protons", machine="Generic", num_of_fractions=30)
pln.prop_stf = {"gantry_angles": [0.0], "couch_angles": [0.0]}
pln.prop_opt = {"solver": "scipy"}

stf = generate_stf(ct, cst, pln)
dij = calc_dose_influence(ct, cst, stf, pln)
result = fluence_optimization(ct, cst, stf, dij, pln)

print("Quantities of the result:", result.stored_quantities)
print("Number of spot weights:", result.fluence.size)

work_dir = Path(tempfile.mkdtemp())

# %% [markdown]
# ## DICOM
#
# DICOM is directory-based. `save_data` with `format="dcm"` writes the CT series and the
# structure set (RTSTRUCT). With a `result` it adds one RTDOSE per dose quantity (physical
# dose and RBE-weighted dose; other quantities such as LET are skipped with a warning). With
# `stf` and `pln` it also writes an RT Ion Plan whose spot weights are numbers of particles per
# fraction (the RTDOSE files reference it). Exporter options are passed in
# `exporter_options`. Here, `dvh=True` adds the DVHs of all structures to the RTDOSE files.
# %%
dicom_dir = work_dir / "dicom"
save_data(
    file_name=dicom_dir,
    format="dcm",
    ct=ct,
    cst=cst,
    result=result,
    pln=pln,
    stf=stf,
    exporter_options={"dvh": True},
)
names = sorted(p.name for p in dicom_dir.glob("*.dcm"))
print(f"{len(names)} DICOM files, e.g.:", [n for n in names if not n.startswith("CT")])

# %% [markdown]
# The patient metadata of an imported CT (`ct.dicom_info`) is kept, so an export of an
# imported patient stays in the same study. `exporter_options={"anonymize": True}` replaces
# the patient identifiers and generates new study UIDs.
#
# An export never replaces existing DICOM files silently. Exporting into the same folder
# again raises a `FileExistsError`, unless `overwrite=True` is given:
# %%
try:
    save_data(file_name=dicom_dir, format="dcm", ct=ct, cst=cst)
except FileExistsError as err:
    print("Refused:", err)

save_data(
    file_name=dicom_dir,
    format="dcm",
    ct=ct,
    cst=cst,
    result=result,
    pln=pln,
    stf=stf,
    exporter_options={"overwrite": True, "anonymize": True},
)

# %% [markdown]
# The exported folder can be imported again. `load_data` returns CT, structures and doses.
# The RT Ion Plan is read back as steering information, spot weights and an `IonPlan`, so
# the dose can be recalculated from it. Spots without weight are not part of the plan.
# %%
data = load_data(dicom_dir)
print("Imported:", sorted(data))

importer = DicomImporter(dicom_dir)
print("Plans:", [(p["label"], p["kind"], p["num_beams"]) for p in importer.list_plans()])
plan = importer.load_plan()
print("Spots with weight:", int((result.fluence > 0).sum()))
print("Imported spots:", plan["stf"].total_number_of_bixels)

# %% [markdown]
# Photon plans are exported as RT Plans as well: field-based beams with their jaws and
# multileaf collimator, bixel-based beams as an open field that only holds the geometry
# (with a warning that it is not deliverable). Leaf sequences are not exported yet.

# %% [markdown]
# ## HDF5 workspaces
#
# HDF5 stores every pyRadPlan object (CT with its DICOM metadata, structure set with
# objectives, plan, steering information, dose influence matrix, results) in one file. In
# contrast to pickle, loading an HDF5 file never runs code from the file, and the files can be
# inspected with any HDF5 tool. An `ExportBundle` collects the objects of a workspace.
# %%
bundle = ExportBundle(ct=ct, cst=cst, pln=pln, stf=stf, dij=dij, result=result)
workspace_file = work_dir / "workspace.h5"
bundle.save(workspace_file)
print(f"Workspace: {workspace_file.stat().st_size / 1e6:.1f} MB")

loaded = ExportBundle.load(workspace_file)
print("Same plan:", loaded.pln == pln, "| same steering information:", loaded.stf == stf)
print("Loaded result:", loaded.result.stored_quantities)

# %% [markdown]
# Saving to a format that cannot store some of the objects warns once and saves the rest.
# For example, NumPy `.npz` files only hold images:
# %%
bundle.save(work_dir / "images.npz")

# %% [markdown]
# ## Lossless JSON
#
# The data models also round-trip through JSON. Arrays, sparse matrices and images are
# written as tagged dictionaries, so the output is standard JSON:
# %%
stf_json = stf.model_dump_json()
print(f"Steering information as JSON: {len(stf_json) / 1e3:.0f} kB")
print("Round trip equal:", validate_stf(json.loads(stf_json)) == stf)

# %% [markdown]
# ## Image formats and segmentations
#
# NIfTI, NRRD and MetaImage store a structure set as a label map by default, with one label
# per voxel. A voxel in several structures can only keep one of them, which `save_data`
# reports. `segmentation="layers"` keeps the overlaps: for NRRD it writes a 3D Slicer
# segmentation (`cst.seg.nrrd`), for NIfTI and MetaImage a one-hot image with one channel
# per structure.
# %%
save_data(
    file_name=work_dir / "slicer",
    format="nrrd",
    ct=ct,
    cst=cst,
    dose=result.physical_dose.values,
    exporter_options={"segmentation": "layers"},
)
print("NRRD export:", sorted(p.name for p in (work_dir / "slicer").iterdir()))

# %% [markdown]
# ## DVH and QI tables
#
# DVHs and quality indicators can be written as CSV, e.g. for a spreadsheet, or as JSON.
# %%
dose = result.physical_dose
dvhs = DVHCollection.from_structure_set(cst=cst, dose=dose)
dvhs.to_csv(work_dir / "dvh.csv")
dvhs.to_json(work_dir / "dvh.json")

qis = QICollection.from_structure_set(cst=cst, dose=dose, ref_doses=[2], ref_vols=[50])
qis.to_csv(work_dir / "qi.csv")
print((work_dir / "qi.csv").read_text().splitlines()[0])

# %% [markdown]
# ## Monte Carlo inputs
#
# `format="topas"` and `format="fred"` write the input files of a Monte Carlo simulation of
# the plan without running it, e.g. to run it on a cluster. The histories are distributed
# over the spots according to the spot weights.
# %%
topas_dir = work_dir / "topas"
save_data(
    file_name=topas_dir,
    format="topas",
    ct=ct,
    cst=cst,
    pln=pln,
    stf=stf,
    result=result,
    exporter_options={"num_histories": 1e5},
)
print("TOPAS inputs:", sorted(p.name for p in topas_dir.iterdir())[:5], "...")
