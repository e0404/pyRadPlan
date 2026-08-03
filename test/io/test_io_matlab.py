"""Tests for the MATLAB import/export backend."""

import numpy as np
import SimpleITK as sitk
import pytest

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.io import (
    load_patient,
    load_data,
    save_data,
    load_tg119,
    MatlabHandler,
    validate_matrad_patient,
)
from pyRadPlan.io.matlab import MatlabImporter, MatlabExporter
from pyRadPlan.io.matlab import _matfile as matfile


def _ct_arrays_close(a: CT, b: CT, atol=1e-6) -> bool:
    return (
        a.cube_hu.GetSize() == b.cube_hu.GetSize()
        and np.allclose(a.cube_hu.GetSpacing(), b.cube_hu.GetSpacing())
        and np.allclose(a.cube_hu.GetOrigin(), b.cube_hu.GetOrigin())
        and np.allclose(
            sitk.GetArrayFromImage(a.cube_hu), sitk.GetArrayFromImage(b.cube_hu), atol=atol
        )
    )


def _voxel_sets(cst: StructureSet) -> dict:
    return {v.name.lower(): set(v.indices_numpy.tolist()) for v in cst.vois}


# --------------------------------------------------------------------------
# Low-level .mat read/write and matRad validation
# --------------------------------------------------------------------------


def test_matfile_load_save_roundtrip(tmp_path, tg119_path):
    data = matfile.load(tg119_path)
    out = tmp_path / "copy.mat"
    matfile.save(out, {"ct": data["ct"], "cst": data["cst"]})
    assert out.exists()
    reloaded = matfile.load(out)
    assert "ct" in reloaded and "cst" in reloaded


def test_matfile_restores_legacy_opaque_layout():
    """MATLAB objects: scipy >= 1.18 renamed the opaque record fields pymatreader expects."""
    from scipy.io.matlab import MatlabOpaque

    opaque = np.empty(
        (1,), dtype=[("_TypeSystem", "O"), ("_Class", "O"), ("_ObjectMetadata", "O")]
    ).view(MatlabOpaque)
    opaque[0] = ("MCOS", "matRad_OptimizerIPOPT", np.array([1, 2], dtype=np.uint32))

    result_gui = np.empty((), dtype=[("physicalDose", "O"), ("usedOptimizer", "O")])
    result_gui[()] = (np.ones((2, 2)), opaque)

    restored = matfile._restore_legacy_opaque({"resultGUI": result_gui})["resultGUI"]
    legacy = restored["usedOptimizer"].item()
    assert legacy.dtype.names == ("s0", "s1", "s2", "arr")
    assert tuple(legacy[0])[:3] == (b"", b"MCOS", b"matRad_OptimizerIPOPT")


def test_validate_matrad_patient(tg119_path):
    mdict = matfile.load(tg119_path)
    patient = validate_matrad_patient(mdict)
    assert isinstance(patient["ct"], CT)
    assert isinstance(patient["cst"], StructureSet)


def test_load_patient_and_tg119(tg119_path):
    ct, cst = load_patient(tg119_path)
    ct2, cst2 = load_tg119()
    assert isinstance(ct, CT) and isinstance(cst, StructureSet)
    assert ct == ct2 and cst == cst2


def test_load_patient_extra_data(tg119_path):
    extra_plan, extra = {}, {}
    ct, cst = load_patient(tg119_path, extra_plan_data=extra_plan, extra_data=extra)
    assert isinstance(ct, CT)
    assert isinstance(extra_plan, dict) and isinstance(extra, dict)


def test_load_patient_missing_file():
    with pytest.raises(FileNotFoundError):
        load_patient("does_not_exist.mat")


# --------------------------------------------------------------------------
# New importer / exporter / handler
# --------------------------------------------------------------------------


def test_matlab_importer_per_object(tg119_path):
    importer = MatlabImporter(tg119_path)
    ct = importer.load_ct()
    cst = importer.load_cst(ct)
    assert isinstance(ct, CT)
    assert isinstance(cst, StructureSet)
    assert cst.ct_image == ct


def test_matlab_importer_load_data(tg119_path):
    data = MatlabImporter(tg119_path).load_data()
    assert isinstance(data["ct"], CT)
    assert isinstance(data["cst"], StructureSet)


def test_matlab_handler_is_importer_and_exporter(tmp_path, tg119_path):
    ct, cst = load_tg119()
    out = tmp_path / "patient.mat"
    handler = MatlabHandler(out)
    handler.save(ct=ct, cst=cst)
    assert out.exists()

    reloaded = MatlabHandler(out)
    ct2 = reloaded.load_ct()
    assert _ct_arrays_close(ct, ct2)


def test_matlab_roundtrip_preserves_ct_and_cst(tmp_path):
    ct, cst = load_tg119()
    out = tmp_path / "patient.mat"

    save_data(ct=ct, cst=cst, file_name=str(out))
    assert out.exists()

    data = load_data(out)
    ct2, cst2 = data["ct"], data["cst"]

    assert _ct_arrays_close(ct, ct2)

    sets_before = _voxel_sets(cst)
    sets_after = _voxel_sets(cst2)
    assert set(sets_before) == set(sets_after)
    for name, voxels in sets_before.items():
        assert voxels == sets_after[name], f"VOI '{name}' changed after roundtrip"


def test_matlab_exporter_requires_data(tmp_path):
    with pytest.raises(ValueError):
        MatlabExporter(tmp_path / "empty.mat").save()


def test_matlab_roundtrip_preserves_plan(tmp_path):
    """A saved plan survives a .mat round-trip"""
    from pyRadPlan.plan import validate_pln  # noqa: PLC0415

    ct, cst = load_tg119()
    pln = validate_pln({"radiationMode": "photons", "machine": "Generic"})
    out = tmp_path / "workspace.mat"

    save_data(ct=ct, cst=cst, pln=pln, file_name=str(out))
    data = load_data(out)

    assert "pln" in data, "plan was dropped on reload"
    reloaded = data["pln"]
    assert reloaded.radiation_mode == "photons"
    assert reloaded.machine == "Generic"
    # The empty workflow property blocks come back as {}, not None.
    for prop in ("prop_stf", "prop_opt", "prop_dose_calc", "prop_seq"):
        assert getattr(reloaded, prop) == {}
    # mult_scen re-validated (the exact regression that crashed the reload).
    assert reloaded.mult_scen.ct_scen_prob == [(0, 1.0)]
