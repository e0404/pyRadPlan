"""Import/export of a PlanResult through the io backends."""

import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan.io import load_data, save_data
from pyRadPlan.io.matlab import MatlabImporter
from pyRadPlan.result import PlanResult, create_result


@pytest.fixture
def result():
    """A plan result with two quantities, per-beam doses and a fluence vector."""
    image = sitk.GetImageFromArray(np.random.rand(4, 6, 8))
    image.SetSpacing((2.0, 2.0, 3.0))
    return create_result(
        {
            "physical_dose": image,
            "let": image,
            "physical_dose_beam": [image, image],
        },
        fluence=np.arange(5.0),
        used_optimizer="ipopt",
    )


@pytest.mark.parametrize("extension", [".mat", ".pkl"])
def test_result_roundtrip_container_formats(tmp_path, result, extension):
    """Container formats carry the whole result, not just a dose cube."""
    path = tmp_path / f"result{extension}"
    save_data(file_name=str(path), result=result)

    loaded = load_data(str(path))["result"]

    assert isinstance(loaded, PlanResult)
    assert sorted(loaded.stored_quantities) == ["let", "physical_dose"]
    assert loaded.stored_beam_quantities == ["physical_dose"]
    assert len(loaded.beam.physical_dose) == 2
    assert loaded.used_optimizer == "ipopt"
    np.testing.assert_allclose(loaded.fluence, np.arange(5.0))
    np.testing.assert_allclose(
        sitk.GetArrayFromImage(loaded["physical_dose"].values),
        sitk.GetArrayFromImage(result["physical_dose"].values),
    )


def test_matlab_export_uses_resultgui_key(tmp_path, result):
    """matRad reads the result from "resultGUI"; exporting must use that name."""
    path = tmp_path / "result.mat"
    save_data(file_name=str(path), result=result)

    mdict = MatlabImporter(str(path)).mdict
    assert "resultGUI" in mdict
    assert "physicalDose" in mdict["resultGUI"]
    assert "physicalDose_beam1" in mdict["resultGUI"]


@pytest.mark.parametrize("extension", [".npz", ".nii.gz"])
def test_single_image_formats_fall_back_to_physical_dose(tmp_path, result, extension):
    """Formats that store one image write the result's physical dose."""
    path = tmp_path / f"result{extension}"
    save_data(file_name=str(path), result=result)

    assert path.exists()


def test_matlab_importer_loads_result(test_data_photons_raw, tmp_path):
    """A matRad resultGUI is validated into a PlanResult by the importer."""
    from scipy.io import savemat

    path = tmp_path / "photons.mat"
    savemat(str(path), {"resultGUI": test_data_photons_raw["resultGUI"]})

    importer = MatlabImporter(str(path))
    loaded = importer.load_result()

    assert isinstance(loaded, PlanResult)
    assert "physical_dose" in loaded
    assert len(loaded.beam.physical_dose) == 2


def test_importer_without_result_returns_none(tg119_path):
    """The TG119 phantom carries no result; the importer reports that as None."""
    assert MatlabImporter(str(tg119_path)).load_result() is None
