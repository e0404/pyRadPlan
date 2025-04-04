import pytest
import SimpleITK as sitk
import os
import numpy as np
from pathlib import Path
from pydantic import ValidationError

from pyRadPlan.result import PlanResult, QuantityCube, validate_result, create_result


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample SimpleITK image for testing."""
    sample_array = np.random.rand(50, 100, 100) * 1000  # Random HU values
    image = sitk.GetImageFromArray(sample_array)
    image.SetOrigin((0, 0, 0))
    image.SetSpacing((1, 1, 2))
    image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    """Save it for testing ct_from_file method"""
    writer = sitk.ImageFileWriter()
    writer.SetUseCompression(True)
    writer.SetImageIO("NiftiImageIO")

    file_path = os.path.join(tmp_path, "DummyCT.nii.gz")

    writer.SetFileName(file_path)
    writer.Execute(image)

    return image, file_path


def test_result_creation(sample_image):
    """Test creation of a PlanResult object from a SimpleITK image."""
    result = create_result(physical_dose=sample_image[0], let=sample_image[0])
    assert isinstance(result, PlanResult)
    assert isinstance(result.physical_dose, QuantityCube)
    assert result.physical_dose.size == (100, 100, 50)  # reverse order of axes
    assert result.physical_dose.resolution == {"x": 1, "y": 1, "z": 2}
    assert result.physical_dose.origin == (0, 0, 0)
    assert result.physical_dose.direction == (1, 0, 0, 0, 1, 0, 0, 0, 1)


def test_result_from_dict(sample_image):
    """Test creation of a PlanResult object from a dictionary."""
    dose_dict = {
        "cube": sample_image[0],
        "origin": (1, 2, 3),
        "direction": (1, 0, 0, 0, 1, 0, 0, 0, 1),
        "resolution": {"x": 2, "y": 2, "z": 4},
    }
    result = create_result(physical_dose=dose_dict)
    assert isinstance(result.physical_dose, QuantityCube)
    assert result.physical_dose.origin == (1, 2, 3)
    assert result.physical_dose.resolution == {"x": 2, "y": 2, "z": 4}


# def test_ct_to_matrad_dict(sample_image):
#     """Test conversion of CT object to matRad dictionary format."""
#     result = create_result(physical_dose=sample_image[0], let=sample_image[0])
#     matrad_dict = result.to_matrad()
#     assert "cubeHU" in matrad_dict
#     assert "cubeDim" in matrad_dict
#     assert matrad_dict["cubeDim"].tolist() == [100.0, 100.0, 50.0]  # reverse order of axes
#     assert matrad_dict["cubeHU"].shape == (1,)
#     assert matrad_dict["cubeHU"][0].shape == (100, 100, 50)  # reverse order of axes


# def test_ct_from_file(sample_image):
#     """Test creation of a CT object from a .nii.gz file."""
#     sample_file = Path(sample_image[1])
#     if sample_file.exists():
#         ct = ct_from_file(sample_file)
#         assert isinstance(ct, CT)
#     else:
#         pytest.skip("Sample DICOM file not available")


def test_validate_result(sample_image):
    """Test validation of PlanResult objects."""
    result = create_result(physical_dose=sample_image[0], let=sample_image[0])
    validated_result = validate_result(result)
    assert result is validated_result

    validated_result = validate_result(physical_dose=sample_image[0], let=sample_image[0])
    assert isinstance(validated_result, PlanResult)


def test_invalid_input(sample_image):
    """Test handling of invalid input."""
    with pytest.raises(ValueError):
        create_result(physical_dose="invalid input")


def test_invalid_input_2(sample_image):
    """Test handling of invalid input."""
    with pytest.raises(ValueError):
        create_result(physical_dose=sample_image[0], let="invalid input")


@pytest.mark.parametrize(
    "size,spacing",
    [((5, 5, 5), (2, 2, 2)), ((20, 30, 40), (0.5, 0.5, 1)), ((100, 100, 100), (0.1, 0.1, 0.1))],
)
def test_various_image_sizes(size, spacing):
    """Test CT creation with various image sizes and spacings."""
    # Numpy axis order is z,y,x and SimpleITK axis order is x,y,z. Need to flip size before assertion
    sample_array = np.random.rand(*size) * 1000
    image = sitk.GetImageFromArray(sample_array)
    image.SetSpacing(spacing)
    result = create_result(physical_dose=image)
    assert result.physical_dose.size == size[::-1]
    assert result.physical_dose.resolution == {"x": spacing[0], "y": spacing[1], "z": spacing[2]}


def test_property_change(sample_image):
    """Test changing properties of an existing CT object."""
    result = create_result(physical_dose=sample_image[0], let=sample_image[0])
    new_origin = (5, 5, 1)
    result.physical_dose.cube.SetOrigin(new_origin)
    assert result.physical_dose.origin == new_origin


def test_check_dict_convert_sitk_image():
    """Test the check_dict_convert_sitk_image method."""
    sample_array = np.random.rand(100, 200, 200) * 1000
    dose_dict = {
        "cube": sample_array,
        "origin": (1, 2, 3),
        "direction": (1, 0, 0, 0, 1, 0, 0, 0, 1),
        "resolution": {"x": 2, "y": 2, "z": 4},
    }
    result = create_result(physical_dose=dose_dict)
    assert isinstance(result.physical_dose.cube, sitk.Image)
    assert result.physical_dose.origin == (1, 2, 3)
    assert result.physical_dose.resolution == {"x": 2, "y": 2, "z": 4}


def test_check_multiple_scenarios_from_numpy():
    """Test the conversion to 4D image."""
    sample_array = np.ndarray((2,), dtype=object)
    sample_array[0] = np.random.rand(100, 200, 200) * 1000
    sample_array[1] = np.random.rand(100, 200, 200) * 1000
    dose_dict = {
        "physical_dose": {
            "cube": sample_array,
            "origin": (1, 2, 3, 0),
            "direction": (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
            "resolution": {"x": 2, "y": 2, "z": 4},
        }
    }

    result = PlanResult.model_validate(dose_dict)
    assert isinstance(result.physical_dose.cube, sitk.Image)
    assert result.physical_dose.origin == (1, 2, 3)
    assert result.physical_dose.resolution == {"x": 2, "y": 2, "z": 4}


def test_missing_cube():
    """Test handling of missing HU cube."""
    with pytest.raises(ValidationError):
        PlanResult.model_validate({})
