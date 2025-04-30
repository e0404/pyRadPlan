import pytest
import SimpleITK as sitk
import os
import numpy as np
from pathlib import Path
from pydantic import ValidationError
from scipy.io import savemat

from pyRadPlan.result import PlanResult, QuantityDistribution, validate_result, create_result


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample SimpleITK image for testing."""
    sample_array = np.random.rand(50, 100, 100) * 1000  # Random HU values
    image = sitk.GetImageFromArray(sample_array)
    image.SetOrigin((0, 0, 0))
    image.SetSpacing((1, 1, 2))
    image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    return image


def test_result_creation(sample_image):
    """Test creation of a PlanResult object from a SimpleITK image."""
    result = create_result(physical_dose=sample_image, let=sample_image)
    assert isinstance(result, PlanResult)
    assert isinstance(result.physical_dose, QuantityDistribution)
    assert result.physical_dose.size == (100, 100, 50)  # reverse order of axes
    assert result.physical_dose.resolution == {"x": 1, "y": 1, "z": 2}
    assert result.physical_dose.origin == (0, 0, 0)
    assert result.physical_dose.direction == (1, 0, 0, 0, 1, 0, 0, 0, 1)


def test_result_from_dict(sample_image):
    """Test creation of a PlanResult object from a dictionary."""
    dose_dict = {
        "cube": sample_image,
        "origin": (1, 2, 3),
        "direction": (1, 0, 0, 0, 1, 0, 0, 0, 1),
        "resolution": {"x": 2, "y": 2, "z": 4},
    }
    result = create_result({"physical_dose": dose_dict, "fluence": np.zeros((1, 1, 1, 1))})
    assert isinstance(result.fluence, np.ndarray)
    assert isinstance(result.physical_dose, QuantityDistribution)
    result.add_quantity("let", dose_dict)
    assert result.physical_dose.origin == (1, 2, 3)
    assert result.physical_dose.resolution == {"x": 2, "y": 2, "z": 4}


def test_result_from_dict2(sample_image):
    """Test creation of a PlanResult object from a dictionary."""
    dose_dict = {
        "cube": sample_image,
        "origin": (1, 2, 3),
        "direction": (1, 0, 0, 0, 1, 0, 0, 0, 1),
        "resolution": {"x": 2, "y": 2, "z": 4},
    }
    result = create_result(
        {"physical_dose": [dose_dict, dose_dict], "fluence": np.zeros((1, 1, 1, 1))}
    )
    assert isinstance(result.fluence, np.ndarray)
    assert isinstance(result.beam.physical_dose[0], QuantityDistribution)
    result.add_quantity("let", dose_dict)
    assert result.beam.physical_dose[0].origin == (1, 2, 3)
    assert result.beam.physical_dose[0].resolution == {"x": 2, "y": 2, "z": 4}


def test_saving_beam_cubes(sample_image):
    """Test saving of beam cubes in PlanResult."""
    result = create_result(
        physical_dose=sample_image,
        let=sample_image,
        fluence=np.zeros((1, 1, 1, 1)),
        physical_dose_beam=[sample_image, sample_image],
        let_beam=[sample_image, sample_image],
    )
    assert isinstance(result.beam.physical_dose[0], QuantityDistribution)
    assert isinstance(result.beam.let[0], QuantityDistribution)
    assert result.beam.physical_dose[0].size == (100, 100, 50)  # reverse order of axes
    assert result.beam.physical_dose[0].resolution == {"x": 1, "y": 1, "z": 2}
    assert result.beam.physical_dose[0].origin == (0, 0, 0)
    assert result.beam.physical_dose[0].direction == (1, 0, 0, 0, 1, 0, 0, 0, 1)


# TODO:
# def test_to_matrad_dict(sample_image):
#     """Test conversion of CT object to matRad dictionary format."""
#     result = create_result(
#         physical_dose=sample_image,
#         let=sample_image,
#         fluence=np.zeros((1, 1, 1, 1)),
#         info={"example_1": "example_str", "example_2": 1.0},
#         physical_dose_beam=[sample_image, sample_image],
#     )
#     matrad_dict = result.to_matrad()
#     assert "physicalDose" in matrad_dict
#     assert "let" in matrad_dict
#     assert "info" in matrad_dict
#     assert matrad_dict["physicalDose"].shape == (1,)
#     assert matrad_dict["physicalDose"][0].shape == (100, 100, 50)
#     assert matrad_dict["let"].shape == (1,)
#     assert matrad_dict["let"][0].shape == (100, 100, 50)
#     assert matrad_dict["physicalDoseBeam1"].shape == (1,)
#     assert matrad_dict["physicalDoseBeam1"][0].shape == (100, 100, 50)
#     # savemat("output.mat", matrad_dict)


def test_validate_result(sample_image):
    """Test validation of PlanResult objects."""
    result = create_result(physical_dose=sample_image, let=sample_image)
    validated_result = validate_result(result)
    assert result is validated_result

    validated_result = validate_result(physical_dose=sample_image, let=sample_image)
    assert isinstance(validated_result, PlanResult)


def test_invalid_input(sample_image):
    """Test handling of invalid input."""
    with pytest.raises(ValueError):
        create_result(physical_dose="invalid input")


def test_invalid_input_2(sample_image):
    """Test handling of invalid input."""
    with pytest.raises(ValueError):
        create_result(physical_dose=sample_image, let="invalid input")


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
    result = create_result(physical_dose=sample_image, let=sample_image)
    new_origin = (5, 5, 1)
    result.physical_dose.distribution.SetOrigin(new_origin)
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
    assert isinstance(result.physical_dose.distribution, sitk.Image)
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
    assert isinstance(result.physical_dose.distribution, sitk.Image)
    assert result.physical_dose.origin == (1, 2, 3)
    assert result.physical_dose.resolution == {"x": 2, "y": 2, "z": 4}


def test_missing_distribution():
    """Test handling of missing HU distribution."""
    with pytest.raises(ValidationError):
        PlanResult.model_validate({})
