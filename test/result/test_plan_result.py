import pytest
import SimpleITK as sitk
import numpy as np
from pydantic import ValidationError
# from scipy.io import savemat

from pyRadPlan.result import PlanResult, QuantityDistribution, validate_result, create_result


@pytest.fixture
def sample_image():
    """Create a sample SimpleITK image for testing."""
    sample_array = np.random.rand(50, 100, 100) * 1000  # Random HU values
    image = sitk.GetImageFromArray(sample_array)
    image.SetOrigin((0, 0, 0))
    image.SetSpacing((1, 1, 2))
    image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return image


def test_result_from_arg(sample_image):
    """Test creation of a PlanResult object from a SimpleITK image."""
    result = create_result(
        physical_dose=sample_image, physical_dose_beam=[sample_image, sample_image]
    )
    assert isinstance(result, PlanResult)
    assert isinstance(result.physical_dose, QuantityDistribution)
    assert result.physical_dose.size == (100, 100, 50)  # reverse order of axes
    assert result.physical_dose.resolution == {"x": 1, "y": 1, "z": 2}
    assert result.physical_dose.origin == (0, 0, 0)
    assert result.physical_dose.direction == (1, 0, 0, 0, 1, 0, 0, 0, 1)

    assert isinstance(result.beam.physical_dose[0], QuantityDistribution)
    assert result.beam.physical_dose[0].size == (100, 100, 50)  # reverse order of axes
    assert result.beam.physical_dose[0].resolution == {"x": 1, "y": 1, "z": 2}
    assert result.beam.physical_dose[0].origin == (0, 0, 0)
    assert result.beam.physical_dose[0].direction == (1, 0, 0, 0, 1, 0, 0, 0, 1)


def test_result_from_dict(sample_image):
    """Test creation of a PlanResult object from a dictionary."""
    dose_dict = {
        "cube": sample_image,
        "origin": (1, 2, 3),
        "direction": (1, 0, 0, 0, 1, 0, 0, 0, 1),
        "resolution": {"x": 2, "y": 2, "z": 4},
    }
    result = create_result(
        {
            "physical_dose": dose_dict,
            "physical_dose_beam": [dose_dict, dose_dict],
        }
    )
    assert isinstance(result.physical_dose, QuantityDistribution)
    assert result.physical_dose.origin == (1, 2, 3)
    assert result.physical_dose.resolution == {"x": 2, "y": 2, "z": 4}
    assert result.physical_dose.direction == (1, 0, 0, 0, 1, 0, 0, 0, 1)

    assert isinstance(result.beam.physical_dose[0], QuantityDistribution)
    assert result.beam.physical_dose[0].origin == (1, 2, 3)
    assert result.beam.physical_dose[0].resolution == {"x": 2, "y": 2, "z": 4}
    assert result.beam.physical_dose[0].direction == (1, 0, 0, 0, 1, 0, 0, 0, 1)


def test_add_quantity_arg(sample_image):
    """Test adding a quantity to a PlanResult object using keyword arg."""
    result = create_result(physical_dose=sample_image)
    assert isinstance(result.physical_dose, QuantityDistribution)

    new_quantity = np.random.rand(50, 100, 100) * 10
    result.add_quantity(let=new_quantity)
    assert "let" in result.quantities
    assert isinstance(result.let, QuantityDistribution)
    assert result.let.size == (100, 100, 50)


def test_add_quantity_str(sample_image):
    """Test adding a quantity to a PlanResult object using string and array."""
    result = create_result(physical_dose=sample_image)
    assert isinstance(result.physical_dose, QuantityDistribution)

    new_quantity = np.random.rand(50, 100, 100) * 10
    result.add_quantity("let", new_quantity)
    assert "let" in result.quantities
    assert isinstance(result.let, QuantityDistribution)
    assert result.let.size == (100, 100, 50)


def test_add_beam_quantity_arg(sample_image):
    """Test adding a beam quantity to a PlanResult object using keyword arg."""
    result = create_result(physical_dose=[sample_image])
    assert isinstance(result.beam.physical_dose[0], QuantityDistribution)

    result.add_beam_quantity(let=[sample_image])
    assert "let" in result.beam
    assert isinstance(result.beam.let[0], QuantityDistribution)
    assert result.beam.let[0].size == (100, 100, 50)


def test_add_beam_quantity_str(sample_image):
    """Test adding a beam quantity to a PlanResult object using string and array."""
    result = create_result(physical_dose=[sample_image])
    assert isinstance(result.beam.physical_dose[0], QuantityDistribution)

    new_quantity = np.random.rand(50, 100, 100) * 10
    result.add_beam_quantity("let", [new_quantity])
    assert "let" in result.beam
    assert isinstance(result.beam.let[0], QuantityDistribution)
    assert result.beam.let[0].size == (100, 100, 50)


def test_stored_quantities(sample_image):
    """Test the stored quantities in a PlanResult object."""
    result = create_result(physical_dose=sample_image, let=sample_image)
    assert isinstance(result.physical_dose, QuantityDistribution)
    assert isinstance(result.let, QuantityDistribution)
    assert result.stored_quantities == ["physical_dose", "let"]


def test_stored_beam_quantities(sample_image):
    """Test the stored beam quantities in a PlanResult object."""
    result = create_result(
        physical_dose=[sample_image, sample_image], let=[sample_image, sample_image]
    )
    assert isinstance(result.beam.physical_dose[0], QuantityDistribution)
    assert isinstance(result.beam.let[0], QuantityDistribution)
    assert result.stored_beam_quantities == ["physical_dose", "let"]


def test_to_matrad_dict(sample_image):
    """Test conversion of CT object to matRad dictionary format."""
    result = create_result(
        physical_dose=sample_image,
        let=sample_image,
        fluence=np.zeros((1, 1, 1, 1)),
        info={"example_1": "example_str", "example_2": 1.0},
        physical_dose_beam=[sample_image, sample_image],
    )
    matrad_dict = result.to_matrad()
    assert "physicalDose" in matrad_dict
    assert "let" in matrad_dict
    assert "info" in matrad_dict
    assert matrad_dict["physicalDose"].shape == (100, 100, 50)
    assert matrad_dict["let"].shape == (100, 100, 50)
    assert matrad_dict["physicalDose_beam1"].shape == (100, 100, 50)

    # savemat("output.mat", matrad_dict)


def test_matrad_import(test_data_photons_raw):
    """Test import of matRad resultGUI dictionary."""
    matRad_file = test_data_photons_raw["resultGUI"]
    result = validate_result(matRad_file)

    assert isinstance(result, PlanResult)
    assert isinstance(result.physical_dose, QuantityDistribution)
    assert isinstance(result.physical_dose.values, sitk.Image)
    assert result.physical_dose.size == (10, 20, 10)

    assert isinstance(result.beam.physical_dose[0], QuantityDistribution)
    assert isinstance(result.beam.physical_dose[1], QuantityDistribution)
    assert isinstance(result.beam.physical_dose[0].values, sitk.Image)
    assert isinstance(result.beam.physical_dose[1].values, sitk.Image)
    assert result.beam.physical_dose[0].size == (10, 20, 10)
    assert result.beam.physical_dose[1].size == (10, 20, 10)


def test_validated_result(sample_image):
    """Test validation of PlanResult objects."""
    result = create_result(physical_dose=sample_image, let=sample_image)
    validated_result = validate_result(result)
    assert result is validated_result
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
    result.physical_dose.values.SetOrigin(new_origin)
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
    assert isinstance(result.physical_dose.values, sitk.Image)
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
    assert isinstance(result.physical_dose.values, sitk.Image)
    assert result.physical_dose.origin == (1, 2, 3)
    assert result.physical_dose.resolution == {"x": 2, "y": 2, "z": 4}


def test_missing_distribution():
    """Test handling of missing HU distribution."""
    with pytest.raises(ValidationError):
        PlanResult.model_validate({})
