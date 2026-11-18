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
    assert "LET" in matrad_dict
    assert "info" in matrad_dict
    # matRad calls the beamlet weights "w".
    assert "w" in matrad_dict
    assert matrad_dict["physicalDose"].shape == (100, 100, 50)
    assert matrad_dict["LET"].shape == (100, 100, 50)
    assert matrad_dict["physicalDose_beam1"].shape == (100, 100, 50)

    # savemat("output.mat", matrad_dict)


def test_to_matrad_roundtrip(sample_image):
    """A result exported for matRad validates back into an equivalent result."""
    result = create_result(
        physical_dose=sample_image,
        let=sample_image,
        fluence=np.arange(5.0),
        physical_dose_beam=[sample_image, sample_image],
        used_optimizer="ipopt",
    )

    reimported = validate_result(result.to_matrad())

    assert reimported.stored_quantities == ["physical_dose", "let"]
    assert reimported.stored_beam_quantities == ["physical_dose"]
    assert len(reimported.beam.physical_dose) == 2
    assert reimported.used_optimizer == "ipopt"
    np.testing.assert_allclose(reimported.fluence, np.arange(5.0))
    np.testing.assert_allclose(
        sitk.GetArrayFromImage(reimported.physical_dose.values),
        sitk.GetArrayFromImage(result.physical_dose.values),
    )


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


def test_matrad_import_orders_beams_by_number(sample_image):
    """Per-beam quantities follow the beam number, not the key order of the file."""
    array = sitk.GetArrayFromImage(sample_image)
    matrad_dict = {"physicalDose": array}
    # Deliberately shuffled, and past beam 9 so lexical ordering would differ.
    for beam in [3, 11, 1, 12, 2, 10]:
        matrad_dict[f"physicalDose_beam{beam}"] = np.full_like(array, float(beam))

    result = validate_result(matrad_dict)

    first_values = [sitk.GetArrayFromImage(qd.values).flat[0] for qd in result.beam.physical_dose]
    assert first_values == [1.0, 2.0, 3.0, 10.0, 11.0, 12.0]


def test_validation_does_not_consume_input(sample_image):
    """The caller's dictionary must survive validation unchanged."""
    data = {"physical_dose": sample_image, "info": {"iterations": 3}}
    snapshot = dict(data)

    validate_result(data)

    assert data == snapshot


def test_result_item_access(sample_image):
    """Quantities are reachable by name as well as by attribute."""
    result = create_result(
        physical_dose=sample_image, physical_dose_beam=[sample_image, sample_image]
    )

    assert result["physical_dose"] is result.physical_dose
    assert "physical_dose" in result
    assert "let" not in result
    assert "physical_dose" in result.beam
    assert result.beam.get("let") is None
    with pytest.raises(KeyError):
        result["let"]


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


def test_scenario_quantities_from_dict(sample_image):
    """``"<key>_scenario"`` lists become per-scenario quantities, keeping empty slots."""
    result = create_result(
        {
            "physical_dose": sample_image,
            "physical_dose_beam": [sample_image],
            "physical_dose_scenario": [sample_image, None, sample_image],
            "scenario_labels": ("nominal", "derived", "setup"),
        }
    )
    assert result.stored_quantities == ["physical_dose"]
    assert result.stored_beam_quantities == ["physical_dose"]
    assert result.stored_scenario_quantities == ["physical_dose"]
    assert isinstance(result.scenario.physical_dose[0], QuantityDistribution)
    assert result.scenario.physical_dose[1] is None
    assert result.scenario_labels == ("nominal", "derived", "setup")

    with pytest.raises(AttributeError):
        result.scenario_quantities = {}

    result.add_scenario_quantity("let", [None, sample_image, None])
    assert result.scenario["let"][0] is None


def test_accessors_follow_a_copy(sample_image):
    result = create_result(
        {"physical_dose_beam": [sample_image], "physical_dose_scenario": [sample_image]}
    )
    assert "physical_dose" in result.beam and "physical_dose" in result.scenario
    copy = result.model_copy(update={"beam_quantities": {}, "scenario_quantities": {}})
    assert "physical_dose" not in copy.beam and "physical_dose" not in copy.scenario


def test_to_matrad_leaves_out_scenario_quantities(sample_image):
    result = create_result(
        {
            "physical_dose": sample_image,
            "physical_dose_scenario": [sample_image, None],
            "scenario_labels": ("a", "b"),
        }
    )
    matrad_dict = result.to_matrad()
    assert "physicalDose" in matrad_dict
    assert "scenarioQuantities" not in matrad_dict
    assert "scenarioLabels" not in matrad_dict


@pytest.mark.parametrize(
    "data",
    [
        {"physicalDose": np.zeros((5, 6, 4))},
        {"physicalDose": np.zeros((5, 6, 4)), "w": np.ones(3)},
        {"RBExDose": np.zeros((5, 6, 4))},
    ],
)
def test_matrad_result_detected_without_beam_fields(data):
    """A matRad result is recognized by its field names, not only by ``_beam1``."""
    result = create_result(data)

    (name,) = result.stored_quantities
    assert name in ("physical_dose", "rbe_x_dose")
    # matRad stores (y, x, z); the image is sized (x, y, z)
    assert result[name].values.GetSize() == (6, 5, 4)


def test_matrad_quantity_names_roundtrip():
    """The irregular matRad field names map to the pyRadPlan quantity identifiers."""
    cube = np.random.rand(5, 6, 4)
    result = PlanResult.from_matrad(
        {"physicalDose": cube, "RBExDose": cube, "LET": cube, "RBExDose_beam1": cube}
    )

    assert sorted(result.stored_quantities) == ["let", "physical_dose", "rbe_x_dose"]
    assert result.stored_beam_quantities == ["rbe_x_dose"]

    exported = result.to_matrad()
    assert {"physicalDose", "RBExDose", "LET", "RBExDose_beam1"} <= set(exported)
    np.testing.assert_allclose(exported["RBExDose"], cube)


def test_matrad_import_ignores_fields_that_are_no_cubes(caplog):
    """Derived data in a resultGUI must not invalidate its cubes."""
    cube = np.zeros((5, 6, 4))
    with caplog.at_level("WARNING"):
        result = PlanResult.from_matrad(
            {
                "physicalDose": cube,
                "w": np.ones(7),
                "qi": [{"name": "target"}],
                "dvh": {"doseGrid": np.zeros(10)},
                "someVector": np.ones(7),
            }
        )

    assert result.stored_quantities == ["physical_dose"]
    assert result.stored_beam_quantities == []
    np.testing.assert_allclose(result.fluence, np.ones(7))
    assert "qi" in caplog.text
    assert "someVector" in caplog.text


def test_matrad_import_takes_geometry_from_reference():
    """matRad cubes carry no geometry; a reference image supplies it for all of them."""
    reference = sitk.GetImageFromArray(np.zeros((4, 5, 6)))
    reference.SetSpacing((2.0, 3.0, 4.0))
    reference.SetOrigin((-10.0, 20.0, 5.0))
    cube = np.zeros((5, 6, 4))

    result = PlanResult.from_matrad(
        {"physicalDose": cube, "LET": cube, "physicalDose_beam1": cube},
        reference_image=reference,
    )

    for distribution in (result.physical_dose, result.let, result.beam.physical_dose[0]):
        assert distribution.values.GetSpacing() == (2.0, 3.0, 4.0)
        assert distribution.values.GetOrigin() == (-10.0, 20.0, 5.0)


def test_unknown_non_quantity_key_names_the_key():
    with pytest.raises(ValueError, match="'weights'"):
        create_result({"physical_dose": np.zeros((4, 5, 6)), "weights": np.ones(7)})


def test_create_result_leaves_input_dict_unchanged(sample_image):
    data = {"physical_dose": sample_image}
    create_result(data, fluence=np.ones(3))
    assert list(data) == ["physical_dose"]


def test_fluence_accepts_sequences_and_backend_arrays(sample_image):
    import array_api_strict as xp

    result = create_result({"physical_dose": sample_image}, fluence=[1, 2, 3])
    assert isinstance(result.fluence, np.ndarray)

    result.fluence = xp.asarray([1.0, 2.0])
    assert isinstance(result.fluence, np.ndarray)
    np.testing.assert_allclose(result.fluence, [1.0, 2.0])


def _image(shape=(4, 5, 6), spacing=(1.0, 1.0, 1.0)):
    image = sitk.GetImageFromArray(np.zeros(shape))
    image.SetSpacing(spacing)
    return image


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"physical_dose": _image(), "let": _image((2, 2, 2))}, "share one grid"),
        ({"physical_dose": _image(), "let": _image(spacing=(2.0, 1.0, 1.0))}, "share one grid"),
        (
            {"physical_dose_beam": [_image()], "let_beam": [_image(), _image()]},
            "number of beams",
        ),
        (
            {
                "physical_dose_scenario": [_image(), None, None],
                "scenario_labels": ("nominal",),
            },
            "scenario labels",
        ),
        ({"quantities": {"fluence": _image()}}, "quantity name"),
        ({"quantities": {"beam": _image()}}, "quantity name"),
    ],
)
def test_inconsistent_results_are_rejected(data, match):
    with pytest.raises(ValueError, match=match):
        create_result(data)


def test_bare_arrays_adopt_the_grid_of_the_result():
    reference = _image(spacing=(2.0, 3.0, 4.0))

    result = create_result({"physical_dose": reference, "let": np.zeros((4, 5, 6))})
    result.add_quantity("effect", np.zeros((4, 5, 6)))

    assert result.let.values.GetSpacing() == (2.0, 3.0, 4.0)
    assert result.effect.values.GetSpacing() == (2.0, 3.0, 4.0)


def test_added_quantities_are_checked_and_rolled_back():
    result = create_result({"physical_dose": _image()})

    with pytest.raises(ValueError, match="share one grid"):
        result.add_quantity("let", _image((2, 2, 2)))
    with pytest.raises(ValueError, match="quantity name"):
        result.add_quantity("info", _image())

    assert result.stored_quantities == ["physical_dose"]
    assert result.info == {}


def test_quantity_without_known_origin_is_accepted():
    """Consistency is structural: an imported quantity need not know its provenance."""
    result = create_result({"physical_dose": _image()})
    result.add_quantity("import_rbe_x_dose", _image())
    assert "import_rbe_x_dose" in result


def test_distribution_takes_geometry_from_grid():
    from pyRadPlan.core import Grid
    from pyRadPlan.result import create_distribution

    reference = _image(spacing=(2.0, 3.0, 4.0))
    reference.SetOrigin((1.0, 2.0, 3.0))
    grid = Grid.from_sitk_image(reference)

    distribution = create_distribution({"values": np.zeros((4, 5, 6)), "grid": grid})

    assert distribution.values.GetSpacing() == (2.0, 3.0, 4.0)
    assert distribution.values.GetOrigin() == (1.0, 2.0, 3.0)

    with pytest.raises(ValueError, match="do not fit the grid"):
        create_distribution({"values": np.zeros((2, 2, 2)), "grid": grid})


def test_distribution_rejects_arrays_that_are_no_volume():
    from pyRadPlan.result import create_distribution

    with pytest.raises(ValueError, match="3D"):
        create_distribution(np.ones(7))
