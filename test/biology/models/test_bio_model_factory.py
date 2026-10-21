import numpy as np
import pytest

from pyRadPlan.biology.models import (
    MKM,
    ConstantRBEModel,
    EmptyModel,
    KernelBasedLQModel,
    TabulatedAlphaBetaModel,
    Wedenberg,
    available_bio_models,
    create_bio_model,
    get_available_models,
    get_bio_model,
)
from pyRadPlan.biology.models._factory import BIO_MODELS
from pyRadPlan.biology.tissues import LEMTissueProperties


def test_model_parameters_and_to_dict():
    assert ConstantRBEModel().to_dict() == {"model": "constant_rbe"}
    assert ConstantRBEModel(rbe=1.0).to_dict() == {"model": "constant_rbe", "rbe": 1.0}
    assert ConstantRBEModel(1.2).parameters == {"rbe": 1.2}
    assert Wedenberg(p1=0.5).to_dict() == {"model": "WED", "p1": 0.5}
    assert KernelBasedLQModel().to_dict() == {"model": "alpha_beta_kernels"}
    assert repr(Wedenberg(p1=0.5)) == "Wedenberg(p1=0.5)"


@pytest.mark.parametrize(
    ("old", "new", "cls"),
    [
        ("kernel_based_lq", "alpha_beta_kernels", KernelBasedLQModel),
        ("LEM", "alpha_beta_kernels", KernelBasedLQModel),
        ("dose_average_alpha_beta", "alpha_beta_tables", TabulatedAlphaBetaModel),
        ("MKM", "zstar_kernels", MKM),
        ("zs_based_lq", "zstar_kernels", MKM),
        ("constRBE", "constant_rbe", ConstantRBEModel),
    ],
)
def test_old_identifiers_create_the_renamed_formalisms(old, new, cls):
    """Plans, workspaces and dijs stored with the old identifiers keep loading."""
    model = create_bio_model({"model": old})
    assert type(model) is cls
    assert model.model == new
    assert model.to_dict()["model"] == new


@pytest.mark.parametrize(
    ("identifier", "matrad"),
    [
        ("alpha_beta_kernels", "LEM"),
        ("alpha_beta_tables", "dose_average_alpha_beta"),
        ("zstar_kernels", "MKM"),
        ("constant_rbe", "constRBE"),
    ],
)
def test_matrad_names_are_unchanged(identifier, matrad):
    assert create_bio_model(identifier).to_matrad() == matrad


def test_every_registered_model_has_a_display_name_and_description():
    expected = {
        "none": "None (physical dose)",
        "constant_rbe": "Constant RBE",
        "WED": "Wedenberg (LET)",
        "MCN": "McNamara (LET)",
        "CAR": "Carabe (LET)",
        "HEL": "Mairani (LET)",
        "LSM": "Linear scaling (LET)",
        "alpha_beta_kernels": "α/β kernels",
        "alpha_beta_tables": "α/β tables + fluence spectra",
        "zstar_kernels": "z* kernels",
    }
    classes = {cls.model: cls for cls in BIO_MODELS.values()}
    for identifier, name in expected.items():
        assert classes[identifier].name == name
        assert classes[identifier].description
        assert "\n" not in classes[identifier].description


def test_tissue_properties_type_is_deprecated():
    with pytest.warns(DeprecationWarning, match="tissue_properties_type"):
        assert KernelBasedLQModel().tissue_properties_type is LEMTissueProperties


def test_model_equality():
    assert ConstantRBEModel(rbe=1.1) == ConstantRBEModel(rbe=1.1)
    assert ConstantRBEModel(rbe=1.1) != ConstantRBEModel(rbe=1.0)
    assert ConstantRBEModel() != EmptyModel()
    frags = [[1.0, 1.0], [12.0, 6.0]]
    assert TabulatedAlphaBetaModel(fragments_to_include=frags) == TabulatedAlphaBetaModel(
        fragments_to_include=np.asarray(frags)
    )


def test_create_bio_model_from_name_dict_instance():
    assert isinstance(create_bio_model("none"), EmptyModel)
    assert isinstance(create_bio_model("LEM"), KernelBasedLQModel)  # alias

    model = create_bio_model({"model": "constant_rbe", "rbe": 1.0})
    assert isinstance(model, ConstantRBEModel) and model.rbe == 1.0
    assert create_bio_model({"name": "WED", "p1": 0.5}).p1_WED == 0.5

    instance = Wedenberg()
    assert create_bio_model(instance) is instance


def test_create_bio_model_errors():
    with pytest.raises(ValueError, match="Unknown biological model"):
        create_bio_model("WEDD")
    with pytest.raises(ValueError, match="needs a 'model' entry"):
        create_bio_model({"rbe": 1.0})
    with pytest.raises(ValueError, match="Accepted parameters: \\['rbe'\\]"):
        create_bio_model({"model": "constant_rbe", "rbee": 1.0})
    with pytest.raises(ValueError, match="does not support radiation mode 'protons'"):
        create_bio_model("HEL", radiation_mode="protons")
    with pytest.raises(ValueError, match="Cannot create a biological model from int"):
        create_bio_model(3)


def test_get_bio_model_checks_machine_quantities():
    model = get_bio_model({"model": "WED", "p1": 0.5}, "protons", ["physical_dose", "let"])
    assert model.p1_WED == 0.5
    with pytest.raises(ValueError, match="Required quantities.*let"):
        get_bio_model("WED", "protons", ["physical_dose"])
    with pytest.raises(ValueError, match="does not support radiation mode"):
        get_bio_model("WED", "carbon", ["physical_dose", "let"])


def test_get_available_models_filters():
    available = get_available_models("protons", ["physical_dose"])
    assert {"none", "constant_rbe"} <= set(available)
    assert "WED" not in available
    assert "WED" in get_available_models("protons", ["physical_dose", "let"])


def test_available_bio_models_by_radiation_mode():
    names = [cls.model for cls in available_bio_models("protons")]
    assert names[0] == "none"
    assert names[1:] == sorted(names[1:])
    assert "HEL" not in names and "LEM" not in names
    assert [cls.model for cls in available_bio_models("helium")].count("HEL") == 1
