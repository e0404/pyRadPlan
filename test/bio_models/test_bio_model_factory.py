import numpy as np
import pytest

from pyRadPlan.bio_models import (
    ConstantRBEModel,
    EmptyModel,
    KernelBasedLQModel,
    TabulatedAlphaBetaModel,
    Wedenberg,
    create_bio_model,
    get_available_models,
    get_bio_model,
)


def test_model_parameters_and_to_dict():
    assert ConstantRBEModel().to_dict() == {"model": "constant_rbe"}
    assert ConstantRBEModel(rbe=1.0).to_dict() == {"model": "constant_rbe", "rbe": 1.0}
    assert ConstantRBEModel(1.2).parameters == {"rbe": 1.2}
    assert Wedenberg(p1=0.5).to_dict() == {"model": "WED", "p1": 0.5}
    assert KernelBasedLQModel().to_dict() == {"model": "kernel_based_lq"}
    assert repr(Wedenberg(p1=0.5)) == "Wedenberg(p1=0.5)"


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
    with pytest.raises(ValueError, match="not available"):
        get_bio_model("WED", "protons", ["physical_dose"])
    with pytest.raises(ValueError, match="does not support radiation mode"):
        get_bio_model("WED", "carbon", ["physical_dose", "let"])


def test_get_available_models_filters():
    available = get_available_models("protons", ["physical_dose"])
    assert {"none", "constant_rbe"} <= set(available)
    assert "WED" not in available
    assert "WED" in get_available_models("protons", ["physical_dose", "let"])
