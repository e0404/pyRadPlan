import pytest
from pydantic import ValidationError
from pyRadPlan.scenarios import NominalScenario
from pyRadPlan.plan import create_pln, PhotonPlan, IonPlan
from pyRadPlan.bio_models import BiologicalModel, ConstantRBEModel, Wedenberg


def test_create_pln_no_args():
    with pytest.raises(ValueError):
        create_pln()


def test_ionPlnEmptyConstructor():
    plan = IonPlan()
    assert plan.radiation_mode == "protons"


def test_photonPlnEmptyConstructor():
    plan = PhotonPlan()
    assert plan.radiation_mode == "photons"


def test_create_pln_dict_photons():
    plan = create_pln({"radiation_mode": "photons"})
    assert isinstance(plan, PhotonPlan)
    assert plan.radiation_mode == "photons"


def test_create_pln_from_Plan():
    plan = PhotonPlan()
    new_plan = create_pln(plan)
    assert isinstance(new_plan, PhotonPlan)

    plan = IonPlan()
    new_plan = create_pln(plan)
    assert isinstance(new_plan, IonPlan)


def test_create_pln_dict_ions():
    plan = create_pln({"radiation_mode": "protons"})
    assert isinstance(plan, IonPlan)
    assert plan.radiation_mode == "protons"

    plan = create_pln({"radiation_mode": "carbon"})
    assert isinstance(plan, IonPlan)
    assert plan.radiation_mode == "carbon"


def test_create_pln_dict_unknown():
    with pytest.raises(ValueError):
        create_pln({"radiation_mode": "unknown"})


def test_create_pln_kwargs_photon():
    plan = create_pln(radiation_mode="photons")
    assert isinstance(plan, PhotonPlan)
    assert plan.radiation_mode == "photons"


def test_create_pln_kwargs_ions():
    plan = create_pln(radiation_mode="protons")
    assert isinstance(plan, IonPlan)
    assert plan.radiation_mode == "protons"

    plan = create_pln(radiation_mode="carbon")
    assert isinstance(plan, IonPlan)
    assert plan.radiation_mode == "carbon"


def test_create_pln_kwargs_unknown():
    with pytest.raises(ValueError):
        create_pln(radiation_mode="unknown")


def test_create_pln_dict_photons_snake():
    scen_dict = NominalScenario().model_dump()

    pln_dict = {
        "radiation_mode": "photons",  # either photons / protons / carbon
        "machine": "Generic",
        "num_of_fractions": 30,
        "prescribed_dose": 60.0,
        "prop_stf": {},
        # dose calculation settings
        "prop_dose_calc": {},
        # optimization settings
        "prop_opt": {},
        "prop_seq": {},
        "mult_scen": scen_dict,
    }

    pln = create_pln(pln_dict)
    assert isinstance(pln, PhotonPlan)
    assert pln.radiation_mode == "photons"
    assert pln.num_of_fractions == 30
    assert pln.machine == "Generic"
    assert pln.prescribed_dose == 60.0
    assert isinstance(pln.mult_scen, NominalScenario)

    pln_from_dict = pln.model_dump()
    # print(set(pln_dict) ^ set(pln_from_dict))
    pln_dict.pop("mult_scen")
    pln_from_dict.pop("mult_scen")
    assert pln_from_dict.pop("bio_model") == {"model": "none"}
    assert pln_from_dict.pop("dose_convention") == "per_fraction"
    assert pln_dict == pln_from_dict


def test_create_pln_dict_photons_camel():
    scen = NominalScenario()
    scen_dict_camel = scen.to_matrad()
    scen_dict_snake = scen.model_dump()

    pln_dict_camel = {
        "radiationMode": "photons",  # either photons / protons / carbon
        "machine": "Generic",
        "numOfFractions": 30,
        "prescribedDose": 60.0,
        "propStf": {},
        # dose calculation settings
        "propDoseCalc": {},
        # optimization settings
        "propOpt": {},
        "propSeq": {},
        "multScen": scen_dict_camel,
    }

    pln_dict_snake = {
        "radiation_mode": "photons",  # either photons / protons / carbon
        "machine": "Generic",
        "num_of_fractions": 30,
        "prescribed_dose": 60.0,
        "prop_stf": {},
        # dose calculation settings
        "prop_dose_calc": {},
        # optimization settings
        "prop_opt": {},
        "prop_seq": {},
        "mult_scen": scen_dict_snake,
    }

    pln = create_pln(pln_dict_camel)
    assert isinstance(pln, PhotonPlan)
    assert pln.radiation_mode == "photons"
    assert pln.num_of_fractions == 30
    assert pln.machine == "Generic"
    assert pln.prescribed_dose == 60.0
    assert isinstance(pln.mult_scen, NominalScenario)

    pln_dict_snake.pop("mult_scen")
    pln_dict_camel.pop("multScen")

    pln_to_dict = pln.model_dump()
    pln_to_dict.pop("mult_scen")
    assert pln_to_dict.pop("bio_model") == {"model": "none"}
    assert pln_to_dict.pop("dose_convention") == "per_fraction"
    assert pln_dict_snake == pln_to_dict

    pln_to_dict_camel = pln.to_matrad()
    pln_to_dict_camel.pop("multScen")
    assert pln_to_dict_camel.pop("bioModel") == "none"
    assert pln_to_dict_camel.pop("doseConvention") == "per_fraction"
    print(set(pln_dict_camel) ^ set(pln_to_dict_camel))
    assert pln_dict_camel == pln_to_dict_camel


def test_plan_to_matrad():
    plan = PhotonPlan(num_of_fractions=30)
    pln = plan.to_matrad()

    assert "numOfFractions" in pln
    # assert isinstance(pln["numOfFractions"],float)
    assert pln["numOfFractions"] == float(30)


@pytest.mark.parametrize(
    "radiation_mode, expected",
    [("protons", "constant_rbe"), ("helium", "HEL"), ("carbon", "kernel_based_lq")],
)
def test_ion_plan_default_bio_model(radiation_mode, expected):
    pln = IonPlan(radiation_mode=radiation_mode)
    assert isinstance(pln.bio_model, BiologicalModel)
    assert pln.bio_model.model == expected
    assert create_pln({"radiation_mode": radiation_mode}).bio_model.model == expected


def test_plan_explicit_bio_model_is_kept():
    assert IonPlan(radiation_mode="protons", bio_model="WED").bio_model.model == "WED"
    assert create_pln({"radiation_mode": "carbon", "bio_model": "LSM"}).bio_model.model == "LSM"
    assert PhotonPlan().bio_model.model == "none"


def test_plan_bio_model_from_dict_and_instance():
    pln = IonPlan(radiation_mode="protons", bio_model={"model": "constant_rbe", "rbe": 1.0})
    assert isinstance(pln.bio_model, ConstantRBEModel)
    assert pln.bio_model.rbe == 1.0
    assert pln.model_dump()["bio_model"] == {"model": "constant_rbe", "rbe": 1.0}
    assert pln.to_matrad()["bioModel"] == "constRBE"

    model = Wedenberg(p1=0.5)
    pln = IonPlan(radiation_mode="protons", bio_model=model)
    assert pln.bio_model is model

    # camelCase keys are accepted like everywhere else in the plan
    pln = create_pln(
        {"radiationMode": "protons", "bioModel": {"model": "constant_rbe", "rbe": 1.2}}
    )
    assert pln.bio_model.rbe == 1.2


def test_plan_bio_model_assignment_is_validated():
    pln = IonPlan(radiation_mode="protons")
    pln.bio_model = "MCN"
    assert pln.bio_model.model == "MCN"
    with pytest.raises(ValidationError):
        pln.bio_model = "HEL"  # helium only
    with pytest.raises(ValidationError):
        pln.bio_model = "does_not_exist"
    with pytest.raises(ValidationError):
        pln.bio_model = {"model": "constant_rbe", "rbee": 1.0}


def test_plan_round_trip_keeps_bio_model_parameters():
    pln = IonPlan(radiation_mode="carbon", bio_model={"model": "LSM", "upper_let_threshold": 20.0})
    pln2 = create_pln(pln.model_dump())
    assert pln2.bio_model == pln.bio_model
    assert pln2.bio_model.p_upperLETThreshold == 20.0


def test_plan_dose_convention():
    pln = IonPlan(radiation_mode="protons", num_of_fractions=30)
    assert pln.dose_convention == "per_fraction"
    assert pln.result_dose_factor == 1

    pln = IonPlan(radiation_mode="protons", num_of_fractions=30, dose_convention="total")
    assert pln.result_dose_factor == 30
    assert (
        create_pln({"radiation_mode": "protons", "doseConvention": "total"}).dose_convention
        == "total"
    )

    with pytest.raises(ValidationError):
        IonPlan(radiation_mode="protons", dose_convention="per_course")


def test_plan_bio_model_matrad_names():
    pln = create_pln({"radiationMode": "protons", "bioModel": "constRBE"})
    assert isinstance(pln.bio_model, ConstantRBEModel)
    assert pln.to_matrad()["bioModel"] == "constRBE"
    assert (
        create_pln({"radiationMode": "carbon", "bioModel": "LEM"}).to_matrad()["bioModel"] == "LEM"
    )
    assert IonPlan(radiation_mode="protons", bio_model="WED").to_matrad()["bioModel"] == "WED"


def test_plan_bio_model_from_matrad_struct_drops_metadata():
    struct = {
        "model": "constRBE",
        "RBE": 1.0,
        "possibleRadiationModes": ["protons"],
        "requiredQuantities": ["physicalDose"],
    }
    pln = create_pln({"radiationMode": "protons", "bioModel": struct})
    assert isinstance(pln.bio_model, ConstantRBEModel)
    assert pln.bio_model.rbe == 1.0

    # the default of the radiation mode is applied for camelCase input as well
    assert create_pln({"radiationMode": "carbon"}).bio_model.model == "kernel_based_lq"
