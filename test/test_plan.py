import pytest
from pyRadPlan.scenarios import NominalScenario
from pyRadPlan.plan import create_pln, PhotonPlan, IonPlan


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
    pln_from_dict.pop("bio_model")
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
    pln_to_dict.pop("bio_model")
    assert pln_dict_snake == pln_to_dict

    pln_to_dict_camel = pln.to_matrad()
    pln_to_dict_camel.pop("multScen")
    pln_to_dict_camel.pop("bioModel")
    print(set(pln_dict_camel) ^ set(pln_to_dict_camel))
    assert pln_dict_camel == pln_to_dict_camel


def test_plan_to_matrad():
    plan = PhotonPlan(num_of_fractions=30)
    pln = plan.to_matrad()

    assert "numOfFractions" in pln
    # assert isinstance(pln["numOfFractions"],float)
    assert pln["numOfFractions"] == float(30)
