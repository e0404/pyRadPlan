import numpy as np
import pytest
from pydantic import ValidationError
from pyRadPlan.machines import (
    IonAccelerator,
    PhotonLINAC,
    find_machine_file,
    load_from_name,
    load_machine_from_mat,
    validate_machine,
)
from pyRadPlan.scenarios import NominalScenario
from pyRadPlan.plan import create_pln, PhotonPlan, IonPlan
from pyRadPlan.biology.models import BiologicalModel, ConstantRBEModel, Wedenberg


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
    [
        ("protons", "constant_rbe"),
        ("helium", "HEL"),
        ("carbon", "alpha_beta_kernels"),
        ("oxygen", "alpha_beta_kernels"),
    ],
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
    matrad = pln.to_matrad()
    assert matrad["bioModel"] == "constRBE"
    assert matrad["bioModelParameters"] == {"rbe": 1.0}
    round_trip = create_pln(matrad)
    assert isinstance(round_trip.bio_model, ConstantRBEModel)
    assert round_trip.bio_model.rbe == 1.0

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

    matrad = pln.to_matrad()
    assert matrad["bioModel"] == "LSM"
    assert matrad["bioModelParameters"] == {"upper_let_threshold": 20.0}
    pln3 = create_pln(matrad)
    assert pln3.bio_model == pln.bio_model


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
    assert create_pln({"radiationMode": "carbon"}).bio_model.model == "alpha_beta_kernels"


@pytest.mark.parametrize(
    ("stored", "identifier"),
    [
        ("kernel_based_lq", "alpha_beta_kernels"),
        ("LEM", "alpha_beta_kernels"),
        ("MKM", "zstar_kernels"),
        ("zs_based_lq", "zstar_kernels"),
        ("dose_average_alpha_beta", "alpha_beta_tables"),
    ],
)
def test_plan_with_old_bio_model_identifier_loads(stored, identifier):
    """Plans and workspaces serialized before the formalism renaming keep loading."""
    dumped = IonPlan(radiation_mode="oxygen").model_dump()
    dumped["bio_model"] = {"model": stored}
    pln = create_pln(dumped)
    assert pln.bio_model.model == identifier
    assert pln.model_dump()["bio_model"]["model"] == identifier
    assert create_pln({"radiation_mode": "oxygen", "bio_model": stored}).bio_model.model == (
        identifier
    )


def test_plan_serializes_bio_model_data_selection():
    pln = IonPlan(
        radiation_mode="carbon",
        bio_model={"model": "alpha_beta_kernels", "data_model": "LEM_I", "kernel_set": "lem1"},
    )
    dumped = pln.model_dump()
    assert dumped["bio_model"] == {
        "model": "alpha_beta_kernels",
        "data_model": "LEM_I",
        "kernel_set": "lem1",
    }
    restored = create_pln(dumped)
    assert restored.bio_model == pln.bio_model
    assert restored.to_matrad()["bioModel"] == "LEM"
    assert restored.to_matrad()["bioModelParameters"] == {
        "data_model": "LEM_I",
        "kernel_set": "lem1",
    }

    let_plan = IonPlan(
        radiation_mode="protons", bio_model={"model": "WED", "quantity_variants": {"let": "LETd"}}
    )
    assert create_pln(let_plan.model_dump()).bio_model.quantity_variant("let") == "LETd"


def test_dij_with_old_bio_model_identifier_loads():
    from pyRadPlan.dij import Dij

    dij = Dij.model_validate(
        {
            "dose_grid": {"resolution": {"x": 1, "y": 1, "z": 1}, "dimensions": [2, 2, 2]},
            "ct_grid": {"resolution": {"x": 1, "y": 1, "z": 1}, "dimensions": [2, 2, 2]},
            "num_of_beams": 1,
            "bixel_num": [0],
            "ray_num": [0],
            "beam_num": [0],
            "bio_model": {"model": "kernel_based_lq"},
        }
    )
    assert dij.bio_model.model == "alpha_beta_kernels"


def test_plan_carries_a_loaded_machine():
    machine = load_from_name("protons", "Generic")
    plan = create_pln(radiation_mode="protons", machine=machine)
    assert plan.machine is machine
    # matRad's pln.machine names the base data file
    assert plan.to_matrad()["machine"] == "Generic"
    assert create_pln(radiation_mode="protons", machine={"name": "x"}).machine == {"name": "x"}


@pytest.mark.parametrize(
    ("radiation_mode", "plan_cls"), [("protons", IonPlan), ("photons", PhotonPlan)]
)
def test_plan_round_trip_keeps_the_concrete_machine(radiation_mode, plan_cls):
    """A dumped plan restores its machine as the loaded subclass, kernels included."""
    machine = load_from_name(radiation_mode, "Generic")
    plan = create_pln(radiation_mode=radiation_mode, machine=machine)
    dumped = plan.model_dump()
    assert dumped["machine"]["radiation_mode"] == radiation_mode
    assert "pb_kernels" in dumped["machine"]

    restored_plans = (
        create_pln(dumped),
        plan_cls.model_validate(dumped),
        create_pln(plan.model_dump(by_alias=True)),
    )
    for restored in restored_plans:
        assert type(restored.machine) is type(machine)
        assert restored.machine is not machine
        assert len(restored.machine.pb_kernels) == len(machine.pb_kernels)
        assert np.array_equal(restored.machine.energies, machine.energies)
        # unset fields stay unset, e.g. undeclared tissue classes stay derived from kernels
        assert restored.machine.model_fields_set == machine.model_fields_set
        assert restored.to_matrad()["machine"] == "Generic"


def test_plan_validates_a_dumped_machine_by_its_radiation_mode():
    """Machine dicts with a registered radiation mode are validated by that machine class."""
    dumped = load_from_name("protons", "Generic").model_dump()
    plan = create_pln(radiation_mode="protons", machine=dumped)
    assert isinstance(plan.machine, IonAccelerator)
    with pytest.raises(ValidationError):
        create_pln(radiation_mode="protons", machine={"radiation_mode": "protons", "name": "x"})


def _legacy_photon_machine():
    raw = load_machine_from_mat(find_machine_file("photons", "Generic"))
    raw["meta"]["version"] = 1
    return validate_machine(raw)


def test_legacy_photon_kernels_are_rescaled_exactly_once():
    """Version-1 kernels are rescaled on import, not again when their dump is validated."""
    current = load_from_name("photons", "Generic")
    machine = _legacy_photon_machine()
    expected = {e: 4.0 * k.kernel_data for e, k in current.pb_kernels.items()}
    for energy, kernel in machine.pb_kernels.items():
        assert np.array_equal(kernel.kernel_data, expected[energy])

    plan = create_pln(radiation_mode="photons", machine=machine)
    once = create_pln(plan.model_dump())
    restored = (
        once,
        create_pln(once.model_dump()),
        PhotonPlan.model_validate(plan.model_dump()),
        PhotonLINAC.model_validate(machine.model_dump()),
    )
    for pln in restored:
        candidate = pln if isinstance(pln, PhotonLINAC) else pln.machine
        for energy, kernel in candidate.pb_kernels.items():
            assert np.array_equal(kernel.kernel_data, expected[energy])
    for energy, kernel in machine.pb_kernels.items():
        assert np.array_equal(kernel.kernel_data, expected[energy])


@pytest.mark.parametrize(
    ("machine_cls", "radiation_mode"), [(PhotonLINAC, "photons"), (IonAccelerator, "protons")]
)
def test_plan_round_trip_keeps_a_default_radiation_mode(machine_cls, radiation_mode):
    """A machine relying on its class default radiation mode is still rebuilt as its class."""
    data = load_from_name(radiation_mode, "Generic").model_dump(exclude_unset=True)
    del data["radiation_mode"]
    machine = machine_cls.model_validate(data)
    assert "radiation_mode" not in machine.model_fields_set

    restored = create_pln(create_pln(radiation_mode=radiation_mode, machine=machine).model_dump())
    assert type(restored.machine) is machine_cls
    assert restored.machine.radiation_mode == radiation_mode
    assert restored.machine.model_fields_set - {"radiation_mode"} == machine.model_fields_set


def test_create_pln_kwargs_vhee():
    plan = create_pln(radiation_mode="VHEE")
    assert isinstance(plan, IonPlan)
    assert plan.radiation_mode == "VHEE"


def test_plan_invalid_mult_scen_raises_validation_error():
    with pytest.raises(ValidationError):
        IonPlan(mult_scen="does_not_exist")


def test_create_pln_dict_without_radiation_mode_uses_its_bio_default():
    plan = create_pln({"num_of_fractions": 5})
    assert plan.radiation_mode == "protons"
    assert plan.bio_model.model == IonPlan().bio_model.model == "constant_rbe"


def test_plan_to_matrad_prop_keys_are_camel_case(tmp_path):
    from pyRadPlan.io import load_data, save_data  # noqa: PLC0415

    pln = IonPlan(
        prop_stf={"gantry_angles": [0.0, 90.0], "bixel_width": 5.0},
        prop_opt={"quantity_opt": "physical_dose"},
        prop_dose_calc={"calc_let": True},
        prop_seq={"run_sequencing": False},
    )
    matrad = pln.to_matrad()
    assert matrad["propStf"] == {"gantryAngles": [0.0, 90.0], "bixelWidth": 5.0}
    assert matrad["propOpt"] == {"quantityOpt": "physical_dose"}
    assert matrad["propDoseCalc"] == {"calcLet": True}
    assert matrad["propSeq"] == {"runSequencing": False}
    assert pln.prop_stf == {"gantry_angles": [0.0, 90.0], "bixel_width": 5.0}

    assert create_pln(matrad).prop_stf == pln.prop_stf
    out = tmp_path / "pln.mat"
    save_data(pln=pln, file_name=str(out))
    reloaded = load_data(out)["pln"]
    assert np.array_equal(reloaded.prop_stf["gantry_angles"], [0.0, 90.0])
    assert reloaded.prop_opt == pln.prop_opt
