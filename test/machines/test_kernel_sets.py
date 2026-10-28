"""Kernel set metadata (machine.meta.kernelSets / tissueParameters) and legacy inference."""

import numpy as np
import pymatreader
import pytest
import scipy.io

from pyRadPlan.biology.models import KernelSetInfo, parse_kernel_set_declarations
from pyRadPlan.machines import find_machine_file, load_from_name, load_machine_from_mat
from pyRadPlan.machines import validate_machine
from pyRadPlan.biology.tissues import (
    LEMTissueProperties,
    MKMTissueProperties,
    TissueProperties,
)


def _mat_round_trip(tmp_path, name, value):
    """Write a struct (array) to a .mat file and read it back as read_mat delivers it."""
    path = tmp_path / f"{name}.mat"
    scipy.io.savemat(path, {name: value})
    return pymatreader.read_mat(str(path))[name]


def _struct_array(rows):
    """A MATLAB struct array (one element per row) for scipy.io.savemat."""
    keys = list(rows[0])
    dtype = [(key, object) for key in keys]
    array = np.empty((1, len(rows)), dtype=dtype)
    for i, row in enumerate(rows):
        for key in keys:
            array[0, i][key] = row[key]
    return array


@pytest.fixture(scope="module")
def raw_carbon():
    return load_machine_from_mat(find_machine_file("carbon", "Generic"))


@pytest.fixture(scope="module")
def raw_oxygen():
    return load_machine_from_mat(find_machine_file("oxygen", "Generic"))


@pytest.fixture
def carbon(raw_carbon):
    """A fresh copy of the raw generic carbon machine dict, without its base-data metadata."""
    return {
        "meta": {
            key: value
            for key, value in raw_carbon["meta"].items()
            if key not in ("kernelSets", "tissueParameters")
        },
        "data": {key: list(value) for key, value in raw_carbon["data"].items()},
    }


def _two_alpha_beta_sets(tmp_path, carbon, default_lem=True):
    """Carbon machine with α/β kernels from LEM I (alpha/beta) and mMKM (alpha_mMKM/...)."""
    carbon["data"]["alpha_mMKM"] = [0.9 * np.asarray(a) for a in carbon["data"]["alpha"]]
    carbon["data"]["beta_mMKM"] = [np.asarray(b) for b in carbon["data"]["beta"]]
    carbon["meta"]["kernelSets"] = _mat_round_trip(
        tmp_path,
        "kernelSets",
        _struct_array(
            [
                {
                    "id": "lem1",
                    "kind": "alpha_beta",
                    "fields": np.asarray(["alpha", "beta"], dtype=object),
                    "variant": "",
                    "model": "LEM",
                    "version": "LEM_I",
                    "reference": "Scholz 1997",
                    "default": float(default_lem),
                },
                {
                    "id": "mmkm",
                    "kind": "alpha_beta",
                    "fields": "alpha_mMKM, beta_mMKM",
                    "variant": "",
                    "model": "mMKM",
                    "version": "",
                    "reference": "Inaniwa 2010",
                    "default": 0.0,
                },
            ]
        ),
    )
    carbon["meta"]["tissueParameters"] = _mat_round_trip(
        tmp_path,
        "tissueParameters",
        {
            "cell_Line": np.asarray(["LEM_a01", "LEM_a05", "HSG_mMKM", "V79_mMKM"], dtype=object),
            "model": np.asarray(["LEM", "LEM_I", "mMKM", "mMKM"], dtype=object),
            "alphaX": np.asarray([0.1, 0.5, 0.313, 0.184]),
            "betaX": np.asarray([0.05, 0.05, 0.0615, 0.02]),
            "Dt": np.asarray([30.0, 30.0, np.nan, np.nan]),
            "rNucleus": np.asarray([5.0, 5.0, 4.1, 4.1]),
            "rDomain": np.asarray([np.nan, np.nan, 0.34, 0.26]),
            "alpha0": np.asarray([np.nan, np.nan, 0.172, 0.13]),
            "beta0": np.asarray([np.nan, np.nan, 0.0615, 0.02]),
            "ref_Kernel": np.asarray(["lem1", "lem1", "mmkm", "mmkm"], dtype=object),
            "column": np.asarray([np.nan, np.nan, 2.0, 1.0]),
        },
    )
    return validate_machine(carbon)


# ----------------------------------------------------------------- legacy inference
def test_carbon_legacy_data_has_one_undeclared_alpha_beta_set(carbon):
    machine = validate_machine(carbon)
    (kernel_set,) = machine.kernel_sets.values()

    assert isinstance(kernel_set, KernelSetInfo)
    assert (kernel_set.id, kernel_set.kind, kernel_set.fields) == (
        "alpha_beta",
        "alpha_beta",
        ("alpha", "beta"),
    )
    assert kernel_set.provenance == "undeclared"
    assert kernel_set.label == "α/β kernels · model not declared"
    assert not kernel_set.declared and not kernel_set.problems
    assert [(c.tissue.alpha_x, c.column) for c in kernel_set.tissue_classes] == [
        (0.1, 0),
        (0.5, 1),
    ]


def test_oxygen_legacy_sets_take_the_provenance_their_tissue_classes_declare(raw_oxygen):
    meta = {k: v for k, v in raw_oxygen["meta"].items() if k != "kernelSets"}
    data = {key: list(value) for key, value in raw_oxygen["data"].items()}
    machine = validate_machine({"meta": meta, "data": data})

    assert sorted(machine.kernel_sets) == ["alpha_beta", "zs"]
    assert not any(s.declared for s in machine.kernel_sets.values())
    alpha_beta, zs = machine.kernel_sets["alpha_beta"], machine.kernel_sets["zs"]
    assert alpha_beta.label == "α/β kernels · LEM I"
    assert [(c.name, c.column) for c in alpha_beta.tissue_classes] == [
        ("Generic_LEMax01", 0),
        ("Generic_LEMax05", 1),
    ]
    assert zs.label == "z* kernels · mMKM"
    assert [c.name for c in zs.tissue_classes] == ["Generic_MKM"]
    assert machine.provided_quantities() == ["physical_dose", "alpha", "beta", "zs"]


@pytest.mark.parametrize("mode", ["protons", "helium"])
def test_let_machines_have_one_tissue_independent_let_set(mode):
    machine = load_from_name(mode, "Generic")
    (kernel_set,) = machine.kernel_sets.values()
    assert (kernel_set.kind, kernel_set.label, kernel_set.tissue_classes) == ("let", "LET", ())
    assert not kernel_set.tissue_dependent


def test_photon_machine_has_no_kernel_sets():
    assert load_from_name("photons", "Generic").kernel_sets == {}


# ----------------------------------------------------------------- declared metadata
def test_two_alpha_beta_sets_are_parsed_with_their_tissue_classes(tmp_path, carbon):
    machine = _two_alpha_beta_sets(tmp_path, carbon)
    sets = machine.kernel_sets

    assert list(sets) == ["lem1", "mmkm"]
    lem, mmkm = sets["lem1"], sets["mmkm"]
    assert (lem.provenance, lem.fields, lem.reference, lem.default, lem.declared) == (
        "LEM_I",
        ("alpha", "beta"),
        "Scholz 1997",
        True,
        True,
    )
    assert (mmkm.provenance, mmkm.fields, mmkm.default) == (
        "mMKM",
        ("alpha_mMKM", "beta_mMKM"),
        False,
    )
    assert mmkm.label == "α/β kernels · mMKM"
    assert not lem.problems and not mmkm.problems

    # LEM classes are matched onto the columns by alpha_x / beta_x, mMKM ones by `column`
    assert [(c.name, c.column) for c in lem.tissue_classes] == [("LEM_a01", 0), ("LEM_a05", 1)]
    assert [(c.name, c.column) for c in mmkm.tissue_classes] == [
        ("HSG_mMKM", 1),
        ("V79_mMKM", 0),
    ]
    assert all(isinstance(c.tissue, LEMTissueProperties) for c in lem.tissue_classes)
    assert all(isinstance(c.tissue, MKMTissueProperties) for c in mmkm.tissue_classes)
    assert {c.tissue.model for c in mmkm.tissue_classes} == {"mMKM"}

    # the second set's data is kept with the kernels, as (n_columns, n_depths)
    kernel = next(iter(machine.pb_kernels.values()))
    assert kernel.additional_data["alpha_mMKM"].shape == kernel.alpha.shape
    assert np.allclose(kernel.additional_data["alpha_mMKM"], 0.9 * kernel.alpha)
    assert "beta_mMKM" in kernel.to_namespace(np)
    assert machine.provided_quantities() == ["physical_dose", "alpha", "beta"]


def test_single_kernel_set_struct_is_squeezed_by_read_mat(tmp_path):
    raw = _mat_round_trip(
        tmp_path,
        "kernelSets",
        {"id": "z", "kind": "zs", "variant": "z1D", "model": "MKM", "default": True},
    )
    assert isinstance(raw["id"], str)
    (declaration,) = parse_kernel_set_declarations(raw)
    assert declaration == {
        "id": "z",
        "kind": "zs",
        "fields": ("zs",),
        "variant": "z1D",
        "model": "MKM",
        "reference": "",
        "default": True,
    }


def test_single_set_with_a_field_cell_array(tmp_path):
    raw = _mat_round_trip(
        tmp_path,
        "kernelSets",
        {"kind": "alpha_beta", "fields": np.asarray(["alpha", "beta"], dtype=object)},
    )
    (declaration,) = parse_kernel_set_declarations(raw)
    assert (declaration["id"], declaration["fields"]) == ("alpha_beta", ("alpha", "beta"))
    assert declaration["model"] == "undeclared"


def test_kernel_set_declarations_accept_a_list_of_structs_and_mat_field_names():
    declarations = parse_kernel_set_declarations(
        [
            {"kind": "let", "fields": "LET", "variant": "LETd", "default": 1},
            {"kind": "let", "fields": "LETt", "variant": "LETt"},
        ]
    )
    assert [(d["id"], d["fields"], d["default"]) for d in declarations] == [
        ("let_1", ("let",), True),
        ("let_2", ("LETt",), False),
    ]


def test_kernel_set_ids_must_be_unique():
    with pytest.raises(ValueError, match="must be unique"):
        parse_kernel_set_declarations([{"id": "a", "kind": "zs"}, {"id": "a", "kind": "zs"}])


def test_kernel_set_needs_a_kind():
    with pytest.raises(ValueError, match="declares no kind"):
        parse_kernel_set_declarations([{"id": "a"}])


def test_let_variants_are_reported_as_provided_quantities(tmp_path):
    raw = load_machine_from_mat(find_machine_file("protons", "Generic"))
    raw["data"]["LETt"] = [0.8 * np.asarray(let) for let in raw["data"]["LET"]]
    raw["meta"]["kernelSets"] = _mat_round_trip(
        tmp_path,
        "kernelSets",
        _struct_array(
            [
                {"id": "letd", "kind": "let", "fields": "LET", "variant": "LETd", "default": 1.0},
                {"id": "lett", "kind": "let", "fields": "LETt", "variant": "LETt", "default": 0.0},
            ]
        ),
    )
    machine = validate_machine(raw)

    assert [s.label for s in machine.kernel_sets.values()] == ["LET · LETd", "LET · LETt"]
    assert machine.provided_quantities() == ["physical_dose", "let", "let:LETd", "let:LETt"]


def test_declared_set_with_missing_kernel_fields_is_incomplete(carbon):
    carbon["meta"]["kernelSets"] = {"id": "z", "kind": "zs", "model": "MKM"}
    machine = validate_machine(carbon)

    kernel_set = machine.kernel_sets["z"]
    assert not kernel_set.complete
    assert any("do not carry" in problem for problem in kernel_set.problems)
    assert "zs" not in machine.provided_quantities()


def test_tissue_class_column_outside_the_kernels_is_a_problem(tmp_path, carbon):
    carbon["meta"]["tissueParameters"] = {
        "alphaX": [0.1, 0.5],
        "betaX": [0.05, 0.05],
        "model": ["LEM", "LEM"],
        "ref_Kernel": ["alpha_beta", "alpha_beta"],
        "column": [1.0, 3.0],
    }
    machine = validate_machine(carbon)
    (kernel_set,) = machine.kernel_sets.values()
    assert any("reference columns [3]" in problem for problem in kernel_set.problems)


def test_tissue_class_referencing_an_unknown_set_belongs_to_none(carbon, caplog):
    carbon["meta"]["tissueParameters"] = {
        "alphaX": [0.1, 0.5, 0.2],
        "betaX": [0.05, 0.05, 0.05],
        "ref_Kernel": ["alpha_beta", "alpha_beta", "elsewhere"],
    }
    machine = validate_machine(carbon)
    (kernel_set,) = machine.kernel_sets.values()
    assert [c.tissue.alpha_x for c in kernel_set.tissue_classes] == [0.1, 0.5]
    assert "unknown kernel set 'elsewhere'" in caplog.text


def test_lem_version_given_separately_selects_the_tissue_class(carbon):
    carbon["meta"]["tissueParameters"] = {
        "alphaX": [0.1, 0.5],
        "betaX": [0.05, 0.05],
        "model": ["LEM", "LEM"],
        "version": ["LEM_II", "II"],
    }
    machine = validate_machine(carbon)
    assert [t.provenance for t in machine.tissue_properties] == ["LEM_II", "LEM_II"]
    assert machine.kernel_sets["alpha_beta"].provenance == "LEM_II"


def test_machine_with_plain_tissue_classes_keeps_matching_by_reference_parameters(carbon):
    carbon["meta"]["tissueParameters"] = {"alphaX": [0.5, 0.1], "betaX": [0.05, 0.05]}
    machine = validate_machine(carbon)
    (kernel_set,) = machine.kernel_sets.values()
    assert [(c.tissue.alpha_x, c.column) for c in kernel_set.tissue_classes] == [
        (0.5, 1),
        (0.1, 0),
    ]
    assert all(type(c.tissue) is TissueProperties for c in kernel_set.tissue_classes)
