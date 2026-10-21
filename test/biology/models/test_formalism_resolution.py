"""Kernel set resolution of the data-driven formalisms (rules 1-5) and per-structure results."""

import numpy as np
import pytest

from pyRadPlan.biology.models import (
    MKM,
    BioMachineSummary,
    BioModelResolution,
    KernelBasedLQModel,
    KernelSetInfo,
    KernelSetSelectionError,
    StructureResolution,
    TissueClassInfo,
    Wedenberg,
    structure_tissues,
)
from pyRadPlan.machines import find_machine_file, load_machine_from_mat, validate_machine
from pyRadPlan.biology.tissues import (
    LEMTissueProperties,
    MKMTissueProperties,
    TissueProperties,
)

LEM_A01 = LEMTissueProperties(alpha_x=0.1, beta_x=0.05, d_t=30.0, r_nucleus=5.0)
LEM_A05 = LEMTissueProperties(alpha_x=0.5, beta_x=0.05, d_t=30.0, r_nucleus=5.0)
HSG = MKMTissueProperties(
    model="mMKM", alpha_x=0.313, beta_x=0.0615, r_nucleus=4.1, r_domain=0.34, alpha_0=0.17
)
V79 = MKMTissueProperties(
    model="mMKM", alpha_x=0.184, beta_x=0.02, r_nucleus=4.1, r_domain=0.26, alpha_0=0.13
)


def _set(set_id, kind, provenance, tissues=(), default=False, fields=None, variant=None):
    return KernelSetInfo(
        id=set_id,
        kind=kind,
        fields=fields or {"alpha_beta": ("alpha", "beta"), "zs": ("zs",), "let": ("let",)}[kind],
        variant=variant,
        provenance=provenance,
        default=default,
        declared=True,
        tissue_classes=tuple(
            TissueClassInfo(tissue=t, kernel_set=set_id, column=i) for i, t in enumerate(tissues)
        ),
    )


def _machine(*sets, mode="carbon", quantities=("physical_dose", "alpha", "beta", "zs", "let")):
    return BioMachineSummary(
        name="Synthetic",
        radiation_mode=mode,
        kernel_sets={s.id: s for s in sets},
        tissue_properties=tuple(t for s in sets for t in s.tissues),
        quantities=quantities,
    )


@pytest.fixture
def one_set():
    return _machine(_set("alpha_beta", "alpha_beta", "LEM_I", (LEM_A01, LEM_A05)))


@pytest.fixture
def two_sets():
    """α/β kernels of two provenances; LEM I is marked default."""
    return _machine(
        _set("lem1", "alpha_beta", "LEM_I", (LEM_A01, LEM_A05), default=True),
        _set("mmkm", "alpha_beta", "mMKM", (V79, HSG)),
    )


@pytest.fixture
def two_sets_no_default():
    return _machine(
        _set("lem1", "alpha_beta", "LEM_I", (LEM_A01, LEM_A05)),
        _set("mmkm", "alpha_beta", "mMKM", (V79, HSG)),
    )


# ----------------------------------------------------------------- rules 1 - 5
def test_unset_parameters_just_work_with_one_set_per_kind(one_set):
    assert KernelBasedLQModel().select_kernel_set(one_set).id == "alpha_beta"


def test_rule_1_explicit_kernel_set_wins(two_sets):
    model = KernelBasedLQModel(kernel_set="mmkm", data_model="LEM_I")
    assert model.select_kernel_set(two_sets, [LEM_A01]).id == "mmkm"


def test_rule_1_unknown_kernel_set_is_an_error(two_sets):
    with pytest.raises(KernelSetSelectionError, match="'lem1' \\(α/β kernels · LEM I\\)"):
        KernelBasedLQModel(kernel_set="nope").select_kernel_set(two_sets)


def test_rule_1_kernel_set_of_another_kind_is_an_error():
    machine = _machine(
        _set("alpha_beta", "alpha_beta", "LEM_I", (LEM_A01,)), _set("z", "zs", "MKM", (HSG,))
    )
    with pytest.raises(KernelSetSelectionError, match="no α/β kernels set of that id"):
        KernelBasedLQModel(kernel_set="z").select_kernel_set(machine)


@pytest.mark.parametrize(("data_model", "expected"), [("mMKM", "mmkm"), ("LEM", "lem1")])
def test_rule_2_data_model_selects_the_provenance(two_sets_no_default, data_model, expected):
    model = KernelBasedLQModel(data_model=data_model)
    assert model.select_kernel_set(two_sets_no_default).id == expected


def test_rule_2_unknown_data_model_lists_the_sets(two_sets):
    with pytest.raises(KernelSetSelectionError, match="data model 'LEM II'.*Available: 'lem1'"):
        KernelBasedLQModel(data_model="LEM_II").select_kernel_set(two_sets)


@pytest.mark.parametrize(
    ("tissues", "expected"), [([HSG, V79], "mmkm"), ([LEM_A05], "lem1"), ([LEM_A01], "lem1")]
)
def test_rule_3_the_set_the_tissues_resolve_to(two_sets_no_default, tissues, expected):
    assert KernelBasedLQModel().select_kernel_set(two_sets_no_default, tissues).id == expected


def test_rule_4_default_set_when_tissues_do_not_decide(two_sets):
    unmatched = TissueProperties(alpha_x=0.3, beta_x=0.03)
    assert KernelBasedLQModel().select_kernel_set(two_sets, [unmatched]).id == "lem1"
    assert KernelBasedLQModel().select_kernel_set(two_sets).id == "lem1"


def test_rule_5_ambiguity_names_the_candidates_and_the_choice(two_sets_no_default):
    with pytest.raises(KernelSetSelectionError) as info:
        KernelBasedLQModel().select_kernel_set(two_sets_no_default)
    message = str(info.value)
    assert "several α/β kernels sets" in message
    assert "'lem1' (α/β kernels · LEM I)" in message and "'mmkm' (α/β kernels · mMKM)" in message
    assert "kernel_set='lem1' or data_model='LEM_I'" in message


def test_rule_5_same_provenance_twice_needs_the_kernel_set():
    machine = _machine(
        _set("lem1_a", "alpha_beta", "LEM_I", (LEM_A01,)),
        _set("lem1_b", "alpha_beta", "LEM_I", (LEM_A01,)),
    )
    with pytest.raises(KernelSetSelectionError) as info:
        KernelBasedLQModel(data_model="LEM_I").select_kernel_set(machine, [LEM_A01])
    assert "kernel_set='lem1_a'" in str(info.value)
    assert "data_model=" not in str(info.value)
    assert KernelBasedLQModel(kernel_set="lem1_b").select_kernel_set(machine).id == "lem1_b"


def test_ambiguity_is_reported_as_model_unavailable(two_sets_no_default):
    (issue,) = KernelBasedLQModel().check_tissues(two_sets_no_default)
    assert issue.kind == "model_unavailable"
    assert "several α/β kernels sets" in issue.message


def test_zstar_quantity_variant_selects_the_z_definition():
    machine = _machine(
        _set("z1d", "zs", "MKM", (HSG,), variant="z1D", default=True),
        _set("zsat", "zs", "MKM", (HSG,), variant="z1D_saturation_corrected", fields=("zsat",)),
    )
    assert MKM().select_kernel_set(machine).id == "z1d"
    model = MKM(quantity_variants={"zs": "z1D_saturation_corrected"})
    assert model.select_kernel_set(machine).id == "zsat"


def test_let_variants():
    machine = _machine(
        _set("letd", "let", "undeclared", variant="LETd", default=True),
        _set("lett", "let", "undeclared", variant="LETt", fields=("LETt",)),
        mode="protons",
    )
    assert Wedenberg().select_kernel_set(machine).id == "letd"
    assert Wedenberg().check_tissues(machine) == []

    model = Wedenberg(quantity_variants={"let": "LETt"})
    assert model.select_kernel_set(machine).id == "lett"
    (issue,) = model.check_tissues(machine)
    assert issue.kind == "model_unavailable"
    assert "read the LET kernel field 'let' only" in issue.message

    with pytest.raises(KernelSetSelectionError, match="variant 'LETx'"):
        Wedenberg(quantity_variants={"let": "LETx"}).select_kernel_set(machine)


def test_let_model_without_let_kernels_relies_on_the_engine():
    """FRED scores LET itself, so a machine without LET kernels is no error per se."""
    machine = _machine(mode="protons", quantities=("physical_dose",))
    assert Wedenberg().select_kernel_set(machine) is None
    issues = Wedenberg().check_tissues(machine, provided_quantities=["physical_dose", "let"])
    assert issues == []


# ----------------------------------------------------------------- per structure
def test_resolution_maps_every_structure_to_a_kernel_set_and_column(two_sets):
    resolution = KernelBasedLQModel(data_model="mMKM").resolve(
        two_sets, {"PTV": HSG, "OAR": V79, "BODY": HSG}
    )
    assert isinstance(resolution, BioModelResolution)
    assert resolution.ok
    assert resolution.kernel_set.id == "mmkm"
    assert resolution.pairs() == {"PTV": ("mmkm", 1), "OAR": ("mmkm", 0), "BODY": ("mmkm", 1)}
    assert all(isinstance(entry, StructureResolution) for entry in resolution.structures)
    assert resolution.structures[0].tissue_class.tissue == HSG


def test_resolution_result_holds_two_provenances_per_structure(monkeypatch):
    """Mixed provenances fit the result shape; dose calculations still need one set."""
    lem = _set("lem1", "alpha_beta", "LEM_I", (LEM_A01,))
    mmkm = _set("mmkm", "alpha_beta", "mMKM", (HSG,))
    resolution = BioModelResolution(
        model="alpha_beta_kernels",
        kernel_set=lem,
        structures=(
            StructureResolution("PTV", LEM_A01, "lem1", 0, lem.tissue_classes[0]),
            StructureResolution("OAR", HSG, "mmkm", 0, mmkm.tissue_classes[0]),
        ),
        issues=(),
    )
    assert resolution.kernel_set_ids == ("lem1", "mmkm")
    assert resolution.pairs() == {"PTV": ("lem1", 0), "OAR": ("mmkm", 0)}

    model = KernelBasedLQModel()
    monkeypatch.setattr(model, "resolve", lambda *args, **kwargs: resolution)
    with pytest.raises(NotImplementedError, match="one kernel set per formalism"):
        model.evaluator(None, {"tissue_class": np.asarray([LEM_A01, HSG])})


@pytest.mark.parametrize(
    "structures",
    [
        {"PTV": LEM_A01},
        [("PTV", LEM_A01)],
        [LEM_A01],
    ],
)
def test_structure_forms(structures):
    ((name, tissue),) = structure_tissues(structures)
    assert tissue is LEM_A01
    assert name in ("PTV", "tissue class 1")


# ----------------------------------------------------------------- machines with data
@pytest.fixture(scope="module")
def machine_two_sets():
    """Generic carbon with a second α/β kernel set (mMKM) in fields of its own."""
    raw = load_machine_from_mat(find_machine_file("carbon", "Generic"))
    raw["data"]["alpha_mMKM"] = [0.9 * np.asarray(a) for a in raw["data"]["alpha"]]
    raw["data"]["beta_mMKM"] = list(raw["data"]["beta"])
    raw["meta"]["kernelSets"] = [
        {"id": "lem1", "kind": "alpha_beta", "model": "LEM_I", "default": True},
        {"id": "mmkm", "kind": "alpha_beta", "fields": "alpha_mMKM,beta_mMKM", "model": "mMKM"},
    ]
    raw["meta"]["tissueParameters"] = {
        "cell_Line": ["a01", "a05", "V79", "HSG"],
        "model": ["LEM_I", "LEM_I", "mMKM", "mMKM"],
        "alphaX": [0.1, 0.5, V79.alpha_x, HSG.alpha_x],
        "betaX": [0.05, 0.05, V79.beta_x, HSG.beta_x],
        "Dt": [30.0, 30.0, np.nan, np.nan],
        "rNucleus": [5.0, 5.0, 4.1, 4.1],
        "rDomain": [np.nan, np.nan, V79.r_domain, HSG.r_domain],
        "ref_Kernel": ["lem1", "lem1", "mmkm", "mmkm"],
        "column": [np.nan, np.nan, 1, 2],
    }
    return validate_machine(raw)


def _voxel_params(tissues):
    return {
        "alpha_x": np.asarray([[t.alpha_x] for t in tissues]),
        "beta_x": np.asarray([[t.beta_x] for t in tissues]),
        "tissue_class": np.asarray(tissues),
        "tissue_class_ix": np.arange(len(tissues))[:, None],
    }


def test_evaluator_reads_the_fields_of_the_resolved_set(machine_two_sets):
    evaluator = KernelBasedLQModel(data_model="mMKM").evaluator(
        machine_two_sets, _voxel_params([HSG, V79])
    )
    assert evaluator.field_map == {"alpha": "alpha_mMKM", "beta": "beta_mMKM"}
    assert evaluator.lookup.kernel_rows.tolist() == [1, 0]

    kernel = next(iter(machine_two_sets.pb_kernels.values())).to_namespace(np)
    quantities = evaluator.kernel_quantities(kernel)
    assert np.allclose(quantities["alpha"], kernel["alpha_mMKM"])


def test_evaluator_defaults_to_the_set_the_tissues_resolve_to(machine_two_sets):
    evaluator = KernelBasedLQModel().evaluator(machine_two_sets, _voxel_params([LEM_A05]))
    assert evaluator.field_map == {"alpha": "alpha", "beta": "beta"}
    assert evaluator.lookup.kernel_rows.tolist() == [1]

    evaluator = KernelBasedLQModel().evaluator(machine_two_sets, _voxel_params([V79]))
    assert evaluator.field_map["alpha"] == "alpha_mMKM"


def test_evaluator_rejects_a_tissue_of_another_provenance(machine_two_sets):
    with pytest.raises(ValueError, match="tabulated for LEM I tissue classes"):
        KernelBasedLQModel(kernel_set="lem1").evaluator(machine_two_sets, _voxel_params([HSG]))
