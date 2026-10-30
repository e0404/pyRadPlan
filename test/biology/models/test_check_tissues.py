"""BiologicalModel.check_tissues and bio_model_options against machines and summaries."""

import math
import re

import pytest

from pyRadPlan.biology.models import (
    ISSUE_KINDS,
    BioMachineSummary,
    BioModelOption,
    KernelBasedLQModel,
    KernelSetInfo,
    TissueClassInfo,
    TissueIssue,
    bio_model_options,
    create_bio_model,
)
from pyRadPlan.machines import (
    find_machine_file,
    load_from_name,
    load_machine_from_mat,
    validate_machine,
)
from pyRadPlan.biology.tissues import (
    LEMTissueProperties,
    MKMTissueProperties,
    TissueProperties,
)

LEM_A01 = LEMTissueProperties(alpha_x=0.1, beta_x=0.05, d_t=30.0, r_nucleus=5.0)
MKM_OXYGEN = MKMTissueProperties(
    model="mMKM", r_nucleus=3.9, r_domain=0.32, alpha_0=0.172, beta_0=0.0615
)


@pytest.fixture(scope="module")
def summaries():
    """Summaries of the bundled ion machines, read once."""
    return {
        mode: BioMachineSummary.from_machine(load_from_name(mode, "Generic"))
        for mode in ("protons", "helium", "carbon", "oxygen")
    }


@pytest.fixture(scope="module")
def undeclared_carbon():
    """The generic carbon machine without its base-data metadata, as legacy data reads."""
    raw = load_machine_from_mat(find_machine_file("carbon", "Generic"))
    meta = {k: v for k, v in raw["meta"].items() if k not in ("kernelSets", "tissueParameters")}
    return BioMachineSummary.from_machine(validate_machine({"meta": meta, "data": raw["data"]}))


def _kinds(issues):
    return [issue.kind for issue in issues]


def test_matching_tissues_have_no_issues(summaries):
    assert (
        create_bio_model("alpha_beta_kernels").check_tissues(
            summaries["oxygen"], {"PTV": LEM_A01, "BODY": TissueProperties(alpha_x=0.5)}
        )
        == []
    )
    assert (
        create_bio_model("zstar_kernels").check_tissues(summaries["oxygen"], {"PTV": MKM_OXYGEN})
        == []
    )
    assert create_bio_model("WED").check_tissues(summaries["protons"], {"PTV": LEM_A01}) == []


@pytest.mark.parametrize(
    ("mode", "model", "message"),
    [
        ("carbon", "zstar_kernels", "z\\* kernels, but the machine carries none|zs"),
        ("protons", "alpha_beta_kernels", "Required quantities.*alpha"),
        ("carbon", "alpha_beta_tables", "fluence"),
    ],
)
def test_model_unavailable(summaries, mode, model, message):
    issues = create_bio_model(model).check_tissues(summaries[mode], {"PTV": LEM_A01})
    assert _kinds(issues) == ["model_unavailable"]
    assert issues[0].structure is None
    assert issues[0].is_error
    assert re.search(message, issues[0].message)


def test_radiation_mode_mismatch_is_model_unavailable(summaries):
    (issue,) = create_bio_model("WED").check_tissues(summaries["carbon"])
    assert issue.kind == "model_unavailable"
    assert "Radiation mode 'carbon' is invalid" in issue.message


def test_inconsistent_base_data(summaries):
    broken = KernelSetInfo(
        id="alpha_beta",
        kind="alpha_beta",
        fields=("alpha", "beta"),
        problems=("α/β kernels require identical tissue classes in the same order",),
    )
    summary = BioMachineSummary(
        radiation_mode="carbon",
        kernel_sets={"alpha_beta": broken},
        quantities=("physical_dose", "alpha", "beta"),
    )
    issues = KernelBasedLQModel().check_tissues(summary, {"PTV": LEM_A01})
    assert _kinds(issues) == ["base_data"]
    assert issues[0].kernel_set == "alpha_beta"


def test_type_mismatch_suggests_a_class(summaries):
    (issue,) = create_bio_model("alpha_beta_kernels").check_tissues(
        summaries["oxygen"], {"PTV": MKM_OXYGEN}
    )
    assert issue.kind == "type_mismatch"
    assert issue.structure == "PTV"
    assert issue.tissue is MKM_OXYGEN
    assert "MKM (MKMTissueProperties)" in issue.message
    assert "α/β kernels · LEM I" in issue.message


def test_lem_version_mismatch_is_a_warning(summaries):
    lem2 = LEM_A01.model_copy(update={"version": "II"})
    model = create_bio_model("alpha_beta_kernels")
    (issue,) = model.check_tissues(summaries["oxygen"], {"PTV": lem2})
    assert issue.kind == "provenance_differs"
    assert issue.severity == "warning" and issue.is_warning and not issue.is_error
    assert issue.fix_provenance == "LEM_I"
    assert "labelled LEM II; the base data is LEM I" in issue.message
    resolution = model.resolve(summaries["oxygen"], {"PTV": lem2})
    assert resolution.ok
    assert [warning.kind for warning in resolution.warnings] == ["provenance_differs"]
    assert resolution.structures[0].resolved
    assert resolution.structures[0].column == 0


def test_lem_tissue_for_mkm_kernels_is_an_error(summaries):
    (issue,) = create_bio_model("zstar_kernels").check_tissues(
        summaries["oxygen"], {"PTV": LEM_A01}
    )
    assert issue.kind == "type_mismatch"
    assert issue.is_error
    assert issue.fix_provenance is None


def test_issue_severity_is_validated():
    assert TissueIssue("base_data", "x").severity == "error"
    with pytest.raises(ValueError, match="severity"):
        TissueIssue("base_data", "x", severity="info")


def _zs_set(set_id, *classes, provenance="undeclared", default=False):
    return KernelSetInfo(
        id=set_id,
        kind="zs",
        fields=("zs",),
        provenance=provenance,
        default=default,
        declared=True,
        tissue_classes=tuple(
            TissueClassInfo(tissue=tissue, kernel_set=set_id, column=column)
            for column, tissue in enumerate(classes)
        ),
    )


def _zs_summary(*sets):
    return BioMachineSummary(
        radiation_mode="oxygen",
        kernel_sets={s.id: s for s in sets},
        quantities=("physical_dose", "zs"),
    )


def test_a_matching_class_of_the_same_provenance_is_preferred():
    geometry = dict(r_nucleus=3.9, r_domain=0.32)
    mkm_class = MKMTissueProperties.from_base_data(model="MKM", cell_line="A", **geometry)
    mmkm_class = MKMTissueProperties.from_base_data(model="mMKM", cell_line="B", **geometry)
    summary = _zs_summary(_zs_set("zs", mmkm_class, mkm_class))
    model = create_bio_model("zstar_kernels")
    tissues = {
        "PTV": MKMTissueProperties(model="MKM", alpha_0=0.1, beta_0=0.05, **geometry),
        "OAR": MKMTissueProperties(model="mMKM", alpha_0=0.1, beta_0=0.05, **geometry),
    }
    resolution = model.resolve(summary, tissues)
    assert resolution.issues == ()
    assert [entry.column for entry in resolution.structures] == [1, 0]


def test_kernel_set_selection_prefers_the_tissues_provenance():
    geometry = dict(r_nucleus=3.9, r_domain=0.32)
    summary = _zs_summary(
        _zs_set(
            "zs_mmkm",
            MKMTissueProperties.from_base_data(model="mMKM", **geometry),
            provenance="mMKM",
        ),
        _zs_set(
            "zs_mkm", MKMTissueProperties.from_base_data(model="MKM", **geometry), provenance="MKM"
        ),
    )
    model = create_bio_model("zstar_kernels")
    for label, expected in (("MKM", "zs_mkm"), ("mMKM", "zs_mmkm")):
        tissue = MKMTissueProperties(model=label, alpha_0=0.1, beta_0=0.05, **geometry)
        resolution = model.resolve(summary, {"PTV": tissue})
        assert resolution.kernel_set.id == expected
        assert resolution.issues == ()


def test_no_matching_class_with_closest_suggestion(summaries):
    tissue = LEM_A01.model_copy(update={"alpha_x": 0.45})
    issues = create_bio_model("alpha_beta_kernels").check_tissues(
        summaries["oxygen"], {"PTV": tissue, "BODY": LEM_A01}
    )
    (issue,) = issues
    assert issue.kind == "no_matching_class"
    assert issue.structure == "PTV"
    assert "No matching tissue class" in issue.message
    assert issue.suggestion.alpha_x == 0.5
    assert issue.suggestion.cell_line == "Generic_LEMax05"


LEM_IV_A01 = LEMTissueProperties(alpha_x=0.1, beta_x=0.05, version="IV", d_t=30.0, r_nucleus=5.0)


def test_parameters_the_base_data_leaves_undeclared_are_not_compared(summaries):
    """The carbon LEM IV classes do not document Dt / rNucleus (NaN)."""
    model = create_bio_model("alpha_beta_kernels")
    resolution = model.resolve(summaries["carbon"], {"PTV": LEM_IV_A01, "BODY": LEM_IV_A01})
    assert resolution.ok
    assert resolution.pairs()["PTV"] == ("alpha_beta", 0)
    undeclared = LEM_IV_A01.model_copy(update={"d_t": math.nan, "r_nucleus": math.nan})
    assert model.check_tissues(summaries["carbon"], {"PTV": undeclared}) == []

    # The declared reference parameters still decide, and the suggestion agrees
    (issue,) = model.check_tissues(
        summaries["carbon"], {"PTV": LEM_IV_A01.model_copy(update={"alpha_x": 0.45})}
    )
    assert issue.kind == "no_matching_class"
    assert issue.suggestion.cell_line == "Generic_LEMax05"
    assert LEM_IV_A01.model_copy(update={"alpha_x": 0.5}).matches(
        issue.suggestion, ignore_undeclared=True
    )


def test_tissue_lookup_ignores_undeclared_class_parameters(summaries):
    from pyRadPlan.biology.models import make_tissue_lookup

    classes = list(summaries["carbon"].tissue_properties)
    tissue = LEM_IV_A01.model_copy(update={"alpha_x": 0.5})
    lookup = make_tissue_lookup("exact", classes, [tissue], LEMTissueProperties)
    assert list(lookup.kernel_rows) == [1]


def test_undeclared_provenance_matches_on_reference_parameters(undeclared_carbon):
    model = create_bio_model("alpha_beta_kernels")
    assert model.check_tissues(undeclared_carbon, {"PTV": LEM_A01}) == []
    (issue,) = model.check_tissues(undeclared_carbon, {"PTV": TissueProperties(alpha_x=0.2)})
    assert issue.kind == "no_matching_class"
    assert issue.suggestion.alpha_x == 0.1


def test_missing_runtime_parameter(summaries):
    tissue = MKM_OXYGEN.model_copy(update={"alpha_0": math.nan})
    (issue,) = create_bio_model("zstar_kernels").check_tissues(
        summaries["oxygen"], {"PTV": tissue}
    )
    assert issue.kind == "missing_parameter"
    assert "lacks alpha_0" in issue.message


@pytest.mark.parametrize("value", [-0.1, math.inf])
def test_invalid_reference(summaries, value):
    tissue = TissueProperties(alpha_x=value, beta_x=0.05)
    issues = create_bio_model("constant_rbe").check_tissues(summaries["protons"], {"PTV": tissue})
    assert _kinds(issues) == ["invalid_reference"]
    assert "alpha_x must be finite and non-negative" in issues[0].message


def test_unset_reference_is_defaulted_but_a_constructed_nan_is_invalid(summaries):
    model = create_bio_model("constant_rbe")
    defaulted = TissueProperties(alpha_x=math.nan, beta_x=0.05)
    assert defaulted.alpha_x == 0.1
    assert model.check_tissues(summaries["protons"], {"PTV": defaulted}) == []
    # only bypassing validation leaves a structure's reference parameter NaN
    unset = TissueProperties.model_construct(alpha_x=math.nan, beta_x=0.05)
    issues = model.check_tissues(summaries["protons"], {"PTV": unset})
    assert _kinds(issues) == ["invalid_reference"]


def test_reference_outside_the_model_domain(summaries):
    tissue = TissueProperties(alpha_x=0.1, beta_x=0.0)
    (issue,) = create_bio_model("MCN").check_tissues(summaries["protons"], {"PTV": tissue})
    assert issue.kind == "reference_domain"
    assert "only defined for beta_x > 0" in issue.message
    # outside every structure there are no reference parameters, which is fine
    assert (
        create_bio_model("MCN").check_tissues(
            summaries["protons"], {"BODY": TissueProperties(alpha_x=0.0, beta_x=0.0)}
        )
        == []
    )


def test_every_structure_must_resolve(summaries):
    issues = create_bio_model("alpha_beta_kernels").check_tissues(
        summaries["oxygen"],
        {
            "PTV": LEM_A01,
            "OAR": MKM_OXYGEN,
            "BODY": TissueProperties(alpha_x=0.3, beta_x=0.1),
            "SKIN": TissueProperties(alpha_x=-0.1),
        },
    )
    assert [(i.structure, i.kind) for i in issues] == [
        ("OAR", "type_mismatch"),
        ("BODY", "no_matching_class"),
        ("SKIN", "invalid_reference"),
    ]
    assert all(isinstance(issue, TissueIssue) and issue.kind in ISSUE_KINDS for issue in issues)
    assert str(issues[0]).startswith("OAR: ")


def test_without_a_machine_only_base_data_free_checks_run():
    tissue = MKMTissueProperties(r_domain=0.3)
    issues = create_bio_model("zstar_kernels").check_tissues(None, {"PTV": tissue})
    assert _kinds(issues) == ["missing_parameter"]


def test_machine_and_summary_give_the_same_issues(summaries):
    machine = load_from_name("oxygen", "Generic")
    structures = {"PTV": MKM_OXYGEN, "BODY": TissueProperties(alpha_x=0.2)}
    model = create_bio_model("alpha_beta_kernels")
    assert [str(i) for i in model.check_tissues(machine, structures)] == [
        str(i) for i in model.check_tissues(summaries["oxygen"], structures)
    ]


def test_check_tissues_accepts_structures_with_vois(summaries):
    """Anything with ``vois`` of named tissues, like a StructureSet."""

    class _Voi:
        def __init__(self, name, tissue):
            self.name, self.tissue_properties = name, tissue

    class _Cst:
        vois = [_Voi("PTV", LEM_A01), _Voi("OAR", MKM_OXYGEN)]

    issues = create_bio_model("alpha_beta_kernels").check_tissues(summaries["oxygen"], _Cst())
    assert [i.structure for i in issues] == ["OAR"]


def test_summary_is_cacheable_and_carries_no_kernels(summaries):
    summary = summaries["oxygen"]
    assert summary.name == "Generic" and summary.radiation_mode == "oxygen"
    assert set(summary.kernel_sets) == {"alpha_beta", "zs"}
    assert summary.provided_quantities() == ["physical_dose", "alpha", "beta", "zs"]
    assert not summary.has_fluence_spectrum
    assert not hasattr(summary, "pb_kernels")
    assert BioMachineSummary.from_machine(summary) is summary
    assert len(summary.tissue_properties) == 3


# ----------------------------------------------------------------- options
@pytest.mark.parametrize(
    ("mode", "available", "unavailable"),
    [
        (
            "carbon",
            ["None (physical dose)", "α/β kernels · LEM IV", "Constant RBE"],
            ["Linear scaling (LET)", "α/β tables + fluence spectra · LEM I", "z* kernels"],
        ),
        (
            "oxygen",
            ["None (physical dose)", "α/β kernels · LEM I", "Constant RBE", "z* kernels · mMKM"],
            ["α/β tables + fluence spectra · LEM I"],
        ),
        (
            "protons",
            [
                "None (physical dose)",
                "Carabe (LET)",
                "Linear scaling (LET)",
                "McNamara (LET)",
                "Wedenberg (LET)",
                "Constant RBE",
            ],
            ["α/β kernels", "α/β tables + fluence spectra · LEM I", "z* kernels"],
        ),
    ],
)
def test_options_for_bundled_machines(summaries, mode, available, unavailable):
    options = bio_model_options(mode, summaries[mode])
    assert all(isinstance(option, BioModelOption) for option in options)
    assert [o.label for o in options if o.available] == available
    assert [o.label for o in options if not o.available] == unavailable
    assert all(o.reason for o in options if not o.available)
    assert options[0].model == "none"


def test_options_of_a_machine_and_its_summary_agree(summaries):
    machine = load_from_name("oxygen", "Generic")
    labels = [(o.label, o.available) for o in bio_model_options("oxygen", machine)]
    assert labels == [
        (o.label, o.available) for o in bio_model_options("oxygen", summaries["oxygen"])
    ]


def test_options_carry_the_tissue_classes_of_their_base_data(summaries):
    options = {o.label: o for o in bio_model_options("oxygen", summaries["oxygen"])}
    lem = options["α/β kernels · LEM I"]
    assert (lem.model, dict(lem.parameters), lem.provenance, lem.kernel_set) == (
        "alpha_beta_kernels",
        {},
        "LEM_I",
        "alpha_beta",
    )
    assert [t.cell_line for t in lem.tissue_classes] == ["Generic_LEMax01", "Generic_LEMax05"]
    assert [t.cell_line for t in options["z* kernels · mMKM"].tissue_classes] == ["Generic_MKM"]
    assert lem.spec == {"model": "alpha_beta_kernels"}
    assert lem.create() == KernelBasedLQModel()
    assert lem.description


def test_options_without_machine_are_the_plain_formalisms():
    options = bio_model_options("oxygen")
    assert [o.label for o in options] == [
        "None (physical dose)",
        "α/β kernels",
        "α/β tables + fluence spectra",
        "Constant RBE",
        "z* kernels",
    ]
    assert all(o.available for o in options)


def test_options_select_among_several_sets_by_provenance_or_id():
    def ab(set_id, provenance, default=False):
        return KernelSetInfo(
            id=set_id,
            kind="alpha_beta",
            fields=("alpha", "beta"),
            provenance=provenance,
            default=default,
            tissue_classes=(TissueClassInfo(LEM_A01, set_id, 0),),
        )

    summary = BioMachineSummary(
        radiation_mode="carbon",
        kernel_sets={
            "lem1": ab("lem1", "LEM_I", default=True),
            "mmkm": ab("mmkm", "mMKM"),
            "mmkm2": ab("mmkm2", "mMKM"),
        },
        quantities=("physical_dose", "alpha", "beta"),
    )
    options = [o for o in bio_model_options("carbon", summary) if o.model == "alpha_beta_kernels"]
    assert [(o.label, dict(o.parameters)) for o in options] == [
        ("α/β kernels · LEM I", {"data_model": "LEM_I"}),
        ("α/β kernels · mMKM", {"kernel_set": "mmkm"}),
        ("α/β kernels · mMKM", {"kernel_set": "mmkm2"}),
    ]
    assert all(o.available for o in options)
