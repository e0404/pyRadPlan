"""Metadata of the base data shipped with pyRadPlan (machines and RBE tables)."""

import pytest

from pyRadPlan.biology.models import BioMachineSummary, bio_model_options
from pyRadPlan.biology.models import TabulatedAlphaBetaModel
from pyRadPlan.machines import (
    find_machine_file,
    get_available_machines,
    load_from_name,
    load_machine_from_mat,
    validate_machine,
)
from pyRadPlan.biology.tissues import (
    LEMTissueProperties,
    MKMTissueProperties,
    TissueProperties,
)


@pytest.fixture(scope="module")
def oxygen():
    return load_from_name("oxygen", "Generic")


@pytest.fixture(scope="module")
def oxygen_summary(oxygen):
    return BioMachineSummary.from_machine(oxygen)


@pytest.fixture(scope="module")
def carbon():
    return load_from_name("carbon", "Generic")


@pytest.fixture(scope="module")
def carbon_summary(carbon):
    return BioMachineSummary.from_machine(carbon)


def _without_metadata(radiation_mode, *keys):
    """The bundled machine with the given ``meta`` entries removed, as legacy data reads."""
    raw = load_machine_from_mat(find_machine_file(radiation_mode, "Generic"))
    meta = {k: v for k, v in raw["meta"].items() if k not in keys}
    return validate_machine({"meta": meta, "data": raw["data"]})


def _labels(radiation_mode, summary):
    return {
        (o.model, o.label, o.available)
        for o in bio_model_options(radiation_mode, summary)
        if o.kernel_set is not None
    }


def _model(radiation_mode, summary, model):
    """The formalism as the machine's option for it configures it."""
    return next(o for o in bio_model_options(radiation_mode, summary) if o.model == model).create()


def test_bundled_kernel_sets_are_consistent():
    for radiation_mode, names in get_available_machines().items():
        for name in names:
            summary = BioMachineSummary.from_machine(load_from_name(radiation_mode, name))
            for kernel_set in summary.kernel_sets.values():
                assert not kernel_set.problems, (radiation_mode, name, kernel_set.problems)


def test_oxygen_declares_its_kernel_sets(oxygen):
    sets = oxygen.kernel_sets
    assert list(sets) == ["alpha_beta", "zs"]
    alpha_beta, zs = sets["alpha_beta"], sets["zs"]

    assert all(s.declared and s.default and s.reference for s in sets.values())
    assert (alpha_beta.kind, alpha_beta.fields, alpha_beta.provenance) == (
        "alpha_beta",
        ("alpha", "beta"),
        "LEM_I",
    )
    # The z* kernels are an mMKM dataset with saturation-corrected z1D.
    assert (zs.kind, zs.fields, zs.provenance) == ("zs", ("zs",), "mMKM")
    assert alpha_beta.variant is None and zs.variant == "z1D_saturation_corrected"
    assert [(c.name, c.column) for c in alpha_beta.tissue_classes] == [
        ("Generic_LEMax01", 0),
        ("Generic_LEMax05", 1),
    ]
    assert [(c.name, c.column) for c in zs.tissue_classes] == [("Generic_MKM", 0)]
    (mmkm,) = zs.tissues
    assert isinstance(mmkm, MKMTissueProperties) and mmkm.provenance == "mMKM"
    assert (mmkm.r_nucleus, mmkm.r_domain) == (3.9, 0.32)
    assert oxygen.provided_quantities() == ["physical_dose", "alpha", "beta", "zs"]


def test_oxygen_declared_sets_resolve_as_the_legacy_inference_did(oxygen):
    """Declaring the sets changes no tissue class or column, hence no computed dose."""
    legacy = _without_metadata("oxygen", "kernelSets")

    def layout(machine):
        return {
            set_id: (
                s.kind,
                s.fields,
                s.provenance,
                [(c.tissue, c.column) for c in s.tissue_classes],
            )
            for set_id, s in machine.kernel_sets.items()
        }

    assert layout(oxygen) == layout(legacy)


def test_carbon_declares_lem_iv_kernels(carbon):
    (kernel_set,) = carbon.kernel_sets.values()
    assert (kernel_set.id, kernel_set.kind, kernel_set.fields, kernel_set.provenance) == (
        "alpha_beta",
        "alpha_beta",
        ("alpha", "beta"),
        "LEM_IV",
    )
    assert kernel_set.declared and kernel_set.default and kernel_set.reference
    assert kernel_set.variant is None and not kernel_set.problems
    assert [(c.name, c.column) for c in kernel_set.tissue_classes] == [
        ("Generic_LEMax01", 0),
        ("Generic_LEMax05", 1),
    ]
    # alpha_x / beta_x as the kernels tabulate them; the LEM cell parameters (d_t, r_nucleus)
    # of this data are not documented and stay unset.
    assert list(kernel_set.tissues) == [
        LEMTissueProperties(cell_line="Generic_LEMax01", alpha_x=0.1, beta_x=0.05, version="IV"),
        LEMTissueProperties(cell_line="Generic_LEMax05", alpha_x=0.5, beta_x=0.05, version="IV"),
    ]


def test_carbon_declared_set_reads_the_columns_the_legacy_inference_did(carbon):
    """Declaring the set changes no column, hence no computed dose."""
    legacy = _without_metadata("carbon", "kernelSets", "tissueParameters")

    def layout(machine):
        return {
            set_id: (
                s.kind,
                s.fields,
                [(c.tissue.alpha_x, c.tissue.beta_x, c.column) for c in s.tissue_classes],
            )
            for set_id, s in machine.kernel_sets.items()
        }

    assert layout(carbon) == layout(legacy)
    assert legacy.kernel_sets["alpha_beta"].provenance == "undeclared"


def test_carbon_lem_iv_kernels_resolve_the_generic_tissues(carbon_summary):
    """Reference-only tissues match on alpha_x / beta_x; other LEM versions with a warning."""
    model = _model("carbon", carbon_summary, "alpha_beta_kernels")
    resolution = model.resolve(
        carbon_summary,
        {
            "generic": TissueProperties(),
            "generic_05": TissueProperties(alpha_x=0.5, beta_x=0.05),
            "lem_iv": LEMTissueProperties(alpha_x=0.5, beta_x=0.05, version="IV"),
        },
    )
    assert resolution.ok
    assert resolution.pairs() == {
        "generic": ("alpha_beta", 0),
        "generic_05": ("alpha_beta", 1),
        "lem_iv": ("alpha_beta", 1),
    }
    lem_i = LEMTissueProperties(alpha_x=0.1, beta_x=0.05, d_t=30.0, r_nucleus=5.0)
    (issue,) = model.check_tissues(carbon_summary, {"PTV": lem_i})
    assert issue.kind == "provenance_differs" and issue.is_warning
    assert issue.fix_provenance == "LEM_IV"
    assert model.resolve(carbon_summary, {"PTV": lem_i}).pairs() == {"PTV": ("alpha_beta", 0)}


def test_oxygen_bio_model_options(oxygen_summary):
    assert _labels("oxygen", oxygen_summary) == {
        ("alpha_beta_kernels", "α/β kernels · LEM I", True),
        ("zstar_kernels", "z* kernels · mMKM", True),
        ("alpha_beta_tables", "α/β tables + fluence spectra · LEM I", False),
    }


def test_carbon_bio_model_options(carbon_summary):
    assert _labels("carbon", carbon_summary) == {
        ("alpha_beta_kernels", "α/β kernels · LEM IV", True),
        ("alpha_beta_tables", "α/β tables + fluence spectra · LEM I", False),
    }


@pytest.mark.parametrize("radiation_mode", ["carbon", "oxygen"])
def test_bundled_tissue_classes_pass_the_check(request, radiation_mode):
    """Every bundled tissue class resolves for the formalism reading its kernel set."""
    summary = request.getfixturevalue(f"{radiation_mode}_summary")
    for option in bio_model_options(radiation_mode, summary):
        if not option.available or option.kernel_set is None:
            continue
        model = option.create()
        for tissue in option.tissue_classes:
            if model.required_tissue_parameters:
                # z* kernels take alpha_0 / beta_0 from the structure, not the base data
                tissue = MKMTissueProperties(
                    model=tissue.model,
                    r_nucleus=tissue.r_nucleus,
                    r_domain=tissue.r_domain,
                    alpha_0=0.172,
                    beta_0=0.0615,
                )
            assert model.check_tissues(summary, {"PTV": tissue}) == [], option.label


def test_oxygen_zstar_kernels_warn_about_mkm_labelled_tissues(oxygen_summary):
    """An MKM-labelled tissue resolves to the mMKM z* kernels by its geometry, with a warning."""
    model = _model("oxygen", oxygen_summary, "zstar_kernels")
    geometry = {"r_nucleus": 3.9, "r_domain": 0.32, "alpha_0": 0.172, "beta_0": 0.0615}
    mmkm = MKMTissueProperties(model="mMKM", **geometry)
    assert model.check_tissues(oxygen_summary, {"PTV": mmkm}) == []
    (issue,) = model.check_tissues(oxygen_summary, {"PTV": MKMTissueProperties(**geometry)})
    assert issue.kind == "provenance_differs" and issue.is_warning
    (lem_issue,) = model.check_tissues(
        oxygen_summary, {"PTV": LEMTissueProperties(alpha_x=0.1, beta_x=0.05)}
    )
    assert lem_issue.kind == "type_mismatch" and lem_issue.is_error


def test_bundled_rbe_table_metadata():
    (table,) = [
        s
        for s in TabulatedAlphaBetaModel.quantity_table_kernel_sets()
        if s.id == "RBEtable_LEMI_Scholz06_AX01_BX005.mat"
    ]
    assert (table.kind, table.provenance, table.label) == (
        "rbe_table",
        "LEM_I",
        "α/β tables · LEM I",
    )
    assert "Survival" in table.reference and not table.problems
    (tissue_class,) = table.tissue_classes
    assert tissue_class.tissue == LEMTissueProperties(
        alpha_x=0.1, beta_x=0.05, d_t=30.0, r_nucleus=5.0, version="I"
    )
