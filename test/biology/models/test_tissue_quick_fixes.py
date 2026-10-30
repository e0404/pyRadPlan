"""Provenance quick fixes of tissue issues and the per-parameter tissue comparison."""

import numpy as np
import pytest

from pyRadPlan.biology.models import (
    BioMachineSummary,
    TissueComparison,
    apply_provenance_fixes,
    compare_tissue_parameters,
    create_bio_model,
)
from pyRadPlan.cst import validate_cst, validate_voi
from pyRadPlan.ct import validate_ct
from pyRadPlan.machines import load_from_name
from pyRadPlan.biology.tissues import (
    LEMTissueProperties,
    MKMTissueProperties,
    TissueProperties,
)

MKM_BASELINE = dict(r_nucleus=3.9, r_domain=0.32, alpha_0=0.172, beta_0=0.0615)


@pytest.fixture(scope="module")
def summaries():
    return {
        mode: BioMachineSummary.from_machine(load_from_name(mode, "Generic"))
        for mode in ("carbon", "oxygen")
    }


def _cst(tissues: dict):
    ct = validate_ct(
        cube_hu=np.zeros((4, 5, 6), dtype=np.float32), resolution={"x": 2, "y": 2, "z": 3}
    )
    vois = []
    for i, (name, tissue) in enumerate(tissues.items()):
        mask = np.zeros((4, 5, 6), dtype=np.uint8)
        mask[i] = 1
        vois.append(
            validate_voi(
                name=name, voi_type="OAR", mask=mask, ct_image=ct, tissue_properties=tissue
            )
        )
    return validate_cst(vois, ct)


def test_mkm_tissue_for_mmkm_kernels_warns_and_offers_a_relabel(summaries):
    model = create_bio_model("zstar_kernels")
    tissue = MKMTissueProperties(model="MKM", cell_line="HSG", **MKM_BASELINE)
    (issue,) = model.check_tissues(summaries["oxygen"], {"PTV": tissue})
    assert issue.kind == "provenance_differs"
    assert issue.severity == "warning"
    assert issue.fix_provenance == "mMKM"
    assert issue.suggestion.cell_line == "Generic_MKM"
    assert str(issue).startswith("PTV: The tissue (HSG) is labelled MKM; the base data is mMKM")
    assert "Evaluated with the base data's model" in issue.message
    assert 'adopt_provenance("mMKM")' in issue.message

    fixed = apply_provenance_fixes({"PTV": tissue}, [issue])
    assert fixed["PTV"].provenance == "mMKM"
    assert tissue.model == "MKM"
    assert model.check_tissues(summaries["oxygen"], fixed) == []


def test_lem_version_mismatch_warns_and_offers_the_version_relabel(summaries):
    tissue = LEMTissueProperties(alpha_x=0.1, beta_x=0.05, d_t=30.0, r_nucleus=5.0)
    model = create_bio_model("alpha_beta_kernels")
    (issue,) = model.check_tissues(summaries["carbon"], {"PTV": tissue})
    assert issue.kind == "provenance_differs"
    assert issue.is_warning
    assert issue.fix_provenance == "LEM_IV"
    assert "labelled LEM I; the base data is LEM IV" in issue.message
    assert model.resolve(summaries["carbon"], {"PTV": tissue}).pairs() == {
        "PTV": ("alpha_beta", 0)
    }


def test_other_parameter_classes_get_no_provenance_fix(summaries):
    tissue = LEMTissueProperties(alpha_x=0.1, beta_x=0.05)
    (issue,) = create_bio_model("zstar_kernels").check_tissues(
        summaries["oxygen"], {"PTV": tissue}
    )
    assert issue.kind == "type_mismatch"
    assert issue.fix_provenance is None
    assert "adopt_provenance" not in issue.message


def test_apply_provenance_fixes_to_a_structure_set(summaries):
    model = create_bio_model("zstar_kernels")
    cst = _cst(
        {
            "PTV": MKMTissueProperties(model="MKM", **MKM_BASELINE),
            "OAR": MKMTissueProperties(model="mMKM", **MKM_BASELINE),
        }
    )
    issues = model.check_tissues(summaries["oxygen"], cst)
    assert [(i.structure, i.fix_provenance) for i in issues] == [("PTV", "mMKM")]

    fixed = apply_provenance_fixes(cst, issues)
    assert fixed is not cst
    assert cst.vois[0].tissue_properties.model == "MKM"
    assert fixed.vois[0].tissue_properties.model == "mMKM"
    # The derived reference follows the relabelling: saturation-corrected for mMKM
    assert fixed.vois[0].alpha_x == fixed.vois[0].tissue_properties.alpha_x
    assert fixed.vois[0].alpha_x == pytest.approx(cst.vois[1].alpha_x)
    assert fixed.vois[0].alpha_x < cst.vois[0].alpha_x
    assert fixed.vois[1] is cst.vois[1]
    assert model.check_tissues(summaries["oxygen"], fixed) == []
    assert [v.tissue_properties.model for v in apply_provenance_fixes(cst.vois, issues)] == [
        "mMKM",
        "mMKM",
    ]


def test_comparison_flags_the_mismatching_parameters(summaries):
    model = create_bio_model("zstar_kernels")
    tissues = {
        "PTV": MKMTissueProperties(model="mMKM", **MKM_BASELINE),
        "OAR": MKMTissueProperties(model="mMKM", **{**MKM_BASELINE, "r_domain": 0.3}),
        "BODY": MKMTissueProperties(model="MKM", **MKM_BASELINE),
    }
    ptv, oar, body = compare_tissue_parameters(model, summaries["oxygen"], tissues)
    assert all(isinstance(c, TissueComparison) for c in (ptv, oar, body))

    assert ptv.matched and not ptv.mismatched and ptv.kernel_set == "zs"
    assert [(p.name, p.declared, p.value) for p in ptv.parameters] == [
        ("r_nucleus", 3.9, 3.9),
        ("r_domain", 0.32, 0.32),
    ]

    assert not oar.matched and not oar.provenance_mismatch
    assert oar.mismatched == ("r_domain",)
    assert oar.tissue_class.cell_line == "Generic_MKM"

    # Matches by its parameters, labelled with another provenance
    assert body.matched and body.provenance_mismatch
    assert body.fix_provenance == "mMKM"
    assert body.issue.kind == "provenance_differs"
    assert body.mismatched == ()


def test_comparison_of_another_parameter_class(summaries):
    tissue = LEMTissueProperties(alpha_x=0.1, beta_x=0.05, d_t=30.0, r_nucleus=5.0)
    (comparison,) = create_bio_model("zstar_kernels").compare_tissues(
        summaries["oxygen"], {"PTV": tissue}
    )
    assert not comparison.matched and comparison.provenance_mismatch
    assert comparison.issue.kind == "type_mismatch" and comparison.issue.is_error
    assert comparison.fix_provenance is None


def test_comparison_excludes_undeclared_parameters(summaries):
    tissue = LEMTissueProperties(alpha_x=0.5, beta_x=0.05, version="IV", d_t=30.0)
    (comparison,) = create_bio_model("alpha_beta_kernels").compare_tissues(
        summaries["carbon"], {"PTV": tissue}
    )
    assert comparison.matched and not comparison.provenance_mismatch
    assert [p.name for p in comparison.parameters] == ["alpha_x", "beta_x"]


def test_comparison_agrees_with_check_tissues(summaries):
    tissues = {
        "a": LEMTissueProperties(alpha_x=0.1, beta_x=0.05, d_t=30.0, r_nucleus=5.0),
        "b": LEMTissueProperties(alpha_x=0.3, beta_x=0.05, d_t=30.0, r_nucleus=5.0),
        "c": MKMTissueProperties(model="mMKM", **MKM_BASELINE),
        "d": TissueProperties(alpha_x=0.5, beta_x=0.05),
        "e": TissueProperties(alpha_x=0.2, beta_x=0.05),
        "f": LEMTissueProperties(alpha_x=0.5, beta_x=0.05, version="III"),
        "g": MKMTissueProperties(model="MKM", **MKM_BASELINE),
    }
    for mode, name in (
        ("oxygen", "alpha_beta_kernels"),
        ("oxygen", "zstar_kernels"),
        ("carbon", "alpha_beta_kernels"),
    ):
        model = create_bio_model(name)
        issues = {
            i.structure: i
            for i in model.check_tissues(summaries[mode], tissues)
            if i.kind in ("type_mismatch", "no_matching_class", "provenance_differs")
        }
        for comparison in model.compare_tissues(summaries[mode], tissues):
            issue = issues.get(comparison.structure)
            assert comparison.matched == (issue is None or issue.is_warning)
            assert comparison.provenance_mismatch == (
                issue is not None and issue.kind in ("type_mismatch", "provenance_differs")
            )
            if issue is not None:
                assert comparison.tissue_class is issue.suggestion or (
                    comparison.tissue_class == issue.suggestion
                )
                assert comparison.fix_provenance == issue.fix_provenance


def test_comparison_without_tissue_dependent_base_data():
    tissue = TissueProperties()
    (comparison,) = compare_tissue_parameters(create_bio_model("constant_rbe"), None, [tissue])
    assert comparison.tissue_class is None and comparison.parameters == ()
