"""Biological model availability depends on the dose engine as well as on the machine."""

import pytest

from pyRadPlan.biology.models import (
    BioMachineSummary,
    EngineBioCapability,
    KernelSetInfo,
    LQModel,
    Wedenberg,
    bio_model_options,
    create_bio_model,
    engine_bio_capability,
)
from pyRadPlan.dose.engines import (
    ParticleFredMCEngine,
    ParticleHongPencilBeamEngine,
    ParticleTOPASMCEngine,
    PhotonPencilBeamSVDEngine,
    get_engine_class,
)
from pyRadPlan.machines import load_from_name
from pyRadPlan.biology.tissues import LEMTissueProperties

LEM_A01 = LEMTissueProperties(alpha_x=0.1, beta_x=0.05, d_t=30.0, r_nucleus=5.0)


@pytest.fixture(scope="module")
def summaries():
    """Summaries of the bundled machines, read once."""
    return {
        mode: BioMachineSummary.from_machine(load_from_name(mode, "Generic"))
        for mode in ("photons", "protons", "helium", "carbon", "oxygen")
    }


def _two_let_sets():
    """Summary of a proton machine with a dose- and a track-averaged LET."""
    sets = {
        "letd": KernelSetInfo(id="letd", kind="let", fields=("let",), variant="LETd"),
        "lett": KernelSetInfo(id="lett", kind="let", fields=("let",), variant="LETt"),
    }
    return BioMachineSummary(
        name="TwoLET",
        radiation_mode="protons",
        kernel_sets=sets,
        quantities=("physical_dose", "let", "let:LETd", "let:LETt"),
    )


def _availability(mode, summary, engine):
    return {o.label: o.available for o in bio_model_options(mode, summary, engine=engine)}


def _reasons(mode, summary, engine):
    return {o.label: o.reason for o in bio_model_options(mode, summary, engine=engine)}


# ------------------------------------------------------------------ declarations
@pytest.mark.parametrize(
    ("engine", "kernel_quantities", "scored", "computes", "scored_modes"),
    [
        (ParticleHongPencilBeamEngine, None, (), True, None),
        (ParticleFredMCEngine, frozenset(), ("let:LETd",), True, None),
        (ParticleTOPASMCEngine, frozenset(), ("let:LETd",), True, frozenset({"protons"})),
        (PhotonPencilBeamSVDEngine, frozenset(), (), False, None),
    ],
)
def test_engines_declare_what_they_supply(
    engine, kernel_quantities, scored, computes, scored_modes
):
    """Each engine class declares what it reads, scores and computes."""
    capability = engine.bio_capability()
    assert capability == EngineBioCapability(
        engine=engine.short_name,
        kernel_quantities=kernel_quantities,
        scored_quantities=scored,
        computes_bio_influence=computes,
        scored_radiation_modes=scored_modes,
    )


def test_capability_narrows_to_the_radiation_modes_it_scores_for():
    """TOPAS scores the LET of protons only; for other ions it scores nothing."""
    capability = ParticleTOPASMCEngine.bio_capability()
    assert capability.for_radiation_mode("protons") is capability
    assert capability.for_radiation_mode("") is capability
    carbon = capability.for_radiation_mode("carbon")
    assert carbon.scored_quantities == ()
    assert carbon.radiation_mode == "carbon"
    assert not carbon.supplies("let")
    fred = ParticleFredMCEngine.bio_capability()
    assert fred.for_radiation_mode("carbon") is fred


def test_engine_capability_from_name_class_instance_or_capability():
    capability = ParticleFredMCEngine.bio_capability()
    assert engine_bio_capability("FRED") == capability
    assert engine_bio_capability(ParticleFredMCEngine) == capability
    assert engine_bio_capability(capability) is capability
    assert engine_bio_capability(None) is None
    with pytest.raises(ValueError, match="Unknown dose engine 'NoSuchEngine'"):
        engine_bio_capability("NoSuchEngine")
    with pytest.raises(TypeError, match="declares no biological capability"):
        engine_bio_capability(object())


def test_engine_class_is_chosen_like_the_factory():
    assert get_engine_class("carbon") is ParticleHongPencilBeamEngine
    assert get_engine_class("carbon", "FRED") is ParticleFredMCEngine
    assert get_engine_class("photons", "FRED") is PhotonPencilBeamSVDEngine
    assert get_engine_class({"radiation_mode": "protons", "prop_dose_calc": {"engine": "TOPAS"}})
    assert get_engine_class("electrons") is None


def test_bio_quantities_follow_what_the_engine_reads_or_scores(summaries):
    two_let = _two_let_sets()
    assert ParticleHongPencilBeamEngine.bio_quantities(summaries["oxygen"]) == [
        "physical_dose",
        "alpha",
        "beta",
        "zs",
    ]
    assert ParticleHongPencilBeamEngine.bio_quantities(two_let) == list(two_let.quantities)
    assert ParticleFredMCEngine.bio_quantities(summaries["oxygen"]) == ["physical_dose", "let"]
    assert ParticleFredMCEngine.bio_quantities(two_let) == ["physical_dose", "let"]
    assert ParticleFredMCEngine.bio_quantities() == ["physical_dose", "let"]
    assert ParticleTOPASMCEngine.bio_quantities(summaries["protons"]) == ["physical_dose", "let"]
    assert ParticleTOPASMCEngine.bio_quantities(summaries["carbon"]) == ["physical_dose"]


# ------------------------------------------------------------------ options per engine
def test_pencil_beam_engine_offers_what_the_machine_offers(summaries):
    for mode in ("protons", "helium", "carbon", "oxygen"):
        assert _availability(mode, summaries[mode], "HongPB") == _availability(
            mode, summaries[mode], None
        )
    oxygen = _availability("oxygen", summaries["oxygen"], "HongPB")
    assert oxygen["α/β kernels · LEM I"]
    assert oxygen["z* kernels · mMKM"]
    assert _availability("protons", summaries["protons"], "HongPB")["Wedenberg (LET)"]
    # The carbon machine tabulates no LET, so the pencil-beam engine has none
    assert not _availability("carbon", summaries["carbon"], "HongPB")["Linear scaling (LET)"]


def test_monte_carlo_engine_offers_let_models_but_no_kernel_formalisms(summaries):
    carbon = _availability("carbon", summaries["carbon"], "FRED")
    reasons = _reasons("carbon", summaries["carbon"], "FRED")
    # FRED scores LET itself, so the machine does not have to tabulate it
    assert carbon["Linear scaling (LET)"]
    assert carbon["Constant RBE"] and carbon["None (physical dose)"]
    assert not carbon["α/β kernels · LEM IV"]
    assert reasons["α/β kernels · LEM IV"] == (
        "The dose engine FRED does not use the machine's α/β kernels."
    )
    assert not carbon["z* kernels"]
    assert not carbon["α/β tables + fluence spectra · LEM I"]
    assert (
        "FRED does not use the machine's fluence spectra"
        in (reasons["α/β tables + fluence spectra · LEM I"])
    )

    oxygen = _reasons("oxygen", summaries["oxygen"], "FRED")
    assert (
        oxygen["z* kernels · mMKM"]
        == "The dose engine FRED does not use the machine's z* kernels."
    )
    assert all(
        o.available
        for o in bio_model_options("protons", summaries["protons"], engine="FRED")
        if o.model in ("WED", "MCN", "CAR", "LSM")
    )


def test_topas_offers_let_models_for_protons_only(summaries):
    protons = _availability("protons", summaries["protons"], "TOPAS")
    assert protons["Wedenberg (LET)"] and protons["Constant RBE"]
    assert all(
        o.available
        for o in bio_model_options("protons", summaries["protons"], engine="TOPAS")
        if o.model in ("WED", "MCN", "CAR", "LSM")
    )
    carbon = _availability("carbon", summaries["carbon"], "TOPAS")
    assert not carbon["Linear scaling (LET)"]
    assert _reasons("carbon", summaries["carbon"], "TOPAS")["Linear scaling (LET)"] == (
        "The dose engine TOPAS neither uses the machine's LET kernels nor scores LET for "
        "carbon (only for protons)."
    )
    assert [label for label, available in carbon.items() if available] == [
        "None (physical dose)",
        "Constant RBE",
    ]


def test_engine_without_biological_influence_offers_constant_rbe_only(summaries):
    photons = _availability("photons", summaries["photons"], "SVDPB")
    assert photons == {"None (physical dose)": True, "Constant RBE": True}


def test_engine_reasons_need_no_machine():
    options = {o.model: o for o in bio_model_options("carbon", engine="FRED")}
    assert not options["alpha_beta_kernels"].available
    assert "FRED does not use the machine's α/β kernels" in options["alpha_beta_kernels"].reason
    assert options["LSM"].available
    assert all(o.available for o in bio_model_options("carbon", engine="HongPB"))


def test_scored_quantities_replace_the_machine_kernel_sets():
    two_let = _two_let_sets()
    hong = [o for o in bio_model_options("protons", two_let, "HongPB") if o.model == "WED"]
    assert [(o.label, o.available) for o in hong] == [
        ("Wedenberg (LET) · LETd", True),
        ("Wedenberg (LET) · LETt", True),
    ]
    fred = [o for o in bio_model_options("protons", two_let, "FRED") if o.model == "WED"]
    assert [(o.label, o.available, dict(o.parameters)) for o in fred] == [
        ("Wedenberg (LET)", True, {})
    ]


# ------------------------------------------------------------------ check_tissues
def test_check_tissues_reports_what_the_engine_cannot_compute(summaries):
    model = create_bio_model("alpha_beta_kernels")
    assert model.check_tissues(summaries["oxygen"], {"PTV": LEM_A01}, engine="HongPB") == []
    (issue,) = model.check_tissues(summaries["oxygen"], {"PTV": LEM_A01}, engine="FRED")
    assert issue.kind == "model_unavailable"
    assert issue.structure is None
    assert "The dose engine FRED does not use the machine's α/β kernels." in issue.message
    # The engine part needs no machine
    (issue,) = model.check_tissues(None, {"PTV": LEM_A01}, engine=ParticleFredMCEngine)
    assert issue.kind == "model_unavailable"


def test_let_model_needs_a_tabulated_or_scored_let(summaries):
    lsm = create_bio_model("LSM")
    assert lsm.check_tissues(summaries["carbon"], engine="FRED") == []
    (issue,) = lsm.check_tissues(summaries["carbon"], engine="HongPB")
    assert "Required quantities not provided" in issue.message
    assert create_bio_model("WED").check_tissues(summaries["protons"], engine="TOPAS") == []
    (issue,) = lsm.check_tissues(summaries["carbon"], engine="TOPAS")
    assert "TOPAS neither uses the machine's LET kernels nor scores LET for carbon" in (
        issue.message
    )


def test_let_variant_must_match_the_scored_one():
    two_let = _two_let_sets()
    lett = Wedenberg(quantity_variants={"let": "LETt"})
    assert lett.check_tissues(two_let, engine="HongPB") == []
    (issue,) = lett.check_tissues(two_let, engine="FRED")
    assert "The dose engine FRED scores LET as LETd, not LETt." in issue.message
    assert Wedenberg(quantity_variants={"let": "letd"}).check_tissues(two_let, engine="FRED") == []
    # Without a variant, FRED's scored LET is the only one: no ambiguity
    assert Wedenberg().check_tissues(two_let, engine="FRED") == []


class _PhotonLQModel(LQModel):
    model = "photon_lq_engine_capability_test"
    possible_radiation_modes = ("photons",)
    required_quantities = ("physical_dose",)

    def evaluator(self, machine, voxel_params):  # pragma: no cover - never evaluated
        raise NotImplementedError


def test_model_with_outputs_needs_an_engine_computing_biological_influence(summaries):
    (issue,) = _PhotonLQModel().check_tissues(summaries["photons"], engine="SVDPB")
    assert "SVDPB does not compute biological influence matrices" in issue.message
    assert (
        create_bio_model("constant_rbe").check_tissues(summaries["photons"], engine="SVDPB") == []
    )


@pytest.mark.parametrize("engine", ["HongPB", "FRED", "TOPAS"])
@pytest.mark.parametrize("mode", ["protons", "carbon", "oxygen"])
def test_options_and_check_agree(summaries, mode, engine):
    """An option is available exactly when its model has no model-level issue."""
    for option in bio_model_options(mode, summaries[mode], engine=engine):
        issues = option.create().check_tissues(summaries[mode], engine=engine)
        model_errors = [i for i in issues if i.is_error and i.structure is None]
        assert option.available is not bool(model_errors), (option.label, model_errors)
