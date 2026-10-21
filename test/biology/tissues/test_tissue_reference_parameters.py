"""Always-defined reference parameters, their source, and converting tissues."""

import math

import pytest

from pyRadPlan.biology.tissues import (
    DEFAULT_ALPHA_X,
    DEFAULT_BETA_X,
    DEFAULT_REFERENCE_RADIATION,
    GY_PER_KEV_PER_UM3,
    REFERENCE_SOURCES,
    LEMTissueProperties,
    MKMTissueProperties,
    ParameterSource,
    ReferenceRadiation,
    TissueProperties,
    adopt_provenance,
    apply_tissue_class,
    declared_reference_parameters,
    default_reference_parameters,
    get_reference_radiation,
    register_reference_radiation,
    registered_reference_radiations,
    saturation_corrected_specific_energy,
    saturation_specific_energy,
    specific_energy_from_lineal_energy,
)
from pyRadPlan.biology.tissues import _reference as reference_module

MKM_BASELINE = dict(alpha_0=0.172, beta_0=0.0615, r_domain=0.32, r_nucleus=3.9)


def test_defaults_live_in_one_place():
    assert default_reference_parameters() == {"alpha_x": DEFAULT_ALPHA_X, "beta_x": DEFAULT_BETA_X}
    assert (DEFAULT_ALPHA_X, DEFAULT_BETA_X) == (0.1, 0.05)
    assert REFERENCE_SOURCES == ("explicit", "published", "estimated", "base_data", "default")


@pytest.mark.parametrize("tissue_cls", [TissueProperties, LEMTissueProperties])
@pytest.mark.parametrize("unset", [{}, {"alpha_x": math.nan, "beta_x": None}])
def test_unset_reference_parameters_get_the_defaults(tissue_cls, unset):
    tissue = tissue_cls(**unset)
    assert (tissue.alpha_x, tissue.beta_x) == (DEFAULT_ALPHA_X, DEFAULT_BETA_X)
    assert tissue.reference_source == "default"


def test_explicit_reference_parameters_are_kept():
    tissue = LEMTissueProperties(alpha_x=0.3, beta_x=0.03)
    assert (tissue.alpha_x, tissue.beta_x) == (0.3, 0.03)
    assert tissue.reference_source == "explicit"


def test_partially_given_reference_parameters_report_the_weakest_source():
    tissue = TissueProperties(alpha_x=0.3, beta_x=math.nan)
    assert (tissue.alpha_x, tissue.beta_x) == (0.3, DEFAULT_BETA_X)
    assert tissue.reference_source == "default"


def test_mkm_estimates_unset_reference_parameters():
    tissue = MKMTissueProperties(**MKM_BASELINE, alpha_x=math.nan)
    expected = MKMTissueProperties.estimate_reference_parameters(
        alpha_0=0.172, beta_0=0.0615, r_domain=0.32
    )
    assert tissue.alpha_x == pytest.approx(
        MKM_BASELINE["alpha_0"]
        + MKM_BASELINE["beta_0"] * MKMTissueProperties.reference_specific_energy(0.32)
    )
    assert tissue.alpha_x == expected["alpha_x"]
    assert tissue.beta_x == MKM_BASELINE["beta_0"]
    assert tissue.reference_source == "estimated"


def test_mkm_without_baseline_parameters_gets_the_defaults():
    tissue = MKMTissueProperties(r_domain=0.32, alpha_x=math.nan, beta_x=math.nan)
    assert (tissue.alpha_x, tissue.beta_x) == (DEFAULT_ALPHA_X, DEFAULT_BETA_X)
    assert tissue.reference_source == "default"
    only_beta = MKMTissueProperties(beta_0=0.06)
    assert (only_beta.alpha_x, only_beta.beta_x) == (DEFAULT_ALPHA_X, 0.06)
    assert only_beta.reference_source == "default"


@pytest.mark.parametrize(
    "data",
    [
        {"model": "MKM", "alphaX": math.nan, "betaX": math.nan, "rDomain": 0.32},
        {"model": "LEM_II", "alpha_x": None, "beta_x": math.nan, "Dt": 6.0},
        {"model": "Generic", "alphaX": math.nan, "betaX": math.nan},
    ],
)
def test_old_files_with_unset_reference_parameters_load(data):
    tissue = TissueProperties.from_dict(data)
    assert math.isfinite(tissue.alpha_x) and math.isfinite(tissue.beta_x)
    assert tissue.reference_source == "default"
    assert type(tissue).model_validate(data).reference_source == "default"


def test_serialized_source_is_kept_and_validated():
    tissue = MKMTissueProperties(**MKM_BASELINE)
    assert tissue.reference_source == "estimated"
    for dump in (tissue.model_dump(), tissue.model_dump(by_alias=True)):
        restored = TissueProperties.from_dict(dump)
        assert restored == tissue
        assert restored.reference_source == "estimated"
    dumped = tissue.model_dump(by_alias=True)
    assert dumped["referenceSources"] == {"alpha_x": "estimated", "beta_x": "estimated"}
    assert "referenceSource" not in dumped
    restored = MKMTissueProperties.model_validate_json(tissue.model_dump_json())
    assert restored.reference_sources == tissue.reference_sources
    with pytest.raises(ValueError, match="reference_source must be one of"):
        TissueProperties(reference_source="guessed")
    with pytest.raises(ValueError, match="reference_source must be one of"):
        TissueProperties(reference_sources={"alpha_x": "guessed"})
    with pytest.raises(ValueError, match="unknown parameters"):
        TissueProperties(reference_sources={"alpha_0": "explicit"})


def test_assigning_a_reference_parameter_makes_it_explicit():
    tissue = MKMTissueProperties(**MKM_BASELINE)
    tissue.alpha_x = 0.25
    assert tissue.reference_sources == {"alpha_x": "explicit", "beta_x": "estimated"}
    tissue.beta_x = 0.06
    assert tissue.reference_source == "explicit"
    default = TissueProperties()
    default.cell_line = "HSG"
    assert default.reference_source == "default"
    default.beta_x = math.nan
    assert default.beta_x == DEFAULT_BETA_X


def test_reference_source_is_not_part_of_equality_or_matching():
    explicit = TissueProperties(alpha_x=0.1, beta_x=0.05)
    default = TissueProperties()
    assert (explicit.reference_source, default.reference_source) == ("explicit", "default")
    assert explicit == default
    assert explicit.matches(default) and default.matches(explicit)
    assert "reference_source" not in repr(explicit)


def test_base_data_classes_keep_undeclared_reference_parameters():
    tissue_class = TissueProperties.from_base_data(
        {"model": "mMKM", "alphaX": math.nan, "betaX": math.nan, "rNucleus": 3.9, "rDomain": 0.32}
    )
    assert type(tissue_class) is MKMTissueProperties
    assert math.isnan(tissue_class.alpha_x) and math.isnan(tissue_class.beta_x)
    assert declared_reference_parameters(tissue_class) == {}

    structure = MKMTissueProperties(model="mMKM", r_nucleus=3.9, r_domain=0.32, beta_0=0.06)
    assert structure.matches(tissue_class, ignore_undeclared=True)
    # a reference-only tissue never matches the class vacuously on its undeclared parameters
    assert not TissueProperties().matches(tissue_class, ignore_undeclared=True)


def test_base_data_classes_declare_given_reference_parameters():
    tissue_class = LEMTissueProperties.from_base_data(
        alpha_x=0.5, beta_x=0.05, version="IV", d_t=math.nan
    )
    assert tissue_class.reference_source == "base_data"
    assert declared_reference_parameters(tissue_class) == {"alpha_x": 0.5, "beta_x": 0.05}
    absent = TissueProperties.from_base_data({"model": "LEM_IV"})
    assert absent.reference_source == "default"
    assert declared_reference_parameters(absent) == {}


def test_bundled_oxygen_mmkm_class_keeps_nan_and_resolves():
    from pyRadPlan.biology.models import create_bio_model  # noqa: PLC0415
    from pyRadPlan.machines import load_from_name  # noqa: PLC0415

    machine = load_from_name("oxygen", "Generic")
    mmkm = next(t for t in machine.tissue_properties if t.model == "mMKM")
    assert math.isnan(mmkm.alpha_x) and math.isnan(mmkm.beta_x)

    structure = MKMTissueProperties(model="mMKM", **MKM_BASELINE)
    assert create_bio_model("zstar_kernels").check_tissues(machine, {"PTV": structure}) == []


# ----------------------------------------------------------------- adopt_provenance
def test_adopt_mkm_preset_as_mmkm():
    preset = TissueProperties.from_preset("CHO", "MKM", reference_source="estimated")
    adoption = adopt_provenance(preset, "mMKM")
    tissue = adoption.tissue
    assert type(tissue) is MKMTissueProperties and tissue.model == "mMKM"
    assert tissue.provenance == "mMKM"
    for name in ("beta_x", "r_nucleus", "r_domain", "alpha_0", "beta_0"):
        assert getattr(tissue, name) == getattr(preset, name)
    # The derived reference follows the target: mMKM saturation-corrects z_1D,ref
    assert tissue.reference_source == "estimated"
    assert tissue.alpha_x == tissue.derived_reference_parameters()["alpha_x"]
    assert tissue.alpha_x < preset.alpha_x
    assert tissue.cell_line == "CHO"
    assert tissue.cell_line_source == preset.cell_line_source
    assert set(adoption.carried) == {
        "alpha_x",
        "beta_x",
        "r_nucleus",
        "r_domain",
        "alpha_0",
        "beta_0",
    }
    assert adoption.defaulted == ()
    assert adoption.describe() == "values from CHO · MKM; adopted as mMKM"
    assert preset.adopt_provenance("mMKM") == tissue
    assert preset.model == "MKM"


def test_adopt_lem_i_as_lem_iv():
    preset = TissueProperties.from_preset("HSG", "LEM_I")
    adoption = adopt_provenance(preset, "LEM_IV")
    tissue = adoption.tissue
    assert (tissue.model, tissue.version, tissue.provenance) == ("LEM", "IV", "LEM_IV")
    assert (tissue.d_t, tissue.r_nucleus) == (preset.d_t, preset.r_nucleus)
    assert set(adoption.carried) == {"alpha_x", "beta_x", "d_t", "r_nucleus"}
    assert adoption.describe() == "values from HSG · LEM I; adopted as LEM IV"


def test_adopt_lem_as_mkm_shares_only_the_reference_parameters():
    preset = TissueProperties.from_preset("HSG", "LEM_I")
    adoption = adopt_provenance(preset, MKMTissueProperties)
    tissue = adoption.tissue
    assert type(tissue) is MKMTissueProperties and tissue.provenance == "MKM"
    assert adoption.carried == ("alpha_x", "beta_x")
    assert set(adoption.defaulted) == {"r_nucleus", "r_domain", "alpha_0", "beta_0"}
    assert (tissue.alpha_x, tissue.beta_x) == (preset.alpha_x, preset.beta_x)
    assert tissue.reference_source == "published"
    assert math.isnan(tissue.r_nucleus)


def test_adopt_keeps_a_published_reference():
    preset = TissueProperties.from_preset("CHO", "MKM")
    tissue = preset.adopt_provenance("mMKM")
    assert tissue.reference_source == "published"
    assert (tissue.alpha_x, tissue.beta_x) == (0.228, 0.02) == (preset.alpha_x, preset.beta_x)


def test_adopt_keeps_the_reference_source_and_skips_unset_parameters():
    tissue = MKMTissueProperties(alpha_0=0.2, beta_0=0.05, r_domain=0.3)
    adoption = adopt_provenance(tissue, "mMKM")
    assert "r_nucleus" in adoption.defaulted and "r_nucleus" not in adoption.carried
    # mMKM needs r_nucleus for the saturation correction, so alpha_x falls back
    assert adoption.tissue.reference_source == "default"
    assert adoption.tissue.alpha_x == DEFAULT_ALPHA_X
    assert adoption.tissue.beta_x == 0.05
    assert not adopt_provenance(tissue, tissue).changed

    with_nucleus = MKMTissueProperties(alpha_0=0.2, beta_0=0.05, r_domain=0.3, r_nucleus=4.0)
    assert adopt_provenance(with_nucleus, "mMKM").tissue.reference_source == "estimated"


# ----------------------------------------------------------------- apply_tissue_class
def test_apply_mmkm_class_keeps_the_tissue_parameters():
    tissue_class = TissueProperties.from_base_data(
        {"model": "mMKM", "alphaX": math.nan, "betaX": math.nan, "rNucleus": 3.9, "rDomain": 0.32}
    )
    preset = TissueProperties.from_preset("CHO", "MKM", reference_source="estimated")
    applied = apply_tissue_class(preset, tissue_class)
    assert applied.provenance == "mMKM"
    assert (applied.r_nucleus, applied.r_domain) == (3.9, 0.32)
    assert (applied.alpha_0, applied.beta_0) == (preset.alpha_0, preset.beta_0)
    # The reference is derived from the class's geometry, as mMKM
    expected = MKMTissueProperties.estimate_reference_parameters(
        alpha_0=preset.alpha_0, beta_0=preset.beta_0, r_domain=0.32, r_nucleus=3.9, model="mMKM"
    )
    assert (applied.alpha_x, applied.beta_x) == (expected["alpha_x"], expected["beta_x"])
    assert applied.reference_source == "estimated"
    assert applied.cell_line == "CHO"
    assert applied.matches(tissue_class, ignore_undeclared=True)

    explicit = preset.model_copy(update={"alpha_x": 0.3})
    explicit.reference_source = "explicit"
    kept = apply_tissue_class(explicit, tissue_class)
    assert (kept.alpha_x, kept.reference_source) == (0.3, "explicit")
    published = apply_tissue_class(TissueProperties.from_preset("CHO", "MKM"), tissue_class)
    assert (published.alpha_x, published.reference_source) == (0.228, "published")


def test_apply_lem_class_copies_its_declared_parameters():
    tissue_class = LEMTissueProperties.from_base_data(
        alpha_x=0.5, beta_x=0.05, version="IV", d_t=math.nan, r_nucleus=math.nan
    )
    tissue = TissueProperties.from_preset("HSG", "LEM_I")
    applied = apply_tissue_class(tissue, tissue_class)
    assert (applied.version, applied.alpha_x, applied.beta_x) == ("IV", 0.5, 0.05)
    assert (applied.d_t, applied.r_nucleus) == (tissue.d_t, tissue.r_nucleus)
    assert applied.reference_source == "base_data"
    assert applied.matches(tissue_class, ignore_undeclared=True)


def test_apply_class_reference_values_replace_only_defaults():
    tissue_class = MKMTissueProperties.from_base_data(
        model="mMKM", alpha_x=0.2, beta_x=0.04, r_nucleus=3.9, r_domain=0.32
    )
    defaulted = apply_tissue_class(MKMTissueProperties(model="mMKM"), tissue_class)
    assert (defaulted.alpha_x, defaulted.beta_x) == (0.2, 0.04)
    assert defaulted.reference_source == "base_data"

    # Only the defaulted parameter is replaced, the explicit one is kept
    half = MKMTissueProperties(model="mMKM", alpha_x=0.3)
    assert half.reference_sources == {"alpha_x": "explicit", "beta_x": "default"}
    applied = apply_tissue_class(half, tissue_class)
    assert (applied.alpha_x, applied.beta_x) == (0.3, 0.04)
    assert applied.reference_sources == {"alpha_x": "explicit", "beta_x": "base_data"}

    estimated = MKMTissueProperties(model="mMKM", **MKM_BASELINE)
    kept = apply_tissue_class(estimated, tissue_class)
    assert (kept.alpha_x, kept.reference_source) == (estimated.alpha_x, "estimated")


def test_apply_reference_only_class_keeps_the_tissue_type():
    tissue_class = TissueProperties.from_base_data(alpha_x=0.5, beta_x=0.05)
    tissue = TissueProperties.from_preset("HSG", "LEM_II")
    applied = apply_tissue_class(tissue, tissue_class)
    assert type(applied) is LEMTissueProperties and applied.version == "II"
    assert (applied.alpha_x, applied.beta_x, applied.d_t) == (0.5, 0.05, tissue.d_t)
    assert applied.reference_source == "base_data"


# ----------------------------------------------------------------- MKM conversion
# Hand-computed for r_domain = 0.32 µm, r_nucleus = 3.9 µm, beta_0 = 0.0615 Gy^-2 and the
# default reference radiation (200 kV X-rays, y_D = 4.2 keV/µm):
#   z_1D  = 0.1602 * 4.2 / (pi * 0.32**2)                         = 2.0915198 Gy
#   z_0   = 3.9**2 / (0.32 * sqrt(0.0615 * (0.32**2 + 3.9**2)))    = 48.980143 Gy
#   z*_1D = z_0**2 * (1 - exp(-(z_1D / z_0)**2)) / z_1D            = 2.0896141 Gy
Z_1D = 2.0915198
Z_0 = 48.980143
Z_STAR_1D = 2.0896141


def test_conversion_building_blocks():
    assert GY_PER_KEV_PER_UM3 == 0.1602
    assert specific_energy_from_lineal_energy(4.2, 0.32) == pytest.approx(Z_1D, rel=1e-7)
    assert saturation_specific_energy(0.32, 3.9, 0.0615) == pytest.approx(Z_0, rel=1e-7)
    assert saturation_corrected_specific_energy(Z_1D, Z_0) == pytest.approx(Z_STAR_1D, rel=1e-7)
    # No saturation without beta_0, and none for a vanishing specific energy
    assert saturation_specific_energy(0.32, 3.9, 0.0) == math.inf
    assert saturation_corrected_specific_energy(Z_1D, math.inf) == Z_1D
    assert saturation_corrected_specific_energy(0.0, Z_0) == 0.0


def test_mkm_and_mmkm_convert_with_their_own_specific_energy():
    mkm = MKMTissueProperties(model="MKM", **MKM_BASELINE)
    mmkm = MKMTissueProperties(model="mMKM", **MKM_BASELINE)
    assert MKMTissueProperties.reference_specific_energy(0.32) == pytest.approx(Z_1D, rel=1e-7)
    assert MKMTissueProperties.reference_specific_energy(
        0.32, 3.9, 0.0615, model="mMKM"
    ) == pytest.approx(Z_STAR_1D, rel=1e-7)
    assert mkm.alpha_x == pytest.approx(0.172 + 0.0615 * Z_1D, rel=1e-7)
    assert mmkm.alpha_x == pytest.approx(0.172 + 0.0615 * Z_STAR_1D, rel=1e-7)
    assert mkm.beta_x == mmkm.beta_x == 0.0615
    assert (mkm.reference_source, mmkm.reference_source) == ("estimated", "estimated")
    # The same derivation from parameters given by name, e.g. as entered in an editor
    assert MKMTissueProperties.derive_reference({"model": "mMKM", **MKM_BASELINE}) == {
        "alpha_x": mmkm.alpha_x,
        "beta_x": mmkm.beta_x,
    }
    assert MKMTissueProperties.derive_reference({"alpha0": 0.172, "beta0": 0.0615}) == {
        "beta_x": 0.0615
    }
    assert LEMTissueProperties.derive_reference({"alpha_x": 0.3, "d_t": 30.0}) == {}
    # The saturation correction needs the nucleus radius
    assert MKMTissueProperties.estimate_reference_parameters(
        alpha_0=0.172, beta_0=0.0615, r_domain=0.32, model="mMKM"
    ) == {"beta_x": 0.0615}
    assert math.isnan(MKMTissueProperties.reference_specific_energy(0.32, model="mMKM"))


@pytest.fixture
def reference_radiations(monkeypatch):
    monkeypatch.setattr(
        reference_module, "_REFERENCE_RADIATIONS", dict(reference_module._REFERENCE_RADIATIONS)
    )


def test_reference_radiation_is_configurable(reference_radiations):
    default = get_reference_radiation()
    assert default.name == DEFAULT_REFERENCE_RADIATION == "xray_200kV"
    assert default.y_d == 4.2 and default.z_1d is None
    assert default in registered_reference_radiations()
    assert MKMTissueProperties().reference_radiation == DEFAULT_REFERENCE_RADIATION

    electrons = register_reference_radiation("test_electrons", "Electrons", y_d=2.0)
    fixed = register_reference_radiation("test_fixed", z_1d=1.5, source="Unpublished")
    assert electrons.describe() == "Electrons (y_D = 2 keV/µm)"
    assert electrons.source == ParameterSource() and not electrons.source.verified
    assert fixed.source.label == "unpublished"
    assert fixed.details() == "z_1D = 1.5 Gy"
    assert fixed.specific_energy(0.1) == fixed.specific_energy(0.5) == 1.5

    tissue = MKMTissueProperties(reference_radiation="test_electrons", **MKM_BASELINE)
    z_electrons = 0.1602 * 2.0 / (math.pi * 0.32**2)
    assert tissue.alpha_x == pytest.approx(0.172 + 0.0615 * z_electrons, rel=1e-9)
    tissue.reference_radiation = "test_fixed"
    assert tissue.alpha_x == pytest.approx(0.172 + 0.0615 * 1.5, rel=1e-9)
    assert tissue.reference_source == "estimated"
    assert MKMTissueProperties.model_validate(tissue.model_dump()) == tissue
    assert tissue != MKMTissueProperties(**MKM_BASELINE)

    with pytest.raises(ValueError, match="Unknown reference radiation"):
        MKMTissueProperties(reference_radiation="nope", **MKM_BASELINE)
    with pytest.raises(ValueError, match="exactly one"):
        ReferenceRadiation("bad", "Bad", y_d=1.0, z_1d=1.0)
    with pytest.raises(KeyError):
        get_reference_radiation("nope")


def test_derived_reference_follows_the_model_parameters():
    tissue = MKMTissueProperties(**MKM_BASELINE)
    tissue.alpha_0 = 0.2
    assert tissue.alpha_x == pytest.approx(0.2 + 0.0615 * Z_1D, rel=1e-7)
    tissue.model = "mMKM"
    assert tissue.alpha_x == pytest.approx(0.2 + 0.0615 * Z_STAR_1D, rel=1e-7)
    assert tissue.reference_source == "estimated"
    assert tissue.derived_reference_parameters() == {
        "alpha_x": tissue.alpha_x,
        "beta_x": tissue.beta_x,
    }

    # An explicit override is kept until it is unset again; the other is still derived
    tissue.alpha_x = 0.25
    tissue.beta_0 = 0.07
    assert (tissue.alpha_x, tissue.beta_x) == (0.25, 0.07)
    assert tissue.reference_sources == {"alpha_x": "explicit", "beta_x": "estimated"}
    tissue.alpha_x = math.nan
    assert tissue.reference_source == "estimated"
    assert tissue.alpha_x == tissue.derived_reference_parameters()["alpha_x"]

    # Losing what alpha_x derives from falls back to the default, never to NaN
    tissue.r_domain = math.nan
    assert (tissue.alpha_x, tissue.beta_x) == (DEFAULT_ALPHA_X, 0.07)
    assert tissue.reference_source == "default"
    tissue.r_domain = 0.32
    assert tissue.reference_source == "estimated"


def test_stale_derived_values_are_derived_again():
    dump = MKMTissueProperties(**MKM_BASELINE).model_dump()
    stale = {**dump, "alpha_0": 0.3}
    restored = MKMTissueProperties.model_validate(stale)
    assert restored.alpha_x == pytest.approx(0.3 + 0.0615 * Z_1D, rel=1e-7)
    explicit = {**dump, "alpha_x": 0.5, "reference_source": "explicit"}
    assert MKMTissueProperties.model_validate(explicit).alpha_x == 0.5
    # Base data keeps what the class declares
    tissue_class = MKMTissueProperties.from_base_data(stale)
    assert tissue_class.alpha_x == dump["alpha_x"]


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"alpha_0": 0.2},
        {"beta_0": 0.05},
        {"r_domain": 0.3, "r_nucleus": 4.0},
        {"model": "mMKM", "alpha_0": 0.2, "beta_0": 0.05, "r_domain": 0.3},
        {"alpha_0": math.nan, "beta_0": math.nan, "alpha_x": math.nan, "beta_x": None},
    ],
)
def test_mkm_reference_parameters_are_never_nan(params):
    tissue = MKMTissueProperties(**params)
    assert math.isfinite(tissue.alpha_x) and math.isfinite(tissue.beta_x)
    tissue.alpha_0 = math.nan
    assert math.isfinite(tissue.alpha_x) and math.isfinite(tissue.beta_x)


# Built-in reference radiations: name -> (y_D [keV/um], site diameter [um], short source, DOI)
BUILTIN_REFERENCE_RADIATIONS = {
    "xray_200kV": (4.2, 0.97, "Coppola 1976", "10.1269/jrr.10073"),
    "xray_200kV_tepc": (4.51, 1.0, "Okamoto 2011", "10.1269/jrr.10073"),
    "co60": (2.34, 1.0, "Okamoto 2011", "10.1269/jrr.10073"),
    "xray_6MV": (2.36, 1.0, "Okamoto 2011", "10.1269/jrr.10073"),
}


def test_builtin_reference_radiations_are_cited():
    registered = {radiation.name: radiation for radiation in registered_reference_radiations()}
    assert set(BUILTIN_REFERENCE_RADIATIONS) <= set(registered)
    for name, (y_d, site, short, doi) in BUILTIN_REFERENCE_RADIATIONS.items():
        radiation = registered[name]
        assert (radiation.y_d, radiation.z_1d) == (y_d, None)
        assert radiation.site_diameter == site
        assert isinstance(radiation.source, ParameterSource) and radiation.source.verified
        assert radiation.source.short == short and doi in radiation.source.citation
        assert radiation.source.note
    default = get_reference_radiation()
    assert default.details() == "y_D = 4.2 keV/µm in a 0.97 µm site"
    assert "2.06 µm" in default.source.note  # y_D depends on the site size


def test_reference_radiation_sources_are_coerced(reference_radiations):
    cited = ParameterSource("Doe et al., J. Test. 1 (2020) 1", short="Doe 2020")
    assert register_reference_radiation("a", y_d=1.0, source=cited).source is cited
    assert register_reference_radiation("b", y_d=1.0, source="Lab book").source == (
        ParameterSource("Lab book")
    )
    assert ReferenceRadiation("c", "C", y_d=1.0, source=None).source == ParameterSource()
    assert register_reference_radiation("d", y_d=1.0, site_diameter=2.0).details() == (
        "y_D = 1 keV/µm in a 2 µm site"
    )


def test_kase_2006_hsg_set_reproduces_its_200kv_alpha(reference_radiations):
    # Kase et al., Radiat. Res. 166 (2006) 629, pp. 632-633: HSG alpha_X = 0.19 Gy^-1 and
    # beta = 0.05 Gy^-2 for 200 kVp X-rays of y_D = 4.3 +- 0.2 keV/um (Coppola et al., TEPC at
    # 0.76 and 0.97 um), alpha_C = 0.35 Gy^-1 for 290 MeV/u carbon ions of y_D = 15.6 keV/um.
    alpha_x, alpha_c, beta, y_x, y_c = 0.19, 0.35, 0.05, 4.3, 15.6
    # Eqs. (11) and (12): r_d = 0.42 +- 0.04 um and alpha_0 = 0.13 +- 0.03 Gy^-1, which takes
    # z_1D = y_D / (rho pi r_d^2) with 0.1602 Gy per keV/um^3 (Eq. 3)
    r_d = math.sqrt(beta * (y_c - y_x) * GY_PER_KEV_PER_UM3 / (math.pi * (alpha_c - alpha_x)))
    assert r_d == pytest.approx(0.42, abs=0.005)
    assert alpha_x - (alpha_c - alpha_x) / (y_c - y_x) * y_x == pytest.approx(0.13, abs=0.005)
    # Their set converts back to their alpha_X, for their y_D as for the default 4.2 keV/um
    register_reference_radiation("kase_2006", "200 kVp X-rays", y_d=y_x, site_diameter=1.0)
    for radiation in ("kase_2006", DEFAULT_REFERENCE_RADIATION):
        kase = MKMTissueProperties(
            model="mMKM",
            alpha_0=0.13,
            beta_0=beta,
            r_domain=0.42,
            r_nucleus=4.1,
            reference_radiation=radiation,
        )
        assert kase.alpha_x == pytest.approx(alpha_x, rel=0.015)
    # Eq. (9): y_0 = rho pi r_d R_n^2 / sqrt(beta (r_d^2 + R_n^2)) = 150 keV/um for R_n = 4.1 um
    z_0 = saturation_specific_energy(0.42, 4.1, beta)
    assert z_0 * math.pi * 0.42**2 / GY_PER_KEV_PER_UM3 == pytest.approx(150, rel=0.01)


# ----------------------------------------------------------------- per-parameter sources
def test_overriding_alpha_x_alone_keeps_it_and_derives_beta_x():
    tissue = MKMTissueProperties(**MKM_BASELINE, alpha_x=0.4)
    assert (tissue.alpha_x, tissue.beta_x) == (0.4, 0.0615)
    assert tissue.reference_sources == {"alpha_x": "explicit", "beta_x": "estimated"}
    assert tissue.reference_source == "estimated"
    # A round trip keeps the override and the derivation of the other
    restored = TissueProperties.from_dict({**tissue.model_dump(by_alias=True), "beta0": 0.07})
    assert (restored.alpha_x, restored.beta_x) == (0.4, 0.07)
    assert restored.reference_sources == tissue.reference_sources


@pytest.mark.parametrize("key", ["reference_source", "referenceSource"])
def test_files_with_a_single_reference_source_load(key):
    dump = MKMTissueProperties(**MKM_BASELINE).model_dump()
    del dump["reference_sources"]
    estimated = MKMTissueProperties.model_validate({**dump, key: "estimated", "alpha_x": 0.9})
    assert estimated.alpha_x == pytest.approx(0.172 + 0.0615 * Z_1D, rel=1e-7)
    assert estimated.reference_sources == {"alpha_x": "estimated", "beta_x": "estimated"}
    explicit = TissueProperties.from_dict({**dump, key: "explicit", "alpha_x": 0.9})
    assert explicit.alpha_x == 0.9 and explicit.reference_source == "explicit"
    # Given besides the mapping of a dump, the single source overrides it
    merged = {**MKMTissueProperties(**MKM_BASELINE).model_dump(), "alpha_x": 0.9, key: "explicit"}
    assert MKMTissueProperties.model_validate(merged).alpha_x == 0.9


def test_published_reference_parameters_are_not_derived_again():
    tissue = MKMTissueProperties(
        **MKM_BASELINE, alpha_x=0.313, beta_x=0.0615, reference_sources="published"
    )
    tissue.r_domain = 0.4
    tissue.beta_0 = 0.07
    assert (tissue.alpha_x, tissue.beta_x, tissue.reference_source) == (0.313, 0.0615, "published")
    assert declared_reference_parameters(tissue) == {"alpha_x": 0.313, "beta_x": 0.0615}
    # Switching to derived values derives them from the current model parameters
    tissue.reference_source = "estimated"
    assert tissue.reference_sources == {"alpha_x": "estimated", "beta_x": "estimated"}
    assert tissue.beta_x == 0.07
