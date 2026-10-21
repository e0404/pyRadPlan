import math

import pytest

from pyRadPlan.biology.tissues import (
    CELL_LINE_PRESETS,
    LEM_VERSIONS,
    SOURCE_SENTINELS,
    UNPUBLISHED_SOURCE,
    UNVERIFIED_SOURCE,
    CellLinePreset,
    LEMTissueProperties,
    MKMTissueProperties,
    ParameterSource,
    TissueProperties,
    get_cell_line_preset,
    get_provenance,
    is_verified_source,
    register_cell_line_preset,
    register_provenance,
    registered_cell_line_presets,
    registered_provenances,
    tissue_provenance,
)


def _lem(**overrides) -> dict:
    return {"alpha_x": 0.1, "beta_x": 0.05, "d_t": 30.0, "r_nucleus": 5.0, **overrides}


def test_lem_version_defaults_to_the_first():
    tissue = LEMTissueProperties(**_lem())
    assert tissue.version == "I"
    assert tissue.model == "LEM"
    assert tissue.display_name == "LEM I"


def test_lem_version_accepts_numerals_and_numbers():
    assert LEMTissueProperties(version="III").version == "III"
    assert LEMTissueProperties(version=3).version == "III"
    assert LEMTissueProperties(version="3").version == "III"
    with pytest.raises(ValueError):
        LEMTissueProperties(version="V")


def test_lem_version_taken_off_the_model_name():
    tissue = LEMTissueProperties(model="LEM_II", **_lem())
    assert tissue.version == "II"
    # The version lives in its own field, the model name stays the plain model
    assert tissue.model == "LEM"


def test_from_dict_builds_the_lem_class_with_its_version():
    tissue = TissueProperties.from_dict(
        dict(model="LEM_III", alphaX=0.228, betaX=0.02, Dt=55.0, rNucleus=4.7)
    )
    assert type(tissue) is LEMTissueProperties
    assert tissue.version == "III"
    assert tissue.d_t == pytest.approx(55.0)

    # A bare "LEM" names LEM I
    assert TissueProperties.from_dict(dict(model="LEM")).version == "I"
    assert type(TissueProperties.from_dict(dict(model="MKM"))) is MKMTissueProperties


def test_from_dict_builds_every_version_off_its_model_name():
    for numeral in LEM_VERSIONS:
        tissue = TissueProperties.from_dict(dict(model=f"LEM_{numeral}", **_lem()))
        assert type(tissue) is LEMTissueProperties
        assert tissue.version == numeral
        assert tissue.model == "LEM"


def test_from_dict_prefers_an_explicit_version_over_the_bare_model_alias():
    """A separate `version` entry must win even though "LEM" alone would default to I."""
    tissue = TissueProperties.from_dict(dict(model="LEM", version="III", **_lem()))
    assert type(tissue) is LEMTissueProperties
    assert tissue.version == "III"
    assert tissue.model == "LEM"


def test_version_labels_the_provenance_of_a_lem_tissue_class():
    assert "version" not in LEMTissueProperties.identifying_fields
    assert LEMTissueProperties.provenance_fields == ("model", "version")
    assert MKMTissueProperties.provenance_fields == ("model",)


def test_lem_tissues_of_different_versions_match_by_their_parameters():
    """The version is a provenance label, not a matching criterion."""
    lem_i = LEMTissueProperties(version="I", **_lem())
    lem_ii = LEMTissueProperties(version="II", **_lem())

    assert lem_i.matches(lem_ii)
    assert lem_ii.matches(lem_i)
    assert not lem_i.matches(LEMTissueProperties(version="II", **_lem(d_t=6.0)))
    assert not lem_i.same_provenance(lem_ii)
    # Equality stays strict about the provenance
    assert lem_i != lem_ii


def test_provenance_labels_are_compared_only_when_asked():
    lem_i = LEMTissueProperties(version="I", **_lem())
    lem_ii = LEMTissueProperties(version="II", **_lem())
    assert lem_i.matches(lem_ii, fields=["d_t", "r_nucleus"])
    assert not lem_i.matches(lem_ii, fields=["d_t", "version"])

    mkm = MKMTissueProperties(r_nucleus=3.9, r_domain=0.32)
    mmkm = MKMTissueProperties(model="mMKM", r_nucleus=3.9, r_domain=0.32)
    assert mkm.matches(mmkm) and mmkm.matches(mkm)
    assert not mkm.matches(mmkm, fields=["model", "r_domain"])
    assert mkm != mmkm


def test_reference_only_tissue_stays_compatible_with_a_lem_version():
    reference = TissueProperties(alpha_x=0.1, beta_x=0.05)
    lem = LEMTissueProperties(version="II", **_lem())
    assert reference.matches(lem)
    assert lem.matches(reference)


def test_lem_and_mkm_never_match():
    lem = LEMTissueProperties(version="II", **_lem())
    mkm = MKMTissueProperties(alpha_x=0.1, beta_x=0.05, r_nucleus=5.0, r_domain=0.32)
    assert not lem.matches(mkm)
    assert not mkm.matches(lem)


def _expected_alpha_x(alpha_0: float, beta_0: float, r_domain: float) -> float:
    z_1d = 4.2 / (math.pi * r_domain**2) * 0.1602  # 200 kV X-rays, y_D = 4.2 keV/µm
    return alpha_0 + beta_0 * z_1d


def test_mkm_reference_parameters_estimated_from_the_baseline_ones():
    tissue = MKMTissueProperties(alpha_0=0.313, beta_0=0.0615, r_domain=0.34, r_nucleus=4.1)
    assert tissue.alpha_x == pytest.approx(_expected_alpha_x(0.313, 0.0615, 0.34))
    assert tissue.beta_x == pytest.approx(0.0615)

    # Aliases are recognized as "given" just as the field names are
    aliased = MKMTissueProperties(alpha0=0.228, beta0=0.02, rDomain=0.2)
    assert aliased.alpha_x == pytest.approx(_expected_alpha_x(0.228, 0.02, 0.2))
    assert aliased.beta_x == pytest.approx(0.02)


def test_mkm_reference_parameters_given_explicitly_are_kept():
    tissue = MKMTissueProperties(
        alpha_x=0.2, beta_x=0.01, alpha_0=0.313, beta_0=0.0615, r_domain=0.34
    )
    assert tissue.alpha_x == pytest.approx(0.2)
    assert tissue.beta_x == pytest.approx(0.01)

    aliased = MKMTissueProperties(
        alphaX=0.2, betaX=0.01, alpha_0=0.313, beta_0=0.0615, rDomain=0.34
    )
    assert aliased.alpha_x == pytest.approx(0.2)
    assert aliased.beta_x == pytest.approx(0.01)


def test_mkm_reference_parameters_fall_back_to_the_generic_defaults():
    # Nothing to estimate from, so the defaults of the base class stand instead of NaN
    tissue = MKMTissueProperties()
    assert tissue.alpha_x == pytest.approx(TissueProperties().alpha_x)
    assert tissue.beta_x == pytest.approx(TissueProperties().beta_x)

    # beta_x is estimable from beta_0 alone, alpha_x is not
    partial = MKMTissueProperties(beta_0=0.03)
    assert partial.alpha_x == pytest.approx(TissueProperties().alpha_x)
    assert partial.beta_x == pytest.approx(0.03)

    # A zero domain radius would divide by zero rather than describe a domain
    degenerate = MKMTissueProperties(alpha_0=0.313, beta_0=0.0615, r_domain=0.0)
    assert degenerate.alpha_x == pytest.approx(TissueProperties().alpha_x)


def test_mkm_estimate_reference_parameters_reports_only_what_it_can():
    assert MKMTissueProperties.estimate_reference_parameters() == {}
    assert set(MKMTissueProperties.estimate_reference_parameters(beta_0=0.02)) == {"beta_x"}
    assert set(
        MKMTissueProperties.estimate_reference_parameters(alpha_0=0.2, beta_0=0.02, r_domain=0.3)
    ) == {"alpha_x", "beta_x"}


def test_mkm_estimated_reference_parameters_are_logged(caplog):
    with caplog.at_level("INFO", logger="pyRadPlan.biology.tissues._base"):
        MKMTissueProperties(cell_line="HSG", alpha_0=0.313, beta_0=0.0615, r_domain=0.34)
    assert "alpha_x" in caplog.text
    assert "beta_x" in caplog.text
    assert "HSG" in caplog.text


def test_presets_carry_their_version():
    tissue = TissueProperties.from_preset("HSG", "LEM_II")
    assert type(tissue) is LEMTissueProperties
    assert tissue.version == "II"
    assert tissue.cell_line == "HSG"
    assert tissue.d_t == pytest.approx(6.0)
    assert not math.isnan(tissue.r_nucleus)

    # The alias resolves to the first version
    assert TissueProperties.from_preset("HSG", "LEM").version == "I"


def test_lookup_matches_by_parameters_and_prefers_the_same_version():
    candidates = [
        LEMTissueProperties(version="I", **_lem()),
        LEMTissueProperties(version="II", **_lem(d_t=6.0)),
        LEMTissueProperties(version="III", **_lem()),
    ]
    (_source, match) = TissueProperties.lookup(
        [LEMTissueProperties(version="II", **_lem(d_t=6.0))], candidates
    )[0]
    assert match is candidates[1]

    # Another version matches by the parameters; one of the tissue's own is preferred
    (_source, match) = TissueProperties.lookup(
        [LEMTissueProperties(version="III", **_lem())], candidates
    )[0]
    assert match is candidates[2]
    (_source, match) = TissueProperties.lookup(
        [LEMTissueProperties(version="IV", **_lem())], candidates
    )[0]
    assert match is candidates[0]

    (_source, match) = TissueProperties.lookup(
        [LEMTissueProperties(version="I", **_lem(d_t=99.0))], candidates
    )[0]
    assert match is None


def test_equality_treats_unset_parameters_as_equal():
    a = LEMTissueProperties(alpha_x=0.1, beta_x=0.05, d_t=float("nan"))
    b = LEMTissueProperties(alpha_x=0.1, beta_x=0.05, d_t=float("nan"))
    assert a is not b
    assert a == b
    assert a in [b]
    assert a != LEMTissueProperties(alpha_x=0.1, beta_x=0.05, d_t=6.0)
    assert a != LEMTissueProperties(alpha_x=0.1, beta_x=0.05, d_t=float("nan"), version="II")


def test_equality_requires_the_same_tissue_class():
    reference = TissueProperties(alpha_x=0.1, beta_x=0.05)
    assert reference != LEMTissueProperties(alpha_x=0.1, beta_x=0.05)
    assert reference != MKMTissueProperties(alpha_x=0.1, beta_x=0.05)
    assert reference != "not a tissue"


def test_tissue_properties_are_unhashable():
    with pytest.raises(TypeError):
        hash(LEMTissueProperties())


@pytest.mark.parametrize(
    "tissue",
    [
        TissueProperties(alpha_x=0.2, beta_x=0.03, cell_line="custom"),
        TissueProperties.from_preset("HSG", "LEM_II"),
        TissueProperties.from_preset("CHO", "LEM_III"),
        TissueProperties.from_preset("HSG", "MKM"),
        LEMTissueProperties(alpha_x=0.1, beta_x=0.05, version="IV"),
    ],
    ids=["reference", "lem_ii", "lem_iii", "mkm", "lem_iv_nan"],
)
def test_dump_round_trips_through_from_dict(tissue):
    for dumped in (tissue.model_dump(), tissue.model_dump(by_alias=True)):
        restored = TissueProperties.from_dict(dumped)
        assert type(restored) is type(tissue)
        assert restored == tissue


def test_json_dump_round_trips_unset_parameters():
    tissue = LEMTissueProperties(alpha_x=0.1, beta_x=0.05, version="II")
    restored = LEMTissueProperties.model_validate_json(tissue.model_dump_json())
    assert math.isnan(restored.d_t)
    assert restored == tissue


@pytest.mark.parametrize(
    ("model", "version", "provenance", "tissue_type", "label"),
    [
        ("LEM", "LEM_II", "LEM_II", LEMTissueProperties, "LEM II"),
        ("LEM", "III", "LEM_III", LEMTissueProperties, "LEM III"),
        ("LEM", None, "LEM_I", LEMTissueProperties, "LEM I"),
        ("LEMIV", None, "LEM_IV", LEMTissueProperties, "LEM IV"),
        ("MKM", None, "MKM", MKMTissueProperties, "MKM"),
        ("mMKM", None, "mMKM", MKMTissueProperties, "mMKM"),
        (None, None, "undeclared", TissueProperties, "model not declared"),
        ("Generic", None, "undeclared", TissueProperties, "model not declared"),
    ],
)
def test_provenance_registry(model, version, provenance, tissue_type, label):
    resolved = get_provenance(model, version)
    assert (resolved.name, resolved.tissue_type, resolved.label) == (
        provenance,
        tissue_type,
        label,
    )

    data = {"alphaX": 0.2, "betaX": 0.05}
    if model is not None:
        data["model"] = model
    if version is not None:
        data["version"] = version
    tissue = TissueProperties.from_dict(data)
    assert type(tissue) is tissue_type
    assert tissue.provenance == provenance


def test_unknown_provenance_is_described_by_reference_parameters():
    resolved = get_provenance("RBEmodelX")
    assert not resolved.registered and resolved.declared
    tissue = TissueProperties.from_dict({"model": "RBEmodelX", "alphaX": 0.3})
    assert type(tissue) is TissueProperties
    assert tissue.provenance == "RBEmodelX"


def test_registering_a_provenance_routes_from_dict(monkeypatch):
    from pyRadPlan.biology.tissues import _provenance

    monkeypatch.setattr(_provenance, "_PROVENANCES", dict(_provenance._PROVENANCES))
    monkeypatch.setattr(_provenance, "_ALIASES", dict(_provenance._ALIASES))
    register_provenance(
        "MKM_Inaniwa", MKMTissueProperties, parameters={"model": "MKM_Inaniwa"}, family="MKM"
    )

    tissue = TissueProperties.from_dict({"model": "mkm_inaniwa", "rDomain": 0.3})
    assert type(tissue) is MKMTissueProperties
    assert tissue.model == "MKM_Inaniwa"
    assert tissue_provenance(tissue) == "MKM_Inaniwa"
    assert "MKM_Inaniwa" in [p.name for p in registered_provenances()]


def test_mat_metadata_extras_are_ignored_by_from_dict():
    tissue = TissueProperties.from_dict(
        {"model": "LEM", "alphaX": 0.1, "betaX": 0.05, "ref_Kernel": "alpha_beta", "column": 1}
    )
    assert isinstance(tissue, LEMTissueProperties)
    assert not hasattr(tissue, "ref_Kernel")


def test_undeclared_class_parameters_are_not_compared_when_asked():
    """Base data may leave parameters undeclared (NaN); only matching may skip them."""
    lem_iv_class = LEMTissueProperties(alpha_x=0.1, beta_x=0.05, version="IV")
    tissue = LEMTissueProperties(alpha_x=0.1, beta_x=0.05, version="IV", d_t=30.0, r_nucleus=5.0)

    assert not tissue.matches(lem_iv_class)
    assert tissue.matches(lem_iv_class, ignore_undeclared=True)
    assert tissue != lem_iv_class
    # Declared parameters are still compared; the version is a label and is not
    assert not tissue.model_copy(update={"alpha_x": 0.2}).matches(
        lem_iv_class, ignore_undeclared=True
    )
    assert tissue.model_copy(update={"version": "I"}).matches(lem_iv_class, ignore_undeclared=True)
    # Only the class side may be undeclared
    assert not lem_iv_class.matches(tissue, ignore_undeclared=True)


def test_ignoring_undeclared_parameters_never_matches_vacuously():
    mkm_class = MKMTissueProperties.from_base_data(
        alpha_x=math.nan, beta_x=math.nan, r_nucleus=3.9, r_domain=0.32
    )
    assert not TissueProperties(alpha_x=0.1, beta_x=0.05).matches(
        mkm_class, ignore_undeclared=True
    )
    pairs = TissueProperties.lookup(
        [LEMTissueProperties(alpha_x=0.1, beta_x=0.05, version="IV", d_t=30.0)],
        [LEMTissueProperties(alpha_x=0.1, beta_x=0.05, version="IV")],
        ignore_undeclared=True,
    )
    assert pairs[0][1] is not None


# ----------------------------------------------------------------- cell line presets
@pytest.fixture
def restore_presets():
    saved = dict(CELL_LINE_PRESETS)
    yield
    CELL_LINE_PRESETS.clear()
    CELL_LINE_PRESETS.update(saved)


def test_every_preset_carries_a_source():
    assert CELL_LINE_PRESETS
    for key, preset in CELL_LINE_PRESETS.items():
        assert isinstance(preset, CellLinePreset)
        assert preset.key == key
        assert preset.source.citation
        tissue = preset.tissue()
        assert tissue.cell_line_source == preset.source.citation


# Published LEM sets, Manganaro et al. 2018, Table 1 (p. 5): cell line -> alpha_X, beta_X,
# R_N and the threshold doses D_t of LEM I / II / III.
PUBLISHED_LEM_PRESETS = {
    "HSG": (0.313, 0.0615, 5.0, (30, 6, 19)),
    "CHO": (0.228, 0.02, 4.7, (40, 9.5, 55)),
    "V79": (0.184, 0.02, 4.0, (60, 5.5, 60)),
}

# The original publication of each LEM preset (short source, DOI, where in the paper), or
# Manganaro et al. 2018 where none gives the threshold dose.
_MANGANARO = ("Manganaro 2018", "10.1088/1361-6560/aab697", "08NT01, Table 1 (p. 5)")
LEM_PRESET_ORIGINALS = {
    ("HSG", "I"): ("Elsässer 2007", "10.1667/RR0467.1", "Fig. 11 (p. 327) and p. 326"),
    ("HSG", "II"): ("Elsässer 2007", "10.1667/RR0467.1", "Fig. 11 (p. 327) and p. 326"),
    ("HSG", "III"): _MANGANARO,
    ("CHO", "I"): ("Elsässer 2007", "10.1667/RR0467.1", "Fig. 8 and p. 325"),
    ("CHO", "II"): ("Elsässer 2007", "10.1667/RR0467.1", "Fig. 8 and p. 325"),
    ("CHO", "III"): _MANGANARO,
    ("V79", "I"): _MANGANARO,
    ("V79", "II"): ("Kase 2008", "10.1088/0031-9155/53/1/003", "Sect. 3.4 (pp. 52-53)"),
    ("V79", "III"): _MANGANARO,
}


@pytest.mark.parametrize("cell_line", list(PUBLISHED_LEM_PRESETS))
def test_published_lem_presets(cell_line):
    alpha_x, beta_x, r_nucleus, thresholds = PUBLISHED_LEM_PRESETS[cell_line]
    for version, d_t in zip(("I", "II", "III"), thresholds):
        preset = get_cell_line_preset(cell_line, f"LEM_{version}")
        short, doi, location = LEM_PRESET_ORIGINALS[cell_line, version]
        assert preset.source.verified and preset.source.short == short
        assert doi in preset.source.citation and location in preset.source.citation
        # Manganaro et al. 2018 tabulate every set, as the secondary source if not the only one
        assert "08NT01, Table 1 (p. 5)" in preset.source.citation
        assert "10.1088/1361-6560/aab697" in preset.source.citation
        assert f"{cell_line} · LEM" in preset.source.note
        assert preset.parameters == dict(
            alpha_x=alpha_x, beta_x=beta_x, d_t=d_t, r_nucleus=r_nucleus
        )


def test_lem_thresholds_without_original_say_so():
    for (cell_line, version), (short, *_) in LEM_PRESET_ORIGINALS.items():
        note = get_cell_line_preset(cell_line, f"LEM_{version}").source.note
        assert ("given by Manganaro et al. only" in note) == (short == "Manganaro 2018")
    # The original LEM III fit for HSG (Elsasser et al. 2008, New J. Phys.) has 23 Gy, not 19
    assert "D_t = 23 Gy" in get_cell_line_preset("HSG", "LEM_III").source.note


@pytest.mark.parametrize("cell_line", ["HSG", "V79"])
def test_lem_xray_parameters_are_furusawa_2000(cell_line):
    # Furusawa et al. 2000, Table 1 (p. 487): 200 kVp X-rays, aerobic cells
    preset = get_cell_line_preset(cell_line, "LEM_II")
    assert "Radiat. Res. 154 (2000) 485, Table 1 (p. 487)" in preset.source.note
    assert "200 kVp X-rays" in preset.source.note


def test_builtin_presets_are_all_attributed():
    for preset in CELL_LINE_PRESETS.values():
        assert preset.source.verified, preset.key
        assert preset.source.note, preset.key


# Published MKM / mMKM sets: (cell line, provenance, label) -> DOI, where in the paper, model
# parameters, the reference radiation they refer to and the published X-ray reference
# (alpha_x, beta_x), None for a reference derived by the conversion.
_KASE_2008_DOI = "10.1088/0031-9155/53/1/003"
PUBLISHED_MKM_PRESETS = {
    ("HSG", "MKM", "Okamoto 2011"): (
        "10.1269/jrr.10073",
        "Table 2 and Fig. 4 (p. 79)",
        dict(r_domain=0.23, alpha_0=0.088, beta_0=0.036),
        "xray_200kV_tepc",
        None,
    ),
    ("HSG", "MKM", "Kase 2008"): (
        _KASE_2008_DOI,
        "Table 1 (p. 45)",
        dict(r_nucleus=4.1, r_domain=0.34, alpha_0=0.313, beta_0=0.0615),
        "xray_200kV",
        (0.313, 0.0615),
    ),
    ("HSG", "MKM", "Kase 2006"): (
        "10.1667/RR0536.1",
        "pp. 632-633 and Eqs. (11)-(12)",
        dict(r_nucleus=4.1, r_domain=0.42, alpha_0=0.13, beta_0=0.05),
        "xray_200kV",
        (0.19, 0.05),
    ),
    ("HSG", "mMKM", "Inaniwa 2010, derived reference"): (
        "10.1088/0031-9155/55/22/008",
        "Sect. 2.2 (pp. 6724-6725) and Fig. 3",
        dict(r_nucleus=3.9, r_domain=0.32, alpha_0=0.172, beta_0=0.0615),
        "xray_200kV",
        None,
    ),
    ("HSG", "mMKM", "Inaniwa 2010 / Furusawa 2000 reference"): (
        "10.1088/0031-9155/55/22/008",
        "Sect. 2.2 (pp. 6724-6725) and Fig. 3",
        dict(r_nucleus=3.9, r_domain=0.32, alpha_0=0.172, beta_0=0.0615),
        "xray_200kV",
        (0.313, 0.0615),
    ),
    ("CHO", "MKM", "Kase 2008"): (
        _KASE_2008_DOI,
        "Sect. 3.3 (p. 52)",
        dict(r_nucleus=4.1, r_domain=0.2, alpha_0=0.228, beta_0=0.02),
        "xray_200kV",
        (0.228, 0.02),
    ),
    ("T1", "MKM", "Kase 2008"): (
        _KASE_2008_DOI,
        "Table 1 (p. 45)",
        dict(r_nucleus=3.5, r_domain=0.35, alpha_0=0.0305, beta_0=0.0585),
        "xray_200kV",
        (0.0305, 0.0585),
    ),
    ("V79", "MKM", "Kase 2008"): (
        _KASE_2008_DOI,
        "Table 1 (p. 45)",
        dict(r_nucleus=4.1, r_domain=0.26, alpha_0=0.184, beta_0=0.02),
        "xray_200kV",
        (0.184, 0.02),
    ),
}


def test_every_mkm_preset_is_listed():
    mkm = {key for key in CELL_LINE_PRESETS if key[1] in ("MKM", "mMKM")}
    assert mkm == set(PUBLISHED_MKM_PRESETS)


@pytest.mark.parametrize("key", list(PUBLISHED_MKM_PRESETS))
def test_published_mkm_presets(key):
    doi, location, params, radiation, reference = PUBLISHED_MKM_PRESETS[key]
    preset = get_cell_line_preset(*key)
    assert preset.key == key
    assert preset.source.verified and preset.label == key[2]
    assert doi in preset.source.citation and location in preset.source.citation
    assert preset.source.note
    model_params = {
        k: v
        for k, v in preset.parameters.items()
        if k not in ("reference_radiation", "alpha_x", "beta_x")
    }
    assert model_params == params
    assert preset.reference_radiation == radiation
    tissue = preset.tissue()
    assert (tissue.model, tissue.cell_line, tissue.reference_radiation) == key[1::-1] + (
        radiation,
    )
    for name, value in params.items():
        assert getattr(tissue, name) == value
    derived = tissue.derived_reference_parameters()
    if reference is None:
        assert preset.published_reference == {}
        assert tissue.reference_source == "estimated"
        assert (tissue.alpha_x, tissue.beta_x) == (derived["alpha_x"], params["beta_0"])
    else:
        # The published X-ray values, kept rather than derived again
        assert preset.published_reference == dict(zip(("alpha_x", "beta_x"), reference))
        assert tissue.reference_sources == {"alpha_x": "published", "beta_x": "published"}
        assert (tissue.alpha_x, tissue.beta_x) == reference
        tissue.r_domain *= 1.1
        assert (tissue.alpha_x, tissue.beta_x) == reference
        assert tissue.alpha_x != pytest.approx(derived["alpha_x"], abs=1e-4)
        assert preset.reference_label in ("Furusawa 2000", "Elsässer 2007", "Kase 2006")


def test_mkm_presets_name_their_reference_radiation():
    for (_cell_line, provenance, _label), preset in CELL_LINE_PRESETS.items():
        if provenance in ("MKM", "mMKM"):
            assert preset.reference_radiation is not None
            assert preset.tissue().reference_radiation == preset.reference_radiation
        else:
            assert preset.reference_radiation is None


def test_okamoto_hsg_reproduces_its_200kv_alpha():
    # Okamoto et al. 2011, Table 2: alpha = 0.241 Gy^-1 for 200 kV X-rays with beta_MK fixed,
    # the line of Fig. 4 alpha = 0.0338 y_D + 0.0885 at the TEPC y_D of 4.51 keV/um.
    hsg = TissueProperties.from_preset("HSG", "MKM")
    assert hsg.alpha_x == pytest.approx(0.241, rel=0.02)
    assert hsg.alpha_x == pytest.approx(0.0338 * 4.51 + 0.0885, rel=0.02)
    # z_1D,ref = 0.1602 * 4.51 / (pi * 0.23**2) = 4.347439 Gy
    assert hsg.alpha_x == pytest.approx(0.088 + 0.036 * 4.347439, rel=1e-6)


def test_inaniwa_hsg_matches_the_oxygen_mmkm_geometry():
    hsg = TissueProperties.from_preset("HSG", "mMKM")
    # z*_1D,ref of 200 kV X-rays (y_D = 4.2 keV/um) for r_d = 0.32, R_n = 3.9: 2.0896141 Gy
    assert hsg.alpha_x == pytest.approx(0.172 + 0.0615 * 2.0896141, rel=1e-6)
    assert (hsg.r_nucleus, hsg.r_domain) == (3.9, 0.32)
    # Inaniwa et al. 2010 give no X-ray z_1D; the note states the derived value
    assert f"alpha_x = {hsg.alpha_x:.3f}" in get_cell_line_preset("HSG", "mMKM").source.note


@pytest.mark.parametrize("cell_line", ["HSG", "CHO", "T1", "V79"])
def test_kase_2008_presets_take_alpha_0_for_the_xray_alpha(cell_line):
    # Kase et al. 2008 (p. 45) set alpha_0 / beta to the X-ray alpha / beta, neglecting the
    # z_1D of X-rays. The preset keeps these published values; the conversion would derive
    # alpha_x > alpha_0, as the note states.
    preset = get_cell_line_preset(cell_line, "MKM", "Kase 2008")
    tissue = preset.tissue()
    assert (tissue.alpha_x, tissue.beta_x) == (tissue.alpha_0, tissue.beta_0)
    derived = tissue.derived_reference_parameters()
    assert derived["alpha_x"] > tissue.alpha_0 and derived["beta_x"] == tissue.beta_0
    assert f"would derive alpha_x = {derived['alpha_x']:.3f} Gy" in preset.source.note
    # Manganaro et al. 2018, Table 1, list alpha_0 / beta as the X-ray alpha_X / beta_X
    alpha_x, beta_x, *_ = PUBLISHED_LEM_PRESETS.get(cell_line, (tissue.alpha_0, tissue.beta_0))
    assert (tissue.alpha_0, tissue.beta_0) == (alpha_x, beta_x)
    # Deriving them on request replaces the published values
    tissue.reference_source = "estimated"
    assert tissue.alpha_x == derived["alpha_x"]


def test_furusawa_xray_reference_values():
    # Furusawa et al., Radiat. Res. 154 (2000) 485, Table 1 (p. 487), 200 kVp X-rays
    furusawa = {"HSG": (0.313, 0.0615), "V79": (0.184, 0.0200), "T1": (0.0305, 0.0585)}
    for cell_line, reference in furusawa.items():
        preset = get_cell_line_preset(cell_line, "MKM", "Kase 2008")
        assert preset.reference.short == "Furusawa 2000"
        assert "Table 1 (p. 487)" in preset.reference.citation
        assert tuple(preset.published_reference.values()) == reference
    # CHO: Elsasser and Scholz 2007, Fig. 8
    cho = get_cell_line_preset("CHO", "MKM")
    assert cho.reference_label == "Elsässer 2007" and "Fig. 8" in cho.reference.citation
    assert cho.published_reference == {"alpha_x": 0.228, "beta_x": 0.02}


def test_source_sentinels():
    assert SOURCE_SENTINELS == (UNVERIFIED_SOURCE, UNPUBLISHED_SOURCE)
    assert SOURCE_SENTINELS == ("Unverified", "Unpublished")
    assert not is_verified_source(None) and not is_verified_source("")
    assert not is_verified_source("unpublished")
    assert is_verified_source("personal communication, J. Doe 2024")
    assert ParameterSource.coerce(None).citation == UNVERIFIED_SOURCE
    assert ParameterSource.coerce("Unpublished").label == "unpublished"
    assert ParameterSource("Some et al. 2020", short="Some 2020").label == "Some 2020"


def test_mkm_presets_derive_or_publish_their_reference_parameters():
    for preset in CELL_LINE_PRESETS.values():
        assert set(preset.published_reference) in (set(), {"alpha_x", "beta_x"})
    cho = TissueProperties.from_preset("CHO", "MKM", reference_source="estimated")
    assert cho.reference_source == "estimated"
    # z_1D,ref = 0.1602 * 4.2 / (pi * 0.2**2) = 5.354291 Gy
    assert cho.alpha_x == pytest.approx(0.228 + 0.02 * 5.354291, rel=1e-6)
    assert cho.beta_x == 0.02
    lem = TissueProperties.from_preset("HSG", "LEM_I")
    assert (lem.alpha_x, lem.beta_x, lem.reference_source) == (0.313, 0.0615, "published")
    # Overriding a published value makes it explicit
    override = TissueProperties.from_preset("T1", "MKM", alpha_x=0.05)
    assert override.reference_sources == {"alpha_x": "explicit", "beta_x": "published"}


def test_preset_source_is_metadata():
    preset = TissueProperties.from_preset("HSG", "LEM_II")
    assert preset.cell_line == "HSG"
    assert "Manganaro" in preset.cell_line_source
    assert "cell_line_source" in TissueProperties.metadata_fields
    assert "cell_line_source" not in repr(preset)

    other = preset.model_copy(update={"cell_line_source": "Unpublished"})
    assert other == preset
    assert other.matches(preset) and preset.matches(other)

    dumped = preset.model_dump(by_alias=True)
    assert dumped["cellLineSource"] == preset.cell_line_source
    for restored in (
        TissueProperties.from_dict(dumped),
        LEMTissueProperties.model_validate_json(preset.model_dump_json()),
    ):
        assert restored.cell_line_source == preset.cell_line_source
    # Converting keeps the cell line and its source
    assert preset.adopt_provenance("LEM_IV").cell_line_source == preset.cell_line_source


def test_register_cell_line_preset(restore_presets):
    preset = register_cell_line_preset(
        "MyCells",
        "mMKM",
        source="Lab notebook 12, 2025",
        r_nucleus=3.9,
        r_domain=0.32,
        alpha_0=0.172,
        beta_0=0.0615,
    )
    assert preset.key == ("MYCELLS", "mMKM", "Lab notebook 12, 2025")
    assert CELL_LINE_PRESETS[preset.key] is preset
    tissue = TissueProperties.from_preset("mycells", "mmkm")
    assert (tissue.model, tissue.cell_line, tissue.cell_line_source) == (
        "mMKM",
        "MyCells",
        "Lab notebook 12, 2025",
    )
    assert tissue.reference_source == "estimated"

    unsourced = register_cell_line_preset("Other", "LEM_IV", alpha_x=0.2, beta_x=0.02)
    assert unsourced.source.citation == UNVERIFIED_SOURCE
    assert TissueProperties.from_preset("other", "LEM_IV").version == "IV"

    with pytest.raises(ValueError, match="Unknown provenance"):
        register_cell_line_preset("Bad", "NoSuchModel", alpha_x=0.1)
    with pytest.raises(ValueError):
        register_cell_line_preset("Bad", "LEM_I", alpha_x="not a number")
    with pytest.raises(ValueError, match="Unknown preset"):
        TissueProperties.from_preset("Bad", "LEM_I")


def test_several_presets_per_cell_line_and_provenance(restore_presets):
    first = register_cell_line_preset("Mine", "LEM_I", source="Doe 2020", alpha_x=0.2, beta_x=0.02)
    second = register_cell_line_preset(
        "Mine", "LEM_I", source=ParameterSource("Roe et al. 2021", short="Roe 2021"), d_t=20.0
    )
    assert registered_cell_line_presets("mine", "LEM") == [first, second]
    # The first registered is the default; a source is selected by label or citation
    assert get_cell_line_preset("Mine", "LEM_I") is first
    assert get_cell_line_preset("mine", "LEM", "roe 2021") is second
    assert get_cell_line_preset("Mine", "LEM_I", "Roe et al. 2021") is second
    tissue = TissueProperties.from_preset("Mine", "LEM_I", "Roe 2021")
    assert (tissue.d_t, tissue.cell_line_source) == (20.0, "Roe et al. 2021")
    with pytest.raises(ValueError, match="Available: \\['Doe 2020', 'Roe 2021'\\]"):
        get_cell_line_preset("Mine", "LEM_I", "Nobody 1999")
    # Registering with default=True makes a preset the default
    third = register_cell_line_preset("Mine", "LEM_I", source="Poe 2022", default=True)
    assert registered_cell_line_presets("Mine", "LEM_I") == [third, first, second]
    assert get_cell_line_preset("Mine", "LEM_I") is third
    # Replacing a preset of the same source keeps one entry
    register_cell_line_preset("Mine", "LEM_I", source="Doe 2020", alpha_x=0.3, beta_x=0.03)
    assert len(registered_cell_line_presets("Mine", "LEM_I")) == 3
    # The citation identifies the preset on a tissue, so it must be unique
    with pytest.raises(ValueError, match="distinct citations"):
        register_cell_line_preset(
            "Mine", "LEM_I", source=ParameterSource("Doe 2020", short="Doe again")
        )


@pytest.mark.parametrize(
    "preset", list(CELL_LINE_PRESETS.values()), ids=lambda preset: "-".join(preset.key)
)
def test_serialized_tissue_identifies_its_preset(preset):
    tissue = preset.tissue()
    for restored in (
        TissueProperties.from_dict(tissue.model_dump(by_alias=True)),
        type(tissue).model_validate_json(tissue.model_dump_json()),
    ):
        found = get_cell_line_preset(
            restored.cell_line, restored.provenance, restored.cell_line_source
        )
        assert found is preset
        assert restored.reference_sources == tissue.reference_sources


def test_hsg_mmkm_presets_tell_the_reference_apart():
    derived, published = registered_cell_line_presets("HSG", "mMKM")
    assert get_cell_line_preset("HSG", "mMKM") is derived
    assert derived.label == "Inaniwa 2010, derived reference"
    assert published.label == "Inaniwa 2010 / Furusawa 2000 reference"
    assert derived.source.citation != published.source.citation
    assert "Radiat. Res. 154 (2000) 485-496, Table 1 (p. 487)" in published.source.citation
    # Inaniwa et al. quote 0.331 / 0.0593 for Furusawa et al., whose Table 1 gives 0.313 / 0.0615
    assert "0.331" in published.source.note and "0.313 / 0.0615" in published.source.note
    a, b = derived.tissue(), published.tissue()
    assert a == b.model_copy(update={"alpha_x": a.alpha_x})
    assert (b.alpha_x, b.beta_x, b.reference_source) == (0.313, 0.0615, "published")
    assert (round(a.alpha_x, 3), a.beta_x, a.reference_source) == (0.301, 0.0615, "estimated")


def test_hsg_lem_iii_presets():
    manganaro, elsasser = registered_cell_line_presets("HSG", "LEM_III")
    assert get_cell_line_preset("HSG", "LEM_III") is manganaro
    assert (manganaro.label, manganaro.parameters["d_t"]) == ("Manganaro 2018", 19)
    assert elsasser.label == "Elsässer 2008"
    assert dict(elsasser.parameters) == dict(alpha_x=0.313, beta_x=0.0615, d_t=23, r_nucleus=5.0)
    assert "10.1088/1367-2630/10/7/075005" in elsasser.source.citation
    assert "Fig. 6 and Table 1 (p. 14)" in elsasser.source.citation
    assert "EDR40" in elsasser.source.note


# Alternative published sets: (cell line, provenance, label) -> parameters, DOI, location
ALTERNATIVE_LEM_PRESETS = {
    ("CHO", "LEM_I", "Krämer 2000"): (
        dict(alpha_x=0.228, beta_x=0.02, d_t=30, r_nucleus=5.5),
        "10.1088/0031-9155/45/11/314",
        "Table 1 (p. 3322)",
    ),
    ("V79", "LEM_I", "Elsässer 2007"): (
        dict(alpha_x=0.093, beta_x=0.026, d_t=30, r_nucleus=3.8),
        "10.1667/RR0467.1",
        "Figs. 6 and 8 (pp. 324-325) and p. 323",
    ),
    ("V79", "LEM_II", "Elsässer 2007"): (
        dict(alpha_x=0.093, beta_x=0.026, d_t=7.5, r_nucleus=3.8),
        "10.1667/RR0467.1",
        "Figs. 6 and 8 (pp. 324-325) and p. 323",
    ),
    ("V79", "LEM_III", "Manganaro 2018, 2nd V79 set"): (
        dict(alpha_x=0.093, beta_x=0.026, d_t=30, r_nucleus=3.8),
        "10.1088/1361-6560/aab697",
        "Table 1 (p. 5), second V79 · LEM row",
    ),
}


@pytest.mark.parametrize("key", list(ALTERNATIVE_LEM_PRESETS))
def test_alternative_lem_presets(key):
    params, doi, location = ALTERNATIVE_LEM_PRESETS[key]
    preset = CELL_LINE_PRESETS[key]
    assert get_cell_line_preset(*key[:2]) is not preset
    assert dict(preset.parameters) == params
    assert doi in preset.source.citation and location in preset.source.citation
    tissue = preset.tissue()
    assert tissue.reference_source == "published"
    assert all(getattr(tissue, name) == value for name, value in params.items())


def test_parameter_classification():
    assert TissueProperties.model_parameter_fields == ()
    assert LEMTissueProperties.model_parameter_fields == ("alpha_x", "beta_x", "d_t", "r_nucleus")
    assert MKMTissueProperties.model_parameter_fields == (
        "r_nucleus",
        "r_domain",
        "alpha_0",
        "beta_0",
    )
    for tissue_cls in (TissueProperties, LEMTissueProperties, MKMTissueProperties):
        assert tissue_cls.reference_fields == ("alpha_x", "beta_x")
    assert MKMTissueProperties.derives_reference
    assert not LEMTissueProperties.derives_reference
    # MKM matching ignores the reference parameters
    a = MKMTissueProperties(r_nucleus=3.9, r_domain=0.32, alpha_x=0.1, beta_x=0.05)
    b = MKMTissueProperties(r_nucleus=3.9, r_domain=0.32, alpha_x=0.3, beta_x=0.02)
    assert a.matches(b) and a != b
