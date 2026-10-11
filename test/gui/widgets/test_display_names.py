import pytest
from PySide6.QtWidgets import QComboBox

from pyRadPlan.bio_models import available_bio_models
from pyRadPlan.gui.widgets._display_names import (
    IDENTIFIER_ROLE,
    add_identified_item,
    bio_model_display_name,
    group_result_keys,
    quantity_display_name,
    result_key_labels,
    show_identifiers_when_open,
)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("physical_dose", "dose"),
        ("rbe_x_dose", "RBExDose"),
        ("let", "LETd"),
        ("alpha_dose_beam", "AlphaDose beam"),
        ("physical_dose_beam 2", "dose beam 2"),
        ("alpha", "alpha"),  # unregistered result keys are kept
        ("physicalDose", "physicalDose"),
    ],
)
def test_quantity_display_name(key, expected):
    assert quantity_display_name(key) == expected


def _summary(keys):
    return [(g.label, [c for c, _ in g.components]) for g in group_result_keys(keys)]


def test_group_result_keys_joins_beams_with_their_total():
    keys = [
        "physical_dose",
        "physical_dose_beam 0",
        "physical_dose_beam 1",
        "alpha",
        "alpha_beam 0",
    ]
    assert _summary(keys) == [
        ("dose", ["full", "beam 0", "beam 1"]),
        ("alpha", ["full", "beam 0"]),
    ]


def test_group_result_keys_splits_snapshot_tags():
    current = ["physical_dose", "physical_dose_beam 0", "alpha", "physical_dose_var"]
    saved = [f"{k}_protons_30fx" for k in ("physical_dose", "alpha", "physical_dose_var")]
    saved.append("physical_dose_beam_protons_30fx 0")
    groups = group_result_keys(current + saved)

    by_label = {g.label: g for g in groups}
    assert by_label["dose (protons_30fx)"].keys == (
        "physical_dose_protons_30fx",
        "physical_dose_beam_protons_30fx 0",
    )
    assert by_label["alpha (protons_30fx)"].tag == "protons_30fx"
    # a single suffixed entry is not mistaken for a snapshot tag
    assert "physical_dose_var" in by_label
    assert "physical_dose_var (protons_30fx)" in by_label


def test_result_key_labels():
    labels = result_key_labels(["rbe_x_dose", "rbe_x_dose_beam 0", "imported"])
    assert labels == {
        "rbe_x_dose": "RBExDose",
        "rbe_x_dose_beam 0": "RBExDose, beam 0",
        "imported": "imported",
    }


@pytest.mark.parametrize("tag", ["dose", "x_dose", "dose_beam"])
def test_snapshot_suffix_does_not_split_live_quantity_names(tag):
    bases = ["physical_dose", "rbe_x_dose", "alpha_dose", "sqrt_beta_dose"]
    current = [key for base in bases for key in (base, f"{base}_beam 0")]
    saved = [key for base in bases for key in (f"{base}_{tag}", f"{base}_beam_{tag} 0")]

    groups = {(g.base, g.tag): g for g in group_result_keys(current + saved)}
    assert len(groups) == 2 * len(bases)
    for base in bases:
        assert groups[base, ""].keys == (base, f"{base}_beam 0")
        assert groups[base, ""].label == quantity_display_name(base)
        assert groups[base, tag].keys == (f"{base}_{tag}", f"{base}_beam_{tag} 0")
        assert groups[base, tag].label == f"{quantity_display_name(base)} ({tag})"


def test_live_quantities_do_not_count_as_snapshot_evidence():
    keys = ["physical_dose", "rbe_x_dose", "physical_dose_dose"]
    assert all(group.tag == "" for group in group_result_keys(keys))


def test_bio_models_have_display_names():
    for cls in available_bio_models("protons"):
        assert bio_model_display_name(cls)
    names = {cls.model: bio_model_display_name(cls) for cls in available_bio_models("helium")}
    assert names["HEL"] == "Mairani (helium)"


def test_identifier_only_widens_the_popup(qapp):
    combo = QComboBox()
    show_identifiers_when_open(combo)
    add_identified_item(combo, "dose", "physical_dose", "physical_dose")

    assert combo.currentText() == "dose"
    assert combo.currentData() == "physical_dose"
    assert combo.itemData(0, IDENTIFIER_ROLE) == "physical_dose"
    # Native combo delegates use platform-specific sizing. Compare the same
    # delegate and style option, changing only the identifier metadata.
    option = _option(combo)
    index = combo.model().index(0, 0)
    widened = combo.itemDelegate().sizeHint(option, index)
    combo.setItemData(0, None, IDENTIFIER_ROLE)
    unchanged = combo.itemDelegate().sizeHint(option, index)
    assert widened.width() > unchanged.width()
    assert widened.height() == unchanged.height()
    assert combo.currentText() == "dose"


def _option(combo):
    from PySide6.QtWidgets import QStyleOptionViewItem

    option = QStyleOptionViewItem()
    option.initFrom(combo.view())
    return option
