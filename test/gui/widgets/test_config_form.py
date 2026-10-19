import pytest

pytest.importorskip("PySide6")

import dataclasses
from typing import Annotated, Callable, Literal, Optional, Union

from pydantic import Field

from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit, QSpinBox

from pyRadPlan.core import AlgorithmParameterMetadata, ConfigurableAlgorithm
from pyRadPlan.gui.widgets import ConfigFormDialog, ConfigFormWidget


class _DemoAlgorithm(ConfigurableAlgorithm):
    threshold: Annotated[float, Field(gt=0.0, le=100.0, description="A threshold")] = 1.0
    iterations: Annotated[int, Field(ge=1, le=50)] = 10
    enabled: bool = True
    mode: Literal["fast", "accurate"] = "fast"
    label: str = "demo"
    angles: list[float] = [0.0, 90.0]
    hidden: Annotated[Optional[str], AlgorithmParameterMetadata(configurable=False)] = None
    unmapped: Optional[Union[dict, str]] = None  # multi-union -> no editor


@pytest.fixture
def form(qapp):
    return ConfigFormWidget(_DemoAlgorithm.config_model())


def test_editor_mapping(form):
    assert isinstance(form._editors["threshold"].__class__.__bases__[0], type)
    assert isinstance(form._editors["iterations"], QSpinBox)
    assert isinstance(form._editors["enabled"], QCheckBox)
    assert isinstance(form._editors["mode"], QComboBox)
    assert isinstance(form._editors["label"], QLineEdit)
    assert isinstance(form._editors["angles"], QLineEdit)
    assert "hidden" not in form._editors  # configurable=False
    # multi-union has no dedicated widget -> generic JSON fallback editor
    assert isinstance(form._editors["unmapped"], QLineEdit)


def test_constraints_applied_to_widgets(form):
    spin = form._editors["iterations"]
    assert spin.minimum() == 1
    assert spin.maximum() == 50
    assert form._editors["threshold"].toolTip() == "A threshold"


def test_values_only_contains_touched_fields(form):
    assert form.values() == {}
    form._editors["iterations"].setValue(20)
    assert form.values() == {"iterations": 20}


def test_initial_values_round_trip(qapp):
    form = ConfigFormWidget(
        _DemoAlgorithm.config_model(),
        initial={"threshold": 5.0, "mode": "accurate"},
    )
    assert form._editors["threshold"].value() == 5.0
    assert form._editors["mode"].currentText() == "accurate"
    assert form.values() == {"threshold": 5.0, "mode": "accurate"}


def test_invalid_initial_values_dropped(qapp):
    form = ConfigFormWidget(
        _DemoAlgorithm.config_model(),
        initial={"threshold": -3.0, "iterations": 20},
    )
    assert "threshold" not in form.values()
    assert form.values()["iterations"] == 20


def test_invalid_edit_flags_error_without_crash(form):
    failures = []
    form.validation_failed.connect(lambda name, msg: failures.append(name))
    form._set_value("threshold", -1.0)
    assert failures == ["threshold"]
    assert "threshold" not in form.values()


def test_list_editor_parses_text(form):
    edit = form._editors["angles"]
    edit.setText("0, 45 90")
    edit.editingFinished.emit()
    assert form.values()["angles"] == [0.0, 45.0, 90.0]


def test_choice_editor_updates_draft(form):
    form._editors["mode"].setCurrentText("accurate")
    assert form.values()["mode"] == "accurate"


# ----------------------------------------------------------------------
# ConfigFormDialog
# ----------------------------------------------------------------------


def test_dialog_wraps_form_and_returns_values(qapp):
    dialog = ConfigFormDialog(
        _DemoAlgorithm.config_model(),
        initial={"threshold": 5.0},
        title="Configure Demo",
    )
    assert dialog.windowTitle() == "Configure Demo"
    dialog.form._set_value("iterations", 25)
    assert dialog.values() == {"threshold": 5.0, "iterations": 25}


# ----------------------------------------------------------------------
# Defaults, choice unions and dataclass-valued fields
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Inner:
    method: Literal["cg", "minres"] = "cg"
    rtol: float = 0.25

    def __post_init__(self):
        if self.rtol <= 0:
            raise ValueError("rtol must be positive")


@dataclasses.dataclass(frozen=True)
class _Options:
    inner: _Inner = dataclasses.field(default_factory=_Inner)
    max_iter: int = 100
    tol: Optional[float] = None


class _NestedAlgorithm(ConfigurableAlgorithm):
    options: Annotated[_Options, AlgorithmParameterMetadata(hidden_entries=("max_iter",))] = (
        _Options()
    )
    method: Union[Literal["a", "b"], str, Callable] = "a"


def test_defaults_are_shown_but_not_reported(qapp):
    form = ConfigFormWidget(_DemoAlgorithm.config_model(), defaults={"iterations": 7})
    assert form._editors["iterations"].value() == 7
    assert form.values() == {}
    form._editors["iterations"].setValue(8)
    assert form.values() == {"iterations": 8}


def test_union_with_literal_gets_an_editable_choice(qapp):
    form = ConfigFormWidget(_NestedAlgorithm.config_model())
    combo = form._editors["method"]
    assert isinstance(combo, QComboBox)
    assert combo.isEditable()
    assert [combo.itemText(i) for i in range(combo.count())] == ["a", "b"]
    combo.setCurrentText("b")
    assert form.values() == {"method": "b"}


def test_dataclass_field_edits_are_reported_as_nested_overrides(qapp):
    form = ConfigFormWidget(
        _NestedAlgorithm.config_model(), initial={"options": {"inner": {"rtol": 0.5}}}
    )
    assert "options.max_iter" not in form._editors  # hidden entry
    assert form._editors["options.inner.rtol"].value() == 0.5
    assert form.values() == {"options": {"inner": {"rtol": 0.5}}}

    form._editors["options.inner.method"].setCurrentText("minres")
    assert form.values() == {"options": {"inner": {"rtol": 0.5, "method": "minres"}}}


def test_dataclass_field_untouched_is_not_reported(qapp):
    form = ConfigFormWidget(_NestedAlgorithm.config_model())
    assert form._editors["options.inner.method"].currentText() == "cg"
    assert form.values() == {}


def test_unset_optional_number_gets_a_blank_editor(qapp):
    form = ConfigFormWidget(_NestedAlgorithm.config_model())
    edit = form._editors["options.tol"]
    assert isinstance(edit, QLineEdit)
    assert edit.text() == ""
    edit.setText("1e-3")
    edit.editingFinished.emit()
    assert form.values() == {"options": {"tol": 1e-3}}


def test_dataclass_instance_given_initially_is_reported_as_instance(qapp):
    initial = _Options(max_iter=5)
    form = ConfigFormWidget(_NestedAlgorithm.config_model(), initial={"options": initial})
    form._editors["options.inner.method"].setCurrentText("minres")
    assert form.values()["options"] == _Options(max_iter=5, inner=_Inner(method="minres"))


def test_nested_edits_update_the_validated_config(qapp):
    form = ConfigFormWidget(
        _NestedAlgorithm.config_model(), initial={"options": {"inner": {"rtol": 0.5}}}
    )
    assert form.config().options.inner.rtol == 0.5
    form._set_nested("options", ("inner", "method"), "minres")
    assert form.config().options.inner == _Inner(method="minres", rtol=0.5)


@pytest.mark.parametrize("initial", [{}, {"options": _Options()}])
@pytest.mark.parametrize("path,value", [(("inner", "rtol"), -1.0), (("inner", "method"), "bad")])
def test_invalid_nested_edits_keep_the_previous_config(qapp, initial, path, value):
    form = ConfigFormWidget(_NestedAlgorithm.config_model(), initial=initial)
    failures, changes = [], []
    form.validation_failed.connect(lambda name, msg: failures.append(name))
    form.value_changed.connect(lambda *args: changes.append(args))
    previous = form.values()
    form._set_nested("options", path, value)
    assert failures == [".".join(("options", *path))]
    assert not changes
    assert form.values() == previous
    assert form.config().options == _Options()


def test_invalid_initial_nested_override_is_dropped(qapp):
    form = ConfigFormWidget(
        _NestedAlgorithm.config_model(), initial={"options": {"inner": {"method": "bad"}}}
    )
    assert form.values() == {}
    assert form.config().options == _Options()
