import pytest

pytest.importorskip("PySide6")

from typing import Annotated, Literal, Optional, Union

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
