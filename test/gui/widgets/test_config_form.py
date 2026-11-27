import pytest

pytest.importorskip("PySide6")

import dataclasses
import threading
import time
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


@dataclasses.dataclass(frozen=True)
class _OptionsWithMapping:
    extra: dict = dataclasses.field(default_factory=lambda: {"x": 1})


class _MappingAlgorithm(ConfigurableAlgorithm):
    options: _OptionsWithMapping = _OptionsWithMapping()


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


# ----------------------------------------------------------------------
# Pydantic models with nested models (e.g. scenario models)
# ----------------------------------------------------------------------


def test_scenario_model_form_edits_nested_dimensions(qapp):
    from pyRadPlan.scenarios import WorstCaseScenarioModel

    form = ConfigFormWidget(
        WorstCaseScenarioModel, initial={"wc_sigma": 2.0, "setup": {"sd": (1.0, 1.0, 1.0)}}
    )
    assert form._editors["wc_sigma"].value() == 2.0
    assert form._editors["setup.sd"].text() == "1 1 1"
    assert isinstance(form._editors["combinations"], QComboBox)
    assert isinstance(form._editors["range.combined"], QCheckBox)
    assert isinstance(form._editors["distribution"], QLineEdit)
    assert form._editors["distribution"].text() == "independent_gaussian"
    assert form._editors["range.rel_sd"].toolTip() == "Relative range SD in percent."
    assert form.values() == {"wc_sigma": 2.0, "setup": {"sd": (1.0, 1.0, 1.0)}}

    form._editors["range.rel_sd"].setValue(0.0)
    form._editors["setup.sd"].setText("2 3 4")
    form._editors["setup.sd"].editingFinished.emit()
    assert form.values() == {
        "wc_sigma": 2.0,
        "setup": {"sd": (2.0, 3.0, 4.0)},
        "range": {"rel_sd": 0.0},
    }
    assert form.config().setup.sd == (2.0, 3.0, 4.0)
    assert form.config().range.rel_sd == 0.0

    model = WorstCaseScenarioModel.model_validate(form.values())
    assert model.setup.sd == (2.0, 3.0, 4.0)
    assert model.range.rel_sd == 0.0
    assert model.setup.shift_mean == (0.0, 0.0, 0.0)


def test_scenario_model_form_rejects_invalid_nested_values(qapp):
    from pyRadPlan.scenarios import WorstCaseScenarioModel

    form = ConfigFormWidget(WorstCaseScenarioModel)
    failures = []
    form.validation_failed.connect(lambda name, msg: failures.append(name))
    form._set_nested("setup", ("sd",), (1.0, 1.0))  # three directions are required
    form._set_nested("ct_phase", ("probabilities",), [0.5, 0.5])  # one phase only
    assert failures == ["setup.sd", "ct_phase.probabilities"]
    assert form.values() == {}
    assert form.config().setup.sd == (2.25, 2.25, 2.25)


def test_companion_field_edits_stay_pending_until_consistent(qapp):
    from pyRadPlan.scenarios import WorstCaseScenarioModel

    dialog = ConfigFormDialog(WorstCaseScenarioModel)
    form = dialog.form

    # two more phases are rejected until their probabilities follow
    form._set_nested("ct_phase", ("phases",), [0, 1])
    assert "ct_phase" not in dialog.values()
    assert "red" in form._editors["ct_phase.phases"].styleSheet()

    form._set_nested("ct_phase", ("probabilities",), [0.5, 0.5])
    assert dialog.values()["ct_phase"] == {"phases": [0, 1], "probabilities": [0.5, 0.5]}
    assert form._editors["ct_phase.phases"].styleSheet() == ""


def test_mapping_option_of_a_nested_option_can_be_edited(qapp):
    form = ConfigFormWidget(_MappingAlgorithm.config_model())
    form._set_nested("options", ("extra",), {"y": 2})
    assert form.config().options.extra == {"y": 2}
    assert form.values()["options"]["extra"] == {"y": 2}


# ---------------------------------------------------------------------------
# Suggestions looked up after the dialog opened (ConfigFormDialog.load_choices)
# ---------------------------------------------------------------------------


class _SuggestingAlgorithm(ConfigurableAlgorithm):
    model: Optional[str] = None
    label: str = "demo"
    iterations: int = 10


class _SlowLookup:
    """``config_choices`` stand-in that blocks until released and records its thread."""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.release = threading.Event()
        self.done = threading.Event()
        self.threads = []

    def __call__(self, radiation_mode):
        self.threads.append(threading.current_thread())
        try:
            if not self.release.wait(10):
                raise TimeoutError("never released")
            if self.error is not None:
                raise self.error
            return self.result
        finally:
            self.done.set()


def _pump(qapp, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    return predicate()


def _items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def test_dialog_opens_before_the_suggestions_arrive(qapp):
    lookup = _SlowLookup({"model": ["org/a", "org/b"]})
    dialog = ConfigFormDialog(_SuggestingAlgorithm.config_model())
    loaded = []
    dialog.choices_loaded.connect(loaded.append)

    dialog.load_choices(lookup, "protons")
    assert not lookup.done.is_set()  # returned while the lookup still blocks
    assert _pump(qapp, lambda: bool(lookup.threads))
    assert lookup.threads[0] is not threading.main_thread()
    assert isinstance(dialog.form._editors["model"], QLineEdit)

    lookup.release.set()
    assert _pump(qapp, lambda: bool(loaded))
    combo = dialog.form._editors["model"]
    assert isinstance(combo, QComboBox) and combo.isEditable()
    assert _items(combo) == ["", "org/a", "org/b"]
    assert combo.currentText() == ""
    assert loaded == [{"model": ["org/a", "org/b"]}]
    assert dialog.values() == {}


def test_late_suggestions_keep_what_the_user_typed(qapp):
    lookup = _SlowLookup({"model": ["org/a", "org/b"], "label": ["x"], "iterations": [1]})
    dialog = ConfigFormDialog(_SuggestingAlgorithm.config_model())
    loaded = []
    dialog.choices_loaded.connect(loaded.append)
    dialog.load_choices(lookup, None)
    dialog.form._editors["model"].setText("org/mine")  # typed, editing not finished

    lookup.release.set()
    assert _pump(qapp, lambda: bool(loaded))
    form = dialog.form
    assert form._editors["model"].currentText() == "org/mine"
    assert form._editors["label"].currentText() == "demo"
    assert isinstance(form._editors["iterations"], QSpinBox)  # not a text field
    assert loaded == [{"model": ["org/a", "org/b"], "label": ["x"]}]
    assert dialog.values() == {"model": "org/mine"}

    # merging into a field that already has suggestions adds only the new ones
    assert form.set_choices("model", ["org/b", "org/c", "org/c"])
    assert _items(form._editors["model"]) == ["", "org/a", "org/b", "org/c"]
    assert form._editors["model"].currentText() == "org/mine"
    assert dialog.values() == {"model": "org/mine"}
    form._editors["model"].setCurrentText("org/c")
    assert dialog.values() == {"model": "org/c"}


def test_suggestions_given_upfront_are_merged_too(qapp):
    dialog = ConfigFormDialog(
        _SuggestingAlgorithm.config_model(),
        initial={"model": "org/b"},
        choices={"model": ["org/a", "org/b"]},
    )
    form = dialog.form
    assert form.set_choices("model", ["org/c", "org/a"])
    assert _items(form._editors["model"]) == ["", "org/a", "org/b", "org/c"]
    assert form._editors["model"].currentText() == "org/b"
    assert not form.set_choices("unknown", ["x"])
    assert dialog.values() == {"model": "org/b"}


def test_closing_the_dialog_before_the_suggestions_arrive_drops_them(qapp):
    lookup = _SlowLookup({"model": ["org/a"]})
    dialog = ConfigFormDialog(_SuggestingAlgorithm.config_model())
    loaded = []
    dialog.choices_loaded.connect(loaded.append)
    dialog.load_choices(lookup, None)
    dialog.reject()

    lookup.release.set()
    assert lookup.done.wait(5)
    _pump(qapp, lambda: False, timeout=0.2)
    assert loaded == []
    assert isinstance(dialog.form._editors["model"], QLineEdit)


def test_deleting_the_dialog_before_the_suggestions_arrive_is_safe(qapp):
    lookup = _SlowLookup({"model": ["org/a"]})
    dialog = ConfigFormDialog(_SuggestingAlgorithm.config_model())
    dialog.load_choices(lookup, None)
    dialog.deleteLater()
    _pump(qapp, lambda: False, timeout=0.1)
    del dialog

    lookup.release.set()
    assert lookup.done.wait(5)
    _pump(qapp, lambda: False, timeout=0.2)


def test_a_failing_lookup_leaves_the_form_as_it_is(qapp):
    lookup = _SlowLookup(error=RuntimeError("hub down"))
    dialog = ConfigFormDialog(_SuggestingAlgorithm.config_model())
    loaded = []
    dialog.choices_loaded.connect(loaded.append)
    dialog.load_choices(lookup, None)
    dialog.form._editors["model"].setText("org/mine")

    lookup.release.set()
    assert lookup.done.wait(5)
    _pump(qapp, lambda: False, timeout=0.2)
    assert loaded == []
    editor = dialog.form._editors["model"]
    assert isinstance(editor, QLineEdit) and editor.text() == "org/mine"
