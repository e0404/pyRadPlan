"""GUI configuration for the particle cutoff placement override."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QCheckBox

from pyRadPlan.dose.engines import ParticleHongPencilBeamEngine
from pyRadPlan.gui.widgets import ConfigFormWidget


def test_particle_cutoff_numpy_override_is_a_checkbox(qapp):
    """The engine's placement override is exposed through the generated boolean editor."""
    form = ConfigFormWidget(ParticleHongPencilBeamEngine.config_model())
    editor = form._editors["cutoff_force_numpy"]
    assert isinstance(editor, QCheckBox)
    assert not editor.isChecked()
    assert "cutoff_backend" not in form._editors
    form._set_value("cutoff_force_numpy", True)
    assert form.values()["cutoff_force_numpy"] is True


def test_particle_batch_configuration_round_trip(qapp):
    """The generated form exposes opt-in packet budgets without changing defaults."""
    model = ParticleHongPencilBeamEngine.config_model()
    form = ConfigFormWidget(model)
    assert model.model_validate(form.values()).batch_size == 0
    form._set_value("batch_size", 64)
    form._set_value("batch_selection_max_entries", 32768)
    form._set_value("batch_evaluation_max_entries", 65536)
    values = model.model_validate(form.values())
    assert values.batch_size == 64
    assert values.batch_selection_max_entries == 32768
    assert values.batch_evaluation_max_entries == 65536
    assert values.batch_max_bytes is None
    form._set_value("batch_max_bytes", 1048576)
    assert model.model_validate(form.values()).batch_max_bytes == 1048576
