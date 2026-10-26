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
