pyRadPlan settings
==================

Global configuration read from ``PYRADPLAN_*`` environment variables or
``.env`` files: a per-user one at ``<data_dir>/.env`` (``~/.pyradplan/.env``
unless ``PYRADPLAN_DATA_DIR`` relocates the data directory) and one in the
working directory, which takes precedence. The runtime configuration is held by
the mutable singleton ``pyRadPlan.settings``.

.. autoclass:: pyRadPlan.PyRadPlanSettings
   :members:
   :show-inheritance:

.. autoclass:: pyRadPlan.XpSettings
   :members:
   :show-inheritance:

The desktop GUI has three presentation modes, selected with ``settings.gui.ui_mode``
(``PYRADPLAN_GUI_UI_MODE``) or in the GUI under **File > UI Mode**: ``"default"`` shows every
control, ``"educational"`` reduces the widgets to the basic operations for teaching, and
``"agentic"`` adds an *Agent* tab in which the planning agent creates the plans (see
:ref:`concept_agentic_ui_mode`).

.. autoclass:: pyRadPlan.GuiSettings
   :members:
   :show-inheritance:

.. autoclass:: pyRadPlan._settings.AiSettings
   :members:
   :show-inheritance:
