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

.. autoclass:: pyRadPlan._settings.AiSettings
   :members:
   :show-inheritance:
