# %% [markdown]
"""# Unified planning GUI with all widgets sharing one WorkspaceManager.

This example wires the workspace-aware widgets together around a single
:class:`~pyRadPlan.gui.workspace.WorkspaceManager`, the way matRad's main GUI
composes its widgets around the base workspace:

- :class:`WorkflowWidget` — run the pipeline (dose influence, optimise, …)
- :class:`PlanWidget` — edit the ``pln`` (radiation mode, angles, dose grid)
- :class:`OptimizationWidget` — edit per-VOI objectives on the ``cst``
- :class:`ViewingWidget` — slice viewer of the resulting dose

Because every widget binds to the *same* workspace, editing the plan, adding an
objective, or running optimisation in one widget is immediately reflected in the
others through the ``workspace_changed`` signal — no manual ``set_data`` plumbing.
"""

# %%
import sys

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QSplitter,
    QTabWidget,
)
from PySide6.QtCore import Qt

from pyRadPlan import load_tg119, PhotonPlan
from pyRadPlan.gui.workspace import WorkspaceManager
from pyRadPlan.gui.widgets import (
    WorkflowWidget,
    PlanWidget,
    OptimizationWidget,
    ViewingWidget,
)

# %%
app = QApplication.instance() or QApplication(sys.argv)

# One workspace shared by every widget.
ws = WorkspaceManager.instance()

# Pre-load the TG119 phantom and a default photon plan.
ct, cst = load_tg119()
ws.set_many(ct=ct, cst=cst, pln=PhotonPlan())

# %%
# Left: stacked control widgets (each bound to ``ws``).  Right: the viewer.
controls = QTabWidget()
controls.addTab(WorkflowWidget(ws), "Workflow")
controls.addTab(PlanWidget(ws), "Plan")
controls.addTab(OptimizationWidget(ws), "Objectives")

viewer = ViewingWidget(ws)

splitter = QSplitter(Qt.Horizontal)
splitter.addWidget(controls)
splitter.addWidget(viewer)
splitter.setStretchFactor(0, 0)
splitter.setStretchFactor(1, 1)
splitter.setSizes([420, 900])

win = QMainWindow()
win.setWindowTitle("pyRadPlan – Planning")
win.setCentralWidget(splitter)
win.resize(1340, 760)
win.show()

# %%
if __name__ == "__main__":
    sys.exit(app.exec())
