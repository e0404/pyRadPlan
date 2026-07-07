# %% [markdown]
"""# WorkflowWidget as an interactive treatment-planning GUI.

This example shows how to launch the :class:`~pyRadPlan.gui.widgets.WorkflowWidget`
as a standalone window.  It pre-loads the built-in TG119 phantom and a default
photon plan so you can immediately click **Calc. Dose Influence → Optimize**.

The :class:`~pyRadPlan.gui.workspace.WorkspaceManager` is the central data store
that replaces MATLAB's base workspace.  Any other widget (plan editor, dose viewer,
…) that accepts a ``workspace`` argument will share the same live data.
"""

# %%
import sys

from PySide6.QtWidgets import QApplication, QMainWindow

from pyRadPlan import load_tg119, PhotonPlan
from pyRadPlan.gui.workspace import WorkspaceManager
from pyRadPlan.gui.widgets import WorkflowWidget

# %%
app = QApplication.instance() or QApplication(sys.argv)

# -- Shared data store --------------------------------------------------------
ws = WorkspaceManager.instance()

# Pre-load the TG119 phantom and a default photon plan so all workflow steps
# are immediately accessible without manually clicking "Load .mat".
ct, cst = load_tg119()
pln = PhotonPlan()
ws.set_many(ct=ct, cst=cst, pln=pln)

# -- Window -------------------------------------------------------------------
win = QMainWindow()
win.setWindowTitle("pyRadPlan – Workflow")

workflow = WorkflowWidget(workspace=ws, parent=win)
win.setCentralWidget(workflow)
win.resize(560, 280)
win.show()


# %%
# Optional: connect workspace_changed to inspect results in a script context.
def _on_changed(keys: list[str]) -> None:
    print(f"workspace changed: {keys}")
    if "result" in keys and ws.has("result"):
        result = ws.result
        print(f"  result keys: {[k for k in result if k != 'w']}")


ws.workspace_changed.connect(_on_changed)

# %%
if __name__ == "__main__":
    sys.exit(app.exec())
