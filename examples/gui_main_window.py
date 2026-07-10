# %% [markdown]
"""# pyRadPlan main window reproducing the matRad MainGUI layout.

Launches the full :class:`~pyRadPlan.gui.windows._main_win.MainWindow`: a fixed
three-column layout (Workflow/Plan/Objectives/Visualization on the left, the
Slice Viewer in the centre under the logo banner, and Viewer Options/Structure
Visibility/Info on the right), all bound to one
:class:`~pyRadPlan.gui.workspace.WorkspaceManager`.

The same window is launched by ``pyRadPlan.gui.gui()``.
"""

# %%
import sys

from PySide6.QtWidgets import QApplication

from pyRadPlan import load_tg119, PhotonPlan
from pyRadPlan.gui.workspace import WorkspaceManager
from pyRadPlan.gui.windows._main_win import MainWindow

# %%
app = QApplication.instance() or QApplication(sys.argv)

# Shared workspace, pre-loaded with the TG119 phantom and a default photon plan.
ws = WorkspaceManager.instance()
ct, cst = load_tg119()
ws.set_many(ct=ct, cst=cst, pln=PhotonPlan())

win = MainWindow(ws)
win.resize(1500, 850)
win.show()

# %%
if __name__ == "__main__":
    sys.exit(app.exec())
