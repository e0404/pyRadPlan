"""Window for DVH and QI analysis."""

from __future__ import annotations

from typing import Any
import weakref

import numpy as np
import SimpleITK as sitk

from PySide6.QtWidgets import QMainWindow, QWidget

from pyRadPlan.gui.widgets._analysis_widget import AnalysisWidget

#: Open analysis windows, kept alive because they are deliberately top-level
#: (see :class:`AnalysisWindow`) and therefore have no Qt parent to own them.
_OPEN_WINDOWS: set[AnalysisWindow] = set()


class AnalysisWindow(QMainWindow):
    """Main window for displaying DVH plots and QI tables.

    The window is deliberately created **without** a Qt parent. On Windows a
    parented window gets a native owner and is then never given its own taskbar
    button, which makes it practically unreachable once other applications are
    in front of it. Setting ``Qt.Window`` does not change that - only the
    absence of an owner does. Since there is no parent to own it, the window
    keeps itself alive in :data:`_OPEN_WINDOWS` until it is closed, and no
    parent closes it either. The supplied ``parent`` is recorded as a logical
    owner only; its host window must call :func:`close_analysis_windows` with
    that owner when closing, so unrelated analyses remain open.
    """

    def __init__(
        self,
        quantities: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
        parent: QWidget | None = None,
        voi_colors: dict[str, tuple[int, int, int]] | None = None,
        overlay_units: dict[str, str] | None = None,
        overlay_labels: dict[str, str] | None = None,
        initial_quantity: str = "",
        initial_vois: list[str] | None = None,
        voi_types: dict[str, str] | None = None,
    ) -> None:
        # Keep ownership separate from Qt parenting to preserve the taskbar entry.
        super().__init__()
        self._owner_ref = weakref.ref(parent) if parent is not None else None
        self.setWindowTitle("DVH / QI Analysis")
        self.resize(1000, 750)

        self.widget = AnalysisWidget(self)
        self.setCentralWidget(self.widget)

        self.widget.set_data(
            quantities=quantities,
            masks=masks,
            voi_colors=voi_colors,
            overlay_units=overlay_units,
            overlay_labels=overlay_labels,
            initial_quantity=initial_quantity,
            initial_vois=initial_vois,
            voi_types=voi_types,
        )

        _OPEN_WINDOWS.add(self)

    def closeEvent(self, event) -> None:  # noqa: N802  (Qt override)
        """Drop the self-reference so the closed window can be collected."""
        _OPEN_WINDOWS.discard(self)
        super().closeEvent(event)


def close_analysis_windows(owner: QWidget) -> None:
    """Close analyses opened with *owner* as their logical parent widget."""
    for window in list(_OPEN_WINDOWS):
        if window._owner_ref is not None and window._owner_ref() is owner:
            window.close()


def close_all_analysis_windows() -> None:
    """Close every open analysis window (they have no parent to do it for them)."""
    for window in list(_OPEN_WINDOWS):
        window.close()


def show_analysis(
    quantities: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    parent: Any = None,
    overlay: dict | None = None,
    overlay_units: dict[str, str] | None = None,
    overlay_labels: dict[str, str] | None = None,
    initial_quantity: str = "",
    initial_vois: list[str] | None = None,
    voi_types: dict[str, str] | None = None,
) -> Any:
    """Create and show an :class:`AnalysisWindow` for the given data.

    Parameters
    ----------
    quantities:
        Mapping of quantity name → 3-D numpy array (e.g. ``{"Dose": arr}``).
    masks:
        Mapping of VOI name → boolean/uint8 mask matching the quantity shape.
    parent:
        Optional logical owner widget, recorded without setting a Qt parent.
        Pass this same widget to :func:`close_analysis_windows` when its host closes.
    overlay:
        Display configuration dict. Recognised keys:

        * ``"voi_colors"`` - ``dict[str, tuple[int,int,int]]`` RGB 0-255 per VOI.
        * ``"units"`` / ``"unit"`` - fallback unit string (used when
          *overlay_units* is not provided).
        * ``"labels"`` / ``"label"`` - fallback label string.

    overlay_units:
        Per-quantity unit strings. Takes precedence over ``overlay["unit"]``.
    overlay_labels:
        Per-quantity label strings. Takes precedence over ``overlay["label"]``.
    initial_quantity:
        Quantity name to preselect in the primary quantity dropdown.
    initial_vois:
        VOI names to pre-check. Defaults to all available masks.

    Returns
    -------
        The :class:`AnalysisWindow` instance, or ``None`` if no data.
    """
    if not quantities or masks is None:
        return None

    cfg = overlay or {}
    voi_colors: dict[str, tuple[int, int, int]] | None = cfg.get("voi_colors")

    # Resolve overlay_units / overlay_labels — prefer explicit params over cfg fallback
    if overlay_units is None:
        fallback_unit = cfg.get("units", cfg.get("unit", ""))
        overlay_units = {q: fallback_unit for q in quantities} if fallback_unit else {}

    if overlay_labels is None:
        fallback_label = cfg.get("labels", cfg.get("label", ""))
        overlay_labels = {q: fallback_label for q in quantities} if fallback_label else {}

    # Normalize masks to a plain dict[str, np.ndarray]
    plain_masks: dict[str, np.ndarray] = {}
    if isinstance(masks, dict):
        if "vois" in masks and isinstance(masks["vois"], list):
            for voi in masks["vois"]:
                try:
                    plain_masks[voi["name"]] = voi["mask"]
                except (KeyError, TypeError):
                    pass
        else:
            plain_masks = {k: v for k, v in masks.items() if isinstance(v, np.ndarray)}
    else:
        # StructureSet-like object
        for voi in getattr(masks, "vois", []):
            try:
                mask_arr = (
                    sitk.GetArrayFromImage(voi.mask)
                    if isinstance(voi.mask, sitk.Image)
                    else np.asarray(voi.mask)
                )
                plain_masks[voi.name] = mask_arr
            except Exception:
                pass

    if not plain_masks:
        return None

    window = AnalysisWindow(
        quantities=quantities,
        masks=plain_masks,
        parent=parent,
        voi_colors=voi_colors,
        overlay_units=overlay_units,
        overlay_labels=overlay_labels,
        initial_quantity=initial_quantity,
        initial_vois=initial_vois,
        voi_types=voi_types,
    )
    window.show()
    return window
