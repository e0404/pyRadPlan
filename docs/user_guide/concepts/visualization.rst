.. _concept_visualization:

Visualization
=============

pyRadPlan provides lightweight helpers for inspecting CT images, structure contours, and dose
distributions. They are deliberately thin wrappers around Matplotlib so that the returned
``Axes`` objects can be further customized with standard Matplotlib calls.

plot_slice
----------

:func:`~pyRadPlan.visualization.plot_slice` is the primary function. It renders a single
cross-sectional slice of the CT image, optionally overlaying a dose (or other quantity)
distribution and structure contours.

.. code-block:: python

    from pyRadPlan import plot_slice

    plot_slice(
        image_volume=ct,
        cst=cst,
        overlay=result["physical_dose"],
        overlay_unit="Gy",
    )

Key parameters
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``image_volume``
     - CT image (:class:`~pyRadPlan.ct.CT`, dict, or :class:`sitk.Image`). Displayed in
       grayscale with HU windowing.
   * - ``cst``
     - Structure set (:class:`~pyRadPlan.cst.StructureSet` or matRad-style list). Contours
       are drawn for all VOIs.
   * - ``overlay``
     - Dose or quantity array to display as a color overlay. Can be a NumPy array or
       :class:`sitk.Image` on either the dose grid or the CT grid.
   * - ``overlay_unit``
     - Physical unit string (e.g. ``"Gy"``, ``"Gy(RBE)"``). Used for the colorbar label and
       automatic unit conversion via `pint <https://pint.readthedocs.io>`_.
   * - ``plane``
     - Slice orientation: ``"axial"`` (default), ``"coronal"``, or ``"sagittal"``.
       Can also be an integer axis index.
   * - ``view_slice``
     - Index of the slice to display. If ``None``, the middle slice is chosen.
   * - ``overlay_alpha``
     - Transparency of the dose overlay (0–1, default 0.5).
   * - ``overlay_rel_threshold``
     - Voxels below this fraction of the maximum overlay value are not shown (default 0.02).
   * - ``overlay_window``
     - ``(min, max)`` display range for the overlay. Clips the colormap to this interval.
   * - ``overlay_cmap``
     - Matplotlib colormap name for the overlay (default ``"jet"``).
   * - ``image_window`` / ``ct_window``
     - ``(min, max)`` HU window for the CT background.
   * - ``contour_line_width``
     - Line width for structure contours (default 2.0).
   * - ``save_filename``
     - If given, the figure is saved to this path instead of displayed interactively.
   * - ``ax``
     - An existing :class:`matplotlib.axes.Axes` to draw into. Useful for multi-panel figures.

.. code-block:: python

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, plane in zip(axes, ["axial", "coronal", "sagittal"]):
        plot_slice(
            image_volume=ct,
            cst=cst,
            overlay=result["physical_dose"],
            overlay_unit="Gy",
            plane=plane,
            show_plot=False,
            ax=ax,
        )

    plt.tight_layout()
    plt.show()

plot_multiple_slices
---------------------

:func:`~pyRadPlan.visualization.plot_multiple_slices` creates a multi-panel figure showing
one or more overlays across one or more slices. The function accepts the same ``image_volume``
and ``cst`` arguments as :func:`~pyRadPlan.visualization.plot_slice`, but uses ``overlays`` and
``view_slice`` for the displayed data:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``overlays``
     - One overlay image or a list of overlay images.
   * - ``view_slice``
     - One slice index or a list of slice indices.
   * - ``plane``
     - Orientation shared by all panels.

.. code-block:: python

    from pyRadPlan.visualization import plot_multiple_slices

    plot_multiple_slices(
        image_volume=ct,
        cst=cst,
        overlays=result["physical_dose"],
        overlay_unit="Gy",
        view_slice=[40, 50, 60],
        plane="axial",
    )

Tips
----

- Pass ``show_plot=False`` to suppress the interactive window (useful in scripts or notebooks
  that call ``plt.savefig()`` manually).
- ``overlay`` can be any quantity array returned by ``dij.compute_result_ct_grid()``, not just
  ``"physical_dose"`` — e.g. ``result["rbe_x_dose"]`` or ``result["let_dose"]``.
- Structure colors are assigned automatically but can be customized by setting ``voi.color``
  on individual VOI objects.
