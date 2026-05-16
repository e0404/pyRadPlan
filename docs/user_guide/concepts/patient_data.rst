.. _concept_patient_data:

CT and Structure Set
====================

Patient data in pyRadPlan is represented by two objects: the CT image (``ct``) and the
structure set (``cst``). Both are pydantic models that enforce data integrity and expose a
``to_matrad()`` method for interoperability.

Loading patient data
--------------------

The most common entry point is :func:`~pyRadPlan.load_patient`, which reads a matRad-format
``*.mat`` file and returns a validated ``(CT, StructureSet)`` pair:

.. code-block:: python

    from importlib import resources
    from pyRadPlan import load_patient

    tg119_path = resources.files("pyRadPlan.data.phantoms").joinpath("TG119.mat")
    ct, cst = load_patient(tg119_path)

The CT object
-------------

:class:`~pyRadPlan.ct.CT` stores the CT image together with its spatial metadata.
Internally the image is held as a `SimpleITK <https://simpleitk.readthedocs.io>`_ image, which
provides sub-millimetre-accurate coordinate transforms and easy resampling.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Attribute
     - Description
   * - ``cube_hu``
     - :class:`sitk.Image` in Hounsfield Units. NumPy arrays are automatically converted.
   * - ``resolution``
     - Voxel spacing as ``{"x": ..., "y": ..., "z": ...}`` in mm.
   * - ``size``
     - Grid dimensions ``(nx, ny, nz)`` in voxels.
   * - ``origin``
     - World coordinates of the image origin in mm (LPS convention).
   * - ``direction``
     - Direction cosines matrix (3×3) describing the image axes in world space.

.. code-block:: python

    import numpy as np

    print(ct.resolution)          # {'x': 3.0, 'y': 3.0, 'z': 3.0}
    print(ct.size)                # (nx, ny, nz)
    hu_array = sitk.GetArrayFromImage(ct.cube_hu)  # (nz, ny, nx) NumPy array

The StructureSet object
-----------------------

:class:`~pyRadPlan.cst.StructureSet` holds a list of
:class:`~pyRadPlan.cst.VOI` (Volume of Interest) objects and keeps a reference to the
associated CT image so that spatial queries always use consistent coordinates.

.. code-block:: python

    print(len(cst.vois))          # number of structures
    for voi in cst.vois:
        print(voi.name, voi.voi_type)

Volumes of Interest (VOI)
--------------------------

Each :class:`~pyRadPlan.cst.VOI` carries:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Attribute
     - Description
   * - ``name``
     - Structure name (e.g. ``"PTV"``, ``"Spinal_Cord"``).
   * - ``voi_type``
     - One of ``"TARGET"``, ``"OAR"``, ``"HELPER"``, ``"EXTERNAL"``.
   * - ``mask``
     - Boolean :class:`sitk.Image` (1 = inside structure, 0 = outside) on the CT grid.
   * - ``indices``
     - Flattened voxel indices (Fortran/column-major order) for efficient sparse lookups.
   * - ``alpha_x``
     - Linear-quadratic α parameter (Gy⁻¹, default 0.1). Used for biological modelling.
   * - ``beta_x``
     - Linear-quadratic β parameter (Gy⁻², default 0.05).
   * - ``overlap_priority``
     - Determines which VOI "wins" in overlapping regions (lower = higher priority).
   * - ``objectives``
     - List of optimization objectives attached to this structure
       (see :ref:`concept_optimization`).

Attaching optimization objectives
----------------------------------

Objectives are stored directly on each VOI and are automatically picked up during optimization.
They can be added programmatically:

.. code-block:: python

    from pyRadPlan.optimization.objectives import SquaredDeviation, MaxDVH

    # Add a squared-deviation objective to the PTV
    ptv = next(v for v in cst.vois if v.voi_type == "TARGET")
    ptv.objectives.append(
        SquaredDeviation(priority=1000, d_ref=60.0, quantity="physical_dose")
    )

    # Penalize dose above 45 Gy in more than 2% of the spinal cord volume
    cord = next(v for v in cst.vois if v.name == "Spinal_Cord")
    cord.objectives.append(
        MaxDVH(priority=100, d=45.0, v_max=2.0)
    )

matRad interoperability
-----------------------

.. code-block:: python

    matrad_ct  = ct.to_matrad()   # dict with 'cubeHU', 'resolution', etc.
    matrad_cst = cst.to_matrad()  # cell array compatible with matRad's cst
