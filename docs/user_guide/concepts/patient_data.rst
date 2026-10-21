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

Importing structures from DICOM
-------------------------------

:func:`~pyRadPlan.load_patient` also accepts a folder of DICOM files, in which case the
structures are read from its ``RTSTRUCT`` and converted from contours into binary masks on
the CT grid.

That conversion has no single correct answer: every tool picks its own rule for voxels whose
centre falls on, or very close to, a contour, and the choice shifts structure boundaries by up
to a voxel. pyRadPlan fills a voxel when its **centre lies inside the contour polygon**,
counting a centre exactly on the boundary as inside, so that a structure narrower than one
voxel does not disappear.

This is validated in two ways:

* against a matRad export of the same patient, which the bundled test data must reproduce
  voxel for voxel;
* against label maps produced by `MITK <https://www.mitk.org/>`_ Workbench's segmentation
  contour-to-mask algorithm. For a 512x512x297 CT with four structures, the two agreed to
  three voxels out of 17.6 million, with two structures voxel-identical.

Structures are read one contour at a time and combined, so a slice carrying several contours
for one structure yields their union. Note that an inner contour is therefore *added* rather
than punched out: ring-shaped structures defined by an outer plus an inner contour are not
currently hollowed.

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
   * - ``tissue_properties``
     - Radiobiological properties of the structure (see `Tissue properties`_).
   * - ``alpha_x``
     - Linear-quadratic α parameter (Gy⁻¹, default 0.1). Mirrors
       ``tissue_properties.alpha_x``.
   * - ``beta_x``
     - Linear-quadratic β parameter (Gy⁻², default 0.05). Mirrors
       ``tissue_properties.beta_x``.
   * - ``overlap_priority``
     - Determines which VOI "wins" in overlapping regions (lower = higher priority).
   * - ``objectives``
     - List of optimization objectives attached to this structure
       (see :ref:`concept_optimization`).

.. _voi_tissue_properties:

Tissue properties
-----------------

``voi.tissue_properties`` describes biological parameters of a tissue that can be used to calculate biological models.
:class:`~pyRadPlan.biology.tissues.TissueProperties` describes these parameters.
The default values are:

.. code-block:: python

    from pyRadPlan.biology.tissues import TissueProperties

    voi.tissue_properties  # TissueProperties(model='Generic', alpha_x=0.1, beta_x=0.05)

A tissue carries two kinds of parameters, which are different things:

- **Model parameters** parameterize one RBE model and are specific to its provenance, e.g.
  the dose threshold ``d_t`` of LEM or the domain radius ``r_domain`` and ``alpha_0`` /
  ``beta_0`` of MKM (``TissueProperties.model_parameter_fields``).
- **Reference parameters** ``alpha_x`` / ``beta_x`` are the LQ parameters of the reference
  radiation (photons) the RBE is taken relative to: RBE-weighted dose and RBE are computed
  against them (``TissueProperties.reference_fields``, see :ref:`reference_parameters`).

Different biological models need different model parameters:

.. list-table::
   :header-rows: 1
   :widths: 25 30 20 25

   * - Class
     - Model parameters
     - Reference parameters
     - Used by
   * - :class:`~pyRadPlan.biology.tissues.TissueProperties`
     - –
     - given
     - reference parameters only (LET-based and constant-RBE models, tabulated tables)
   * - :class:`~pyRadPlan.biology.tissues.LEMTissueProperties`
     - ``alpha_x``, ``beta_x``, ``d_t``, ``r_nucleus``
     - given; LEM takes them as model inputs too
     - α/β kernels of LEM provenance (``"alpha_beta_kernels"``)
   * - :class:`~pyRadPlan.biology.tissues.MKMTissueProperties`
     - ``r_nucleus``, ``r_domain``, ``alpha_0``, ``beta_0``
     - derived from the model parameters (:ref:`mkm_reference_parameters`)
     - z\* kernels (``"zstar_kernels"``) and MKM / mMKM base data

Model parameters left unset stay ``nan``. Assign the class the plan's model expects:

.. code-block:: python

    from pyRadPlan.biology.tissues import MKMTissueProperties

    cell = MKMTissueProperties(
        model="mMKM",
        r_nucleus=3.9,
        r_domain=0.32,
        alpha_0=0.172,
        beta_0=0.0615,
    )  # alpha_x / beta_x derived from these
    for voi in cst.vois:
        voi.tissue_properties = cell

The tissue must be one the model can find in the machine's base data, which is matched on
the fields identifying a tissue class (``alpha_x``, ``beta_x``, ``d_t`` and ``r_nucleus``
for LEM; the ``r_nucleus`` / ``r_domain`` geometry for MKM and mMKM, whose ``alpha_0`` /
``beta_0`` come from the tissue itself). The machine lists the classes it ships:

.. code-block:: python

    from pyRadPlan.machines import load_from_name

    machine = load_from_name("oxygen", "Generic")
    for tissue in machine.tissue_properties:
        print(type(tissue).__name__, tissue.model, tissue.cell_line)

A machine can ship tissue classes of several provenances, e.g. LEM I classes for its α/β
kernels and an mMKM class for its z\* kernels; ``machine.kernel_sets`` tells which classes
belong to which kernel set (see :ref:`biological_base_data`). The tissue's provenance — its
``model`` label (MKM or mMKM) and the LEM ``version`` — is a soft label, not part of the
match: an MKM tissue with the geometry of the mMKM class resolves to it, and a LEM I tissue
to a LEM IV class with the same parameters. The model is then evaluated with the base data's
provenance, and the check and the dose calculation report a warning
(``provenance_differs``) suggesting to relabel the tissue, which is optional (see
:ref:`provenance_as_label`). Parameter classes sharing no identifying parameter, such as a
LEM tissue and the MKM class of the z\* kernels, never match. If no class matches,
:meth:`~pyRadPlan.biology.models.BiologicalModel.check_tissues` reports an error, and so does
the dose calculation before computing anything.

In the GUI both tissue editors — the VOI popup in the viewer and "Set tissue parameters" in
the plan widget — offer the machine's tissue classes to pick from, which keeps the tissue
and the base data consistent. Editing them marks the dose influence as outdated in the
workflow panel, since the biological model derives α/β per voxel from these parameters.
Pressing OK only asks about tissues the model cannot be evaluated for; a model label
differing from the base data's is tinted softly instead, with an optional "Relabel as
<model>" action.

Both editors show the model parameters and the reference parameters in separate groups (in
the plan widget's table, the reference columns come last and are marked "ref."). For MKM and
mMKM the reference group offers a "Reference radiation" selector and names the conversion,
and α_x / β_x are derived live from the model parameters and the selected reference
radiation while you edit them; typing a value overrides them, and "Derive from model"
discards the override again (see :ref:`mkm_reference_parameters`).

.. _cell_line_presets:

Cell line presets and their sources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both editors also offer cell line presets under "Tissue"
(:data:`~pyRadPlan.biology.tissues.CELL_LINE_PRESETS`: HSG, CHO, T1 and V79, for MKM,
mMKM and the LEM variants). Unlike a base data tissue class, a preset describes a whole
tissue, so applying one selects the RBE model and fills every parameter of that model,
including the ones the model does not match on (MKM's ``alpha_0`` / ``beta_0``). An MKM /
mMKM preset also selects the reference radiation its values were published for
(``preset.reference_radiation``, see :ref:`mkm_reference_parameters`). The same presets are
available in scripts through :meth:`~pyRadPlan.biology.tissues.TissueProperties.from_preset`:

.. code-block:: python

    from pyRadPlan.biology.tissues import TissueProperties

    cell = TissueProperties.from_preset("HSG", "LEM_II")
    cell.cell_line, cell.cell_line_source  # ('HSG', "Elsässer and Scholz, ... Fig. 11 (p. 327) ...")

A cell line may have several presets for one provenance, one per publication, e.g. HSG ·
LEM III with the threshold dose of Manganaro et al. (2018) and with that of Elsässer et al.
(2008). They are told apart by the short form of their source, and the first registered is
the default; pass the source to pick another, by its short form or its full citation:

.. code-block:: python

    from pyRadPlan.biology.tissues import get_cell_line_preset, registered_cell_line_presets

    TissueProperties.from_preset("HSG", "LEM_III")                   # D_t = 19 Gy (Manganaro 2018)
    TissueProperties.from_preset("HSG", "LEM_III", "Elsässer 2008")  # D_t = 23 Gy
    [p.label for p in registered_cell_line_presets("HSG", "mMKM")]
    # ['Inaniwa 2010, derived reference', 'Inaniwa 2010 / Furusawa 2000 reference']

    # The preset a tissue was built from, e.g. after loading it from a file
    get_cell_line_preset(cell.cell_line, cell.provenance, cell.cell_line_source)

Every preset carries its source (a :class:`~pyRadPlan.biology.tissues.ParameterSource`):
any string naming where the values come from — a paper, a toolkit, a dataset, a personal
communication — or one of the sentinels ``"Unverified"``
(:data:`~pyRadPlan.biology.tissues.UNVERIFIED_SOURCE`, the default when none is given) and
``"Unpublished"`` (:data:`~pyRadPlan.biology.tissues.UNPUBLISHED_SOURCE`). A tissue built
from a preset records its full citation as ``cell_line_source`` next to ``cell_line``; the
citations of the presets of one cell line and provenance differ, so the two identify the
preset. Like ``reference_sources`` it is descriptive metadata: it is saved with the tissue in
every format (alias ``cellLineSource``), but not compared by ``==`` nor in matching the base
data. The editors list every preset, each labelled with its source in short, e.g. "HSG · LEM
III (Elsässer 2008)", or "(unverified)" for a preset registered without a confirmed source,
and give the full source, with notes, in its tooltip. Every built-in preset has been checked
against the publication it cites; the notes say which values a source confirms and which it
gives alone.

A preset may give the reference parameters ``alpha_x`` / ``beta_x`` as published: the X-ray
LQ parameters measured for the cell line. A tissue built from it records them as
``"published"`` (see :ref:`reference_parameters`): they are kept as they are rather than
derived again, also for MKM / mMKM, whose reference parameters otherwise follow from the model
parameters (see :ref:`mkm_reference_parameters`). The LEM presets give them as the model
inputs they are; of the MKM presets, those of Kase et al. give the X-ray values their fit
refers to.

.. list-table:: Sources of the built-in presets
   :header-rows: 1
   :widths: 22 50 28

   * - Preset
     - Values and source
     - Reference parameters
   * - HSG · MKM (Okamoto 2011), the default
     - ``r_domain`` = 0.23 µm, ``alpha_0`` = 0.088 Gy⁻¹, ``beta_0`` = 0.036 Gy⁻²: Okamoto et
       al., J. Radiat. Res. 52 (2011) 75, Table 2 and Fig. 4 (p. 79)
       (`doi:10.1269/jrr.10073 <https://doi.org/10.1269/jrr.10073>`_), fitted to HSG
       survival after 200 kV X-rays, Co-60 and 6 MV X-rays with β fixed to the mean of the
       free fits. Plain MKM, so no ``r_nucleus``.
     - derived for 200 kV X-rays (TEPC), y_D = 4.51 keV/µm, ``"xray_200kV_tepc"``: α_x =
       0.244 Gy⁻¹, reproducing their α of 0.241 Gy⁻¹ within 2 %
   * - HSG · MKM (Kase 2008)
     - ``r_nucleus`` = 4.1 µm, ``r_domain`` = 0.34 µm, ``alpha_0`` = 0.313 Gy⁻¹, ``beta_0``
       = 0.0615 Gy⁻²: Kase et al., Phys. Med. Biol. 53 (2008) 37, Table 1 (p. 45)
       (`doi:10.1088/0031-9155/53/1/003 <https://doi.org/10.1088/0031-9155/53/1/003>`_),
       fitted to the D10 of Furusawa et al. (2000); the HSG · MKM row of Manganaro et al.
       (2018), Table 1.
     - published α_x = 0.313 Gy⁻¹, β_x = 0.0615 Gy⁻² (Furusawa et al., Radiat. Res. 154
       (2000) 485, Table 1, p. 487, 200 kVp X-rays); derived: 0.427 Gy⁻¹
   * - HSG · MKM (Kase 2006)
     - ``r_nucleus`` = 4.1 µm, ``r_domain`` = 0.42 µm, ``alpha_0`` = 0.13 Gy⁻¹, ``beta_0``
       = 0.05 Gy⁻²: Kase et al., Radiat. Res. 166 (2006) 629, pp. 632-633 and Eqs. (11)-(12)
       (`doi:10.1667/RR0536.1 <https://doi.org/10.1667/RR0536.1>`_), from the α of HSG cells
       for 200 kVp X-rays and 290 MeV/u carbon ions; saturation-corrected by y\* (y_0 =
       150 keV/µm) rather than the mMKM.
     - published α_x = 0.19 Gy⁻¹, β_x = 0.05 Gy⁻² (p. 632, 200 kVp X-rays); derived:
       0.191 Gy⁻¹
   * - HSG · mMKM (Inaniwa 2010, derived reference), the default
     - ``r_nucleus`` = 3.9 µm, ``r_domain`` = 0.32 µm, ``alpha_0`` = 0.172 Gy⁻¹, ``beta_0``
       = 0.0615 Gy⁻²: Inaniwa et al., Phys. Med. Biol. 55 (2010) 6721, Sect. 2.2 and Fig. 3
       (`doi:10.1088/0031-9155/55/22/008 <https://doi.org/10.1088/0031-9155/55/22/008>`_),
       fitted to HSG D10 for He, C and Ne ions (Furusawa et al. 2000); β taken over from Kase
       et al. (2008). The geometry of the generic oxygen machine's mMKM class. The paper gives
       no z_1D or y_D of X-rays. For the cells of 2010 it corrects ``alpha_0`` to
       0.0708 Gy⁻¹ (Sect. 2.5).
     - derived for 200 kV X-rays, y_D = 4.2 keV/µm, ``"xray_200kV"``: α_x = 0.301 Gy⁻¹
   * - HSG · mMKM (Inaniwa 2010 / Furusawa 2000 reference)
     - The model parameters of the preset above.
     - published α_x = 0.313 Gy⁻¹, β_x = 0.0615 Gy⁻² (Furusawa et al. 2000, Table 1).
       Inaniwa et al. (p. 6728) quote α_x = 0.331 Gy⁻¹, β_x = 0.0593 Gy⁻² for these
       measurements, which Table 1 of Furusawa et al. does not give.
   * - CHO · MKM (Kase 2008)
     - ``r_nucleus`` = 4.1 µm, ``r_domain`` = 0.2 µm, ``alpha_0`` = 0.228 Gy⁻¹, ``beta_0``
       = 0.02 Gy⁻²: Kase et al. (2008), Sect. 3.3 (p. 52); also Manganaro et al. (2018).
     - published α_x = 0.228 Gy⁻¹, β_x = 0.02 Gy⁻² (Elsässer and Scholz, Radiat. Res. 167
       (2007) 319, Fig. 8); derived: 0.335 Gy⁻¹
   * - T1 · MKM (Kase 2008)
     - ``r_nucleus`` = 3.5 µm, ``r_domain`` = 0.35 µm, ``alpha_0`` = 0.0305 Gy⁻¹,
       ``beta_0`` = 0.0585 Gy⁻²: Kase et al. (2008), Table 1.
     - published α_x = 0.0305 Gy⁻¹, β_x = 0.0585 Gy⁻² (Furusawa et al. 2000, Table 1);
       derived: 0.133 Gy⁻¹
   * - V79 · MKM (Kase 2008)
     - ``r_nucleus`` = 4.1 µm, ``r_domain`` = 0.26 µm, ``alpha_0`` = 0.184 Gy⁻¹, ``beta_0``
       = 0.02 Gy⁻²: Kase et al. (2008), Table 1.
     - published α_x = 0.184 Gy⁻¹, β_x = 0.0200 Gy⁻² (Furusawa et al. 2000, Table 1);
       derived: 0.247 Gy⁻¹
   * - HSG · LEM I / LEM II (Elsässer 2007)
     - α_X = 0.313 Gy⁻¹, β_X = 0.0615 Gy⁻², ``r_nucleus`` = 5.0 µm, ``d_t`` = 30 / 6 Gy:
       Elsässer and Scholz, Radiat. Res. 167 (2007) 319, Fig. 11 (p. 327) and p. 326
       (`doi:10.1667/RR0467.1 <https://doi.org/10.1667/RR0467.1>`_); also Manganaro et al.,
       Phys. Med. Biol. 63 (2018) 08NT01, Table 1.
     - published (Furusawa et al. 2000, Table 1)
   * - HSG · LEM III (Manganaro 2018), the default
     - α_X, β_X and ``r_nucleus`` as for LEM I / II, ``d_t`` = 19 Gy: Manganaro et al.
       (2018), Table 1; no original publication of this threshold dose was found.
     - published
   * - HSG · LEM III (Elsässer 2008)
     - α_X = 0.313 Gy⁻¹, β_X = 0.0615 Gy⁻², ``r_nucleus`` = 5 µm, ``d_t`` = 23 Gy: Elsässer
       et al., New J. Phys. 10 (2008) 075005, Fig. 6 and Table 1 (p. 14)
       (`doi:10.1088/1367-2630/10/7/075005 <https://doi.org/10.1088/1367-2630/10/7/075005>`_),
       fitted with the track structure of LEM III (EDR40, p. 8).
     - published
   * - CHO · LEM I / LEM II (Elsässer 2007), the default for LEM I
     - α_X = 0.228 Gy⁻¹, β_X = 0.02 Gy⁻², ``r_nucleus`` = 4.7 µm, ``d_t`` = 40 / 9.5 Gy:
       Elsässer and Scholz (2007), Fig. 8 and p. 325; also Manganaro et al. (2018).
     - published
   * - CHO · LEM I (Krämer 2000)
     - α_X = 0.228 Gy⁻¹, β_X = 0.02 Gy⁻², ``r_nucleus`` = 5.5 µm, ``d_t`` (D_cut) = 30 Gy:
       Krämer and Scholz, Phys. Med. Biol. 45 (2000) 3319, Table 1 (p. 3322)
       (`doi:10.1088/0031-9155/45/11/314 <https://doi.org/10.1088/0031-9155/45/11/314>`_).
     - published
   * - CHO · LEM III (Manganaro 2018)
     - As for LEM I / II, ``d_t`` = 55 Gy: Manganaro et al. (2018) only.
     - published
   * - V79 · LEM I / LEM III (Manganaro 2018), the defaults
     - α_X = 0.184 Gy⁻¹, β_X = 0.02 Gy⁻², ``r_nucleus`` = 4 µm, ``d_t`` = 60 / 60 Gy: the
       first V79 row of Manganaro et al. (2018), Table 1; the threshold doses are given by
       Manganaro et al. only.
     - published (Furusawa et al. 2000, Table 1)
   * - V79 · LEM II (Kase 2008), the default
     - The same α_X, β_X and ``r_nucleus``, ``d_t`` = 5.5 Gy: Kase et al. (2008), Sect. 3.4
       (pp. 52-53).
     - published (Furusawa et al. 2000, Table 1)
   * - V79 · LEM I / LEM II (Elsässer 2007)
     - α_X = 0.093 Gy⁻¹, β_X = 0.026 Gy⁻², ``r_nucleus`` = 3.8 µm, ``d_t`` = 30 / 7.5 Gy:
       Elsässer and Scholz (2007), Figs. 6 and 8 (pp. 324-325) and p. 323; the second V79
       row of Manganaro et al. (2018).
     - published
   * - V79 · LEM III (Manganaro 2018, 2nd V79 set)
     - As the set above, ``d_t`` = 30 Gy: the second V79 row of Manganaro et al. (2018);
       no original publication of this threshold dose was found.
     - published

Inaniwa et al., Phys. Med. Biol. 60 (2015) 3271, reuse the HSG mMKM set of 2010 for the
clinical NIRS system with a carbon-ion reference radiation; it is not offered as a preset,
since its reference is no photon radiation the built-in reference radiations describe.

Register further presets, with their source, through
:func:`~pyRadPlan.biology.tissues.register_cell_line_preset`; they show up in both
editors and in :meth:`~pyRadPlan.biology.tissues.TissueProperties.from_preset`. A preset of
a cell line and provenance that already has one is added next to it (``default=True`` makes it
the default), one with the same source replaces it. A model deriving its reference parameters
(MKM) is given its model parameters, and the published reference parameters if there are any:

.. code-block:: python

    from pyRadPlan.biology.tissues import (
        UNPUBLISHED_SOURCE,
        ParameterSource,
        register_cell_line_preset,
    )

    register_cell_line_preset(
        "MyCells",
        "mMKM",
        source=UNPUBLISHED_SOURCE,  # or e.g. "Doe et al., Radiat. Res. 1 (2025) 1"
        r_nucleus=3.9,
        r_domain=0.32,
        alpha_0=0.172,
        beta_0=0.0615,
    )
    TissueProperties.from_preset("MyCells", "mMKM")

    register_cell_line_preset(
        "HSG",
        "LEM_II",
        source=ParameterSource("Doe et al., Radiat. Res. 1 (2025) 1", short="Doe 2025"),
        alpha_x=0.3,
        beta_x=0.06,
        d_t=7.0,
        r_nucleus=5.0,
    )
    TissueProperties.from_preset("HSG", "LEM_II", "Doe 2025")

Whether a preset matches your machine's base data still depends on the machine, so check
the classes it ships before relying on one. Of the built-in presets, only the two HSG · mMKM
presets match a tissue class of the bundled machines: the mMKM class of the z\* kernels of the
generic oxygen machine (``r_nucleus`` = 3.9, ``r_domain`` = 0.32), whose geometry is that of
Inaniwa et al. 2010.

A preset (or any tissue) of another provenance is a starting point to edit:
:func:`~pyRadPlan.biology.tissues.adopt_provenance` converts it, keeping every parameter
both provenances share — all of them between MKM and mMKM, ``d_t`` and ``r_nucleus`` between
LEM versions, only ``alpha_x`` / ``beta_x`` between LEM and MKM — and the cell line.
:func:`~pyRadPlan.biology.tissues.apply_tissue_class` copies a tissue class of the
machine onto a tissue: it adopts the class's provenance and takes over the parameters
identifying the class, leaving the others (``alpha_0`` / ``beta_0``, the cell line) as they
are.

.. code-block:: python

    from pyRadPlan.biology.tissues import TissueProperties, adopt_provenance, apply_tissue_class

    cho = TissueProperties.from_preset("CHO", "MKM")
    adoption = adopt_provenance(cho, "mMKM")
    adoption.tissue.model      # 'mMKM', all parameters kept, the published alpha_x as well
    adoption.carried, adoption.defaulted
    adoption.describe()        # 'values from CHO · MKM; adopted as mMKM'
    cho.adopt_provenance("LEM_IV")  # the converted tissue only

    mmkm_class = next(t for t in machine.tissue_properties if t.model == "mMKM")
    tissue = apply_tissue_class(cho, mmkm_class)  # r_nucleus / r_domain of the class

.. _reference_parameters:

Reference parameters
~~~~~~~~~~~~~~~~~~~~

Every tissue carries the reference parameters ``alpha_x`` / ``beta_x``, the LQ parameters of
the reference radiation the RBE is taken relative to; they are never unset. A value that is
not given, or given as ``nan`` (as in files written by earlier versions), is resolved in this
order:

1. the value given explicitly, or as published with a cell line preset;
2. the model's own derivation from its model parameters: MKM / mMKM derive them by a
   conversion from ``alpha_0``, ``beta_0`` and the domain geometry (see
   :ref:`mkm_reference_parameters`);
3. the values a tissue class of the base data declares, when the class is applied with
   :func:`~pyRadPlan.biology.tissues.apply_tissue_class`;
4. the generic defaults ``alpha_x = 0.1``, ``beta_x = 0.05``
   (:func:`~pyRadPlan.biology.tissues.default_reference_parameters`), the values the
   generic classes of the bundled base data are tabulated for.

``tissue.reference_sources`` records for each of the two which applied
(:data:`~pyRadPlan.biology.tissues.REFERENCE_SOURCES`): ``"explicit"`` (given),
``"published"`` (given as published by the tissue's ``cell_line_source``, e.g. the X-ray
values of a preset), ``"estimated"`` (derived), ``"base_data"`` or ``"default"``;
``tissue.reference_source`` summarizes them as the least specific of the two. Explicit and
published values are kept as they are; the others follow the resolution above. The sources
are saved with the tissue in every format and matRad's ``tissueProperties`` struct
(``referenceSources``), but they are metadata: they are not compared by ``==`` nor in
matching the base data. Files written before the sources were tracked per parameter carry a
single ``referenceSource``, which is read as the source of both. Assigning a reference
parameter, to the tissue or through ``voi.alpha_x`` / ``voi.beta_x``, makes that parameter
``"explicit"`` and leaves the other as it is; assigning ``reference_source`` sets the source
of both, e.g. ``"estimated"`` to derive both again.

.. code-block:: python

    from pyRadPlan.biology.tissues import TissueProperties

    TissueProperties().reference_source                      # 'default'
    TissueProperties(alpha_x=0.3, beta_x=0.03).reference_source  # 'explicit'
    voi.alpha_x = 0.25
    voi.tissue_properties.reference_sources  # {'alpha_x': 'explicit', 'beta_x': 'default'}
    TissueProperties.from_preset("T1", "MKM").reference_sources
    # {'alpha_x': 'published', 'beta_x': 'published'}

Tissue classes of the base data are the exception: a class may leave its reference parameters
``nan`` (not declared), as the mMKM class of the bundled oxygen machine does, so that matching
does not compare them (see :ref:`base_data_metadata`).

.. _mkm_reference_parameters:

Reference parameters of an MKM tissue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``alpha_x`` / ``beta_x`` are not parameters of the MKM. The MKM describes a tissue by its model
parameters ``alpha_0`` / ``beta_0`` and the domain geometry (``r_domain``, ``r_nucleus``), and
gives the LQ parameters of *any* radiation from the dose-mean specific energy it deposits in a
domain: ``alpha = alpha_0 + beta_0 * z_1D`` and ``beta = beta_0`` (Hawkins, Radiat. Res. 140
(1994) 366; Kase et al., Radiat. Res. 166 (2006) 629). Evaluated for the reference radiation,
this conversion gives the reference parameters:

.. math::

   \alpha_x = \alpha_0 + \beta_0 \, z_{1D,\mathrm{ref}}, \qquad \beta_x = \beta_0 .

- **MKM** uses the plain dose-mean specific energy of the reference radiation,
  :math:`z_{1D,\mathrm{ref}} = y_D / (\rho \pi r_d^2)`, from its dose-mean lineal energy
  :math:`y_D` and the domain radius :math:`r_d` (unit density; 1 keV/µm³ = 0.1602 Gy,
  :func:`~pyRadPlan.biology.tissues.specific_energy_from_lineal_energy`).
- **mMKM** uses the saturation-corrected
  :math:`z^*_{1D,\mathrm{ref}} = z_0^2 \left(1 - e^{-z_{1D}^2/z_0^2}\right) / z_{1D}` with the
  saturation parameter :math:`z_0 = R_n^2 / \left(r_d \sqrt{\beta_0 (r_d^2 + R_n^2)}\right)`
  (Kase et al. 2006; Inaniwa et al., Phys. Med. Biol. 55 (2010) 6721), which needs the
  nucleus radius :math:`R_n` as well
  (:func:`~pyRadPlan.biology.tissues.saturation_specific_energy`,
  :func:`~pyRadPlan.biology.tissues.saturation_corrected_specific_energy`). For a reference
  radiation known only by its dose-mean value the single-event spectrum is taken as
  concentrated there; the correction is far below a percent for photons.

The reference radiation is explicit and configurable: ``tissue.reference_radiation`` names a
registered :class:`~pyRadPlan.biology.tissues.ReferenceRadiation`, described by its
dose-mean lineal energy ``y_d`` or by a fixed ``z_1d`` for one domain size, together with
its source (a :class:`~pyRadPlan.biology.tissues.ParameterSource`) and the diameter of the
site ``y_d`` was measured in (``site_diameter``). Register others with
:func:`~pyRadPlan.biology.tissues.register_reference_radiation`. The built-in ones:

.. list-table:: Built-in reference radiations
   :header-rows: 1
   :widths: 22 14 14 50

   * - Name
     - y_D [keV/µm]
     - Site [µm]
     - Source
   * - ``"xray_200kV"`` (default)
     - 4.2
     - 0.97
     - Coppola et al., Proc. 5th Symp. on Microdosimetry, EUR 5452 (1976) 377, walled
       proportional counter; quoted as the published value by Okamoto et al., J. Radiat. Res.
       52 (2011) 75, Table 1 (p. 79)
       (`doi:10.1269/jrr.10073 <https://doi.org/10.1269/jrr.10073>`_).
   * - ``"xray_200kV_tepc"``
     - 4.51
     - 1.0
     - Okamoto et al. 2011, Table 1, TEPC (GEANT4: 4.41).
   * - ``"co60"``
     - 2.34
     - 1.0
     - Okamoto et al. 2011, Table 1, TEPC (GEANT4: 2.24), equal to the published 2.34 keV/µm
       for a 0.95 µm site (Lindborg 1976).
   * - ``"xray_6MV"``
     - 2.36
     - 1.0
     - Okamoto et al. 2011, Table 1, TEPC (GEANT4: 2.27).

y_D depends on the size of the site it is measured in: for 200 kV X-rays Coppola et al.
measured 4.2 keV/µm in a 0.97 µm site but 3.3 keV/µm in a 2.06 µm one (Okamoto et al. 2011,
Discussion, p. 80). The MKM literature nonetheless uses y_D (or y\*) measured or simulated for
a site of about 1 µm diameter, whatever the domain radius it converts it for: Kase et al. (2006,
2011) with a TEPC simulating 1 µm, Okamoto et al. (2011) fitting r_d = 0.23 µm against their
1 µm TEPC values. pyRadPlan follows that convention, so ``site_diameter`` is descriptive
only and ``z_1D,ref = y_D / (ρ π r_d²)`` uses ``y_d`` as it is. The default 4.2 keV/µm is
consistent with Kase et al. (2006, p. 633), who take y_D = 4.3 ± 0.2 keV/µm (Coppola et al.,
0.76 and 0.97 µm sites) for their 200 kVp X-rays: their HSG ``alpha_0`` = 0.13 Gy⁻¹,
``beta_0`` = 0.05 Gy⁻², ``r_domain`` = 0.42 µm convert to α_x = 0.19 Gy⁻¹ with either value,
the α they measured for those X-rays. Not every MKM variant refers to X-rays: the clinical mMKM of NIRS (Inaniwa
et al., Phys. Med. Biol. 60 (2015) 3271) takes a carbon-ion beam as reference, which a
photon y_D cannot describe; register such a reference by its ``z_1d`` instead.

:class:`~pyRadPlan.biology.tissues.MKMTissueProperties` keeps ``alpha_x`` / ``beta_x``
derived from its current model parameters (``reference_source="estimated"``), also when those
are changed later, unless you give them explicitly: an explicit value is an override and kept
as it is, and so is a published one (a preset's measured X-ray values). Each parameter is
treated on its own, so overriding ``alpha_x`` alone keeps deriving ``beta_x``. A reference
parameter the model parameters do not determine falls back to the
default rather than to ``nan`` (``alpha_x`` needs ``alpha_0``, ``beta_0`` and ``r_domain``,
and ``r_nucleus`` for mMKM; ``beta_x`` only ``beta_0``):

.. code-block:: python

    from pyRadPlan.biology.tissues import MKMTissueProperties, register_reference_radiation

    cell = MKMTissueProperties(r_nucleus=4.1, r_domain=0.42, alpha_0=0.13, beta_0=0.05)
    cell.alpha_x, cell.beta_x, cell.reference_source  # derived: 0.190, 0.05, 'estimated'
    cell.model = "mMKM"                                # derived again, with z*_1D,ref
    cell.alpha_0 = 0.2                                 # ... and again

    cell.alpha_x = 0.3          # an explicit override, kept from now on
    cell.beta_0 = 0.06          # beta_x is still derived: 0.06
    cell.alpha_x = float("nan")  # unset: derived again
    cell.reference_source = "estimated"  # or derive both again at once

    cell.reference_radiation = "co60"                  # derived for Co-60 γ-rays
    register_reference_radiation(
        "my_photons", "My photon beam", y_d=2.0, site_diameter=1.0, source="own data"
    )
    cell.reference_radiation = "my_photons"            # ... or for your own

    MKMTissueProperties(beta_0=0.0615).alpha_x         # 0.1, the generic default

:meth:`~pyRadPlan.biology.tissues.MKMTissueProperties.estimate_reference_parameters`
performs the conversion on its own and reports only what the arguments allow;
:meth:`~pyRadPlan.biology.tissues.MKMTissueProperties.derived_reference_parameters` gives
what a tissue's model parameters convert to, whatever ``alpha_x`` / ``beta_x`` hold.

The GUI tissue editors show the reference parameters in a group of their own, "Reference
radiation (for RBE)". For MKM / mMKM it starts with a "Reference radiation" selector listing
every registered reference radiation with its source in short (its tooltip gives y_D, the
site and the full citation); in the plan widget's table it is the first reference column.
Changing it derives α_x / β_x anew (unless they were overridden), and a changed reference
radiation counts as an edit of the tissue, while leaving it as it was does not. Applying a
preset selects the preset's reference radiation. Derived values are shown in grey italics and
follow the model parameters live while you edit them. Typing a value overrides it, the other
one staying derived; "Derive from model" discards the override and derives them again (in the
plan widget's table for the selected structures, or all MKM / mMKM structures if none are
selected). Published values, e.g. those of the Kase et al. presets, are shown as given, with
their publication in the tooltip, and do not follow the model parameters; "Derive from model"
switches them to derived values too. A structure whose reference parameters were explicit,
e.g. one loaded from matRad, keeps them until you press it.

Matching the base data is not affected: MKM and mMKM tissue classes are identified by
``r_nucleus`` / ``r_domain`` only.

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
