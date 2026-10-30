.. _biological_base_data:

Biological base data
====================

Variable-RBE models read precomputed radiobiological data: α/β or z\* kernels stored per
energy in the machine file, or RBE tables averaged over fragment fluence spectra. This page
explains how pyRadPlan tells apart *what it computes* from *where the data comes from*, how to
choose between several data sets, and which metadata a machine file or RBE table should carry.
The last part is written for people producing base data.

Formalism and provenance
------------------------

**Formalism.** What pyRadPlan evaluates at runtime. A formalism is code: one
:class:`~pyRadPlan.biology.models.BiologicalModel` class, selected by its identifier in
``pln.bio_model``. It is named by what it reads, not by the model that produced the data.

**Provenance.** The radiobiological model and its parameterization that produced a set of
kernels or tables, e.g. LEM I, LEM IV, MKM or mMKM. Provenance is data: it is declared in the
base-data metadata and never hard-coded in a formalism. α/β kernels computed with mMKM are
evaluated by the same formalism as α/β kernels computed with LEM I.

**Kernel set.** One family of per-energy kernel data in a machine, e.g. "the α/β kernels" or
"the z\* kernels". A kernel set has a kind, an optional variant, a provenance and, if it is
tissue dependent, one column per tissue class.

**Tissue class.** One parameterization of one provenance in the base data: one column of a
kernel set, or one RBE table. A structure's ``tissue_properties`` must resolve to exactly one
tissue class of the kernel set the formalism uses (see :ref:`voi_tissue_properties`).

Options are labelled **formalism · provenance**, e.g. "α/β kernels · LEM I" or, for data whose
metadata names no model, "α/β kernels · model not declared".

Formalism identifiers
---------------------

.. list-table::
   :header-rows: 1
   :widths: 24 22 22 32

   * - Identifier
     - Former identifiers (aliases)
     - Display name
     - Reads
   * - ``"none"``
     -
     - None (physical dose)
     - –
   * - ``"constant_rbe"``
     - ``"constRBE"``
     - Constant RBE
     - –
   * - ``"WED"``, ``"MCN"``, ``"CAR"``, ``"HEL"``, ``"LSM"``
     -
     - Wedenberg, McNamara, Carabe, Mairani, Linear scaling (LET)
     - a LET kernel set and the structures' α_x/β_x
   * - ``"alpha_beta_kernels"``
     - ``"kernel_based_lq"``, ``"LEM"``
     - α/β kernels
     - an α/β kernel set of the machine
   * - ``"alpha_beta_tables"``
     - ``"dose_average_alpha_beta"``
     - α/β tables + fluence spectra
     - an RBE table (see `Where base data is found`_) and the machine's fragment fluence spectra
   * - ``"zstar_kernels"``
     - ``"MKM"``, ``"zs_based_lq"``
     - z\* kernels
     - a z\* kernel set of the machine; α₀/β₀ from the structures' tissues

Plans store the identifier. The former identifiers keep plans, workspaces and scripts loading
and select the same class; matRad export keeps matRad's names (``LEM``, ``constRBE``,
``MKM``). In :mod:`pyRadPlan.biology.models` the classes are also available as
``AlphaBetaKernelModel``, ``AlphaBetaTableModel`` and ``ZStarKernelModel``.

Choosing base data
------------------

On a machine with one set per kind, which covers all bundled data, nothing has to be
configured. Where a machine carries several sets of one kind, the data-driven formalisms take
optional parameters that are stored with the plan:

``data_model``
    The provenance to use, e.g. ``{"model": "alpha_beta_kernels", "data_model": "LEM_I"}``.
    For ``alpha_beta_tables`` it selects the RBE table of that provenance.
``kernel_set``
    The id of a kernel set, for machines carrying several sets of one provenance.
``quantity_variants``
    For quantity-based formalisms, the variant of a quantity, e.g.
    ``{"model": "WED", "quantity_variants": {"let": "LETt"}}``.

Resolution is deterministic: an explicit ``kernel_set`` wins; otherwise the sets of the right
kind matching ``data_model``; otherwise the sets the structures' tissues resolve to; otherwise
the set marked ``default`` or the only set of that kind. A remaining ambiguity raises an error
that lists the candidate sets.

:func:`~pyRadPlan.biology.models.bio_model_options` lists what a machine offers, one
:class:`~pyRadPlan.biology.models.BioModelOption` per formalism and kernel set, with the reason
for options the machine cannot run.
:meth:`~pyRadPlan.biology.models.BiologicalModel.check_tissues` returns one
:class:`~pyRadPlan.biology.models.TissueIssue` per structure that does not resolve; the dose
engines run the same check before a biological dose calculation.

.. code-block:: python

    from pyRadPlan.biology.models import bio_model_options, create_bio_model
    from pyRadPlan.machines import load_from_name

    machine = load_from_name("oxygen", "Generic")
    for kernel_set in machine.kernel_sets.values():
        print(kernel_set.id, kernel_set.label, [c.name for c in kernel_set.tissue_classes])
    # alpha_beta α/β kernels · LEM I ['Generic_LEMax01', 'Generic_LEMax05']
    # zs z* kernels · mMKM ['Generic_MKM']

    for option in bio_model_options("oxygen", machine):
        print(option.model, option.label, option.available, option.reason)

    issues = create_bio_model("alpha_beta_kernels").check_tissues(machine, cst)

Base data files run to hundreds of megabytes.
:meth:`BioMachineSummary.from_machine <pyRadPlan.biology.models.BioMachineSummary.from_machine>`
keeps only the metadata and can be passed wherever these functions take a machine.

.. _base_data_folders:

Where base data is found
------------------------

Machine files, RBE tables and stopping power tables are selected by file name: a machine by
``<radiation_mode>_<machine_name>.mat``, a table by its ``quantity_table_name`` or
``sp_table_name``. Each kind is looked up in two folders, the shipped one first:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Kind
     - Shipped with pyRadPlan
     - Added by you
   * - Machine files
     - ``pyRadPlan/data/machines``
     - ``~/.pyradplan/machines``
   * - RBE tables
     - ``pyRadPlan/data/rbe``
     - ``~/.pyradplan/rbe``
   * - Stopping power tables
     - ``pyRadPlan/data/physics/stopping``
     - ``~/.pyradplan/physics/stopping``

The user folders lie below the data directory, ``~/.pyradplan`` unless the
``PYRADPLAN_DATA_DIR`` environment variable names another one (see
:func:`pyRadPlan.core.get_data_dir`). A file copied there by hand is found like one added
through the API or the GUI. A file named like a shipped one is ignored; shipped files cannot be
replaced.

Files are added with a check of their content:

.. code-block:: python

    from pyRadPlan.machines import add_machine_file
    from pyRadPlan.biology.models import add_rbe_table, add_stopping_power_table

    add_machine_file("carbon_MyMachine.mat")  # listed as carbon machine "MyMachine"
    add_rbe_table("RBEtable_MKM_HSG.mat")
    add_stopping_power_table("SPtable_custom.mat")

:func:`~pyRadPlan.core.list_base_data` lists the files of a kind with their source, and
:func:`~pyRadPlan.core.remove_base_data_file` deletes one the user added. In the GUI, the
*Base Data* menu lists all files and adds or removes them.

.. _base_data_metadata:

Metadata specification (version 1)
----------------------------------

This part is for people producing machine files and RBE tables. The specification describes
matRad-style ``.mat`` files. Field names are camelCase as in matRad; pyRadPlan maps them to
snake_case when loading. **Every field is optional**: data without it loads as before
(see `Data without metadata`_), but labels then read "model not declared" and several sets
of one kind cannot be told apart.

Kernel sets: ``machine.meta.kernelSets``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A 1×n struct array, one element per kernel set (a cell array of structs is read as well). All
values are char unless noted.

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Meaning
   * - ``id``
     - Unique name within the machine, referenced by tissue classes (``ref_Kernel``).
       Defaults to the kind, or ``<kind>_<n>`` if the kind repeats.
   * - ``kind``
     - ``alpha_beta``, ``zs`` or ``let`` (required).
   * - ``fields``
     - Kernel fields of ``machine.data`` holding the set, as a cellstr (``{'alpha','beta'}``)
       or one comma-separated char (``'alpha_mMKM,beta_mMKM'``). Defaults to the kind's
       standard fields: ``alpha``/``beta``, ``zs``, ``LET``.
   * - ``variant``
     - Kind-specific definition, e.g. for ``let``: ``LETd``, ``LETt``,
       ``LETd_restricted_<keV>``; for ``zs``: ``z1D``, ``z1D_saturation_corrected``. Leave it
       empty if the definition is not known.
   * - ``model``
     - Provenance, e.g. ``LEM_I`` or ``mMKM`` (see `Provenance names`_). Empty for kinds that
       are not tissue dependent (LET).
   * - ``version``
     - Optional model version, e.g. ``model: LEM`` with ``version: LEM_I``.
   * - ``reference``
     - Free text: publication, code and version used to produce the data.
   * - ``default``
     - Logical or 0/1. Marks the set used when several sets of one kind exist and nothing is
       configured.

Several sets of one kind store their per-energy arrays in fields of their own in
``machine.data``, next to matRad's standard ones; the convention is ``<field>_<suffix>``
(``alpha_mMKM``, ``beta_mMKM``, ``LETt``). Tissue-dependent arrays have the shape of ``alpha``
(n_depths × n_columns). Only the set reading the standard ``alpha``/``beta`` identifies its
columns by the per-energy ``alphaX``/``betaX``; the tissue classes of any other set give a
``column`` or are mapped in declaration order.

.. code-block:: matlab

    machine.meta.kernelSets(1) = struct('id','lem1', 'kind','alpha_beta', ...
        'fields',{{'alpha','beta'}}, 'variant','', 'model','LEM', 'version','LEM_I', ...
        'reference','Scholz and Kraft 1996, ...', 'default',true);
    machine.meta.kernelSets(2) = struct('id','mmkm', 'kind','alpha_beta', ...
        'fields',{{'alpha_mMKM','beta_mMKM'}}, 'variant','', 'model','mMKM', 'version','', ...
        'reference','Inaniwa et al. 2010, ...', 'default',false);
    save('carbon_MyMachine.mat', 'machine', '-v7');

The same struct array written from Python:

.. code-block:: python

    import numpy as np
    import scipy.io

    keys = ["id", "kind", "fields", "variant", "model", "version", "reference", "default"]
    rows = [
        ["lem1", "alpha_beta", np.array([["alpha", "beta"]], dtype=object), "", "LEM",
         "LEM_I", "Scholz and Kraft 1996, ...", True],
        ["mmkm", "alpha_beta", np.array([["alpha_mMKM", "beta_mMKM"]], dtype=object), "",
         "mMKM", "", "Inaniwa et al. 2010, ...", False],
    ]
    kernel_sets = np.empty((1, len(rows)), dtype=[(key, object) for key in keys])
    for i, row in enumerate(rows):
        for key, value in zip(keys, row):
            kernel_sets[0, i][key] = value

When adding metadata to an existing file, edit it in MATLAB (``load``, assign, ``save -v7``)
where possible: a round trip through ``scipy.io`` keeps the values, but MATLAB stores
integer-valued doubles compactly and ``scipy.io`` writes them back as full doubles, which
changes the dtype readers such as ``pymatreader`` report (``uint8`` becomes ``float64``).

Tissue classes: ``machine.meta.tissueParameters``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A struct with one element per tissue class (a struct array, or a scalar struct whose fields
hold one value per class). The fields are passed to
:meth:`TissueProperties.from_base_data <pyRadPlan.biology.tissues.TissueProperties.from_base_data>`,
so the provenance selects the parameter class; unknown fields are ignored. Unlike a
structure's tissue, whose reference parameters are never unset (see
:ref:`reference_parameters`), a tissue class keeps ``alphaX`` / ``betaX`` given as NaN as
"not declared"; reference parameters it leaves out entirely get the generic defaults.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``cell_Line``
     - Display name of the class, e.g. ``Generic_LEMax01``.
   * - ``model``, ``version``
     - Provenance, e.g. ``LEM`` (LEM I), ``LEM_IV``, ``MKM``, ``mMKM``.
   * - model parameters
     - ``alphaX``, ``betaX`` and the provenance's parameters: ``Dt`` and ``rNucleus`` for LEM;
       ``rNucleus``, ``rDomain`` (and optionally ``alpha0``, ``beta0``) for MKM / mMKM. Unset
       values are NaN. A parameter left unset in the base data is not declared and is not
       compared when a structure's tissue is matched to the class; at least one declared
       parameter must match.
   * - ``ref_Kernel``
     - ``id`` of the kernel set the class belongs to.
   * - ``column``
     - One-based column of that kernel set. NaN or missing: matched by ``alphaX``/``betaX``
       for the standard α/β set, by declaration order otherwise.

Without ``ref_Kernel`` a class belongs to the set of its kind (MKM / mMKM parameters: z\*;
all others: α/β), chosen by provenance or by the ``default`` flag if the kind has several
sets. Which fields of a class a structure has to match depends on its parameter class: α_x,
β_x, ``d_t`` and ``r_nucleus`` for LEM; the ``r_nucleus`` / ``r_domain`` geometry for MKM and
mMKM, whose α₀/β₀ come from the structure's tissue. A parameter the class leaves unset (NaN)
is not compared. The provenance of the structure's tissue (its ``model`` label, MKM or mMKM,
and the LEM ``version``) is a soft label, not a match criterion: an MKM tissue resolves to an
mMKM class of the same geometry, a LEM I tissue to a LEM IV class with the same parameters,
and the model is evaluated with the base data, i.e. with the class's provenance. Such a
structure gets a warning (see :ref:`provenance_as_label`). Tissues of parameter classes
sharing no identifying parameter (a LEM tissue and an MKM class) never match.

Example, the generic oxygen machine:

.. code-block:: matlab

    machine.meta.tissueParameters = struct( ...
        'cell_Line', {'Generic_MKM', 'Generic_LEMax01', 'Generic_LEMax05'}, ...
        'model', {'mMKM', 'LEM', 'LEM'}, ...
        'alphaX', {NaN, 0.1, 0.5}, 'betaX', {NaN, 0.05, 0.05}, ...
        'rNucleus', {3.9, 5, 5}, 'rDomain', {0.32, NaN, NaN}, 'Dt', {NaN, 30, 30}, ...
        'ref_Kernel', {'zs', 'alpha_beta', 'alpha_beta'});

and the generic carbon machine, whose LEM cell parameters are not documented and hence left
unset:

.. code-block:: matlab

    machine.meta.tissueParameters = struct( ...
        'cell_Line', {'Generic_LEMax01', 'Generic_LEMax05'}, ...
        'alphaX', {0.1, 0.5}, 'betaX', {0.05, 0.05}, 'rNucleus', {NaN, NaN}, 'Dt', {NaN, NaN}, ...
        'model', {'LEM', 'LEM'}, 'version', {'LEM_IV', 'LEM_IV'}, ...
        'ref_Kernel', {'alpha_beta', 'alpha_beta'}, 'column', {1, 2});

RBE tables: ``RBEtable.meta``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RBE tables (see `Where base data is found`_) are read by ``alpha_beta_tables``. Their metadata
describes the tabulated values only; how the values are combined (averaged over the fragment
fluence spectra for ``alpha_beta_tables``) is decided by the model using the table, so a table
declares no formalism.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``model``, ``version``
     - Provenance of the table, e.g. ``model: LEMI`` or ``model: LEM`` with
       ``version: LEM_I``.
   * - ``modelParameters``
     - The tissue parameters the table was computed with, under the names
       ``tissueParameters`` uses (``alphaX``, ``betaX``, ``rNucleus``, ``Dt``, ...). Scalars
       for a table of one tissue class; equally long arrays for one tabulated per class, with
       scalars applying to every class.
   * - ``reference``
     - Free text: publication, code and version used to produce the table.

Provenance names
~~~~~~~~~~~~~~~~

Provenance names are resolved by :func:`~pyRadPlan.biology.tissues.get_provenance`, case
insensitively:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Provenance
     - Accepted spellings
     - Parameter class
   * - ``LEM_I`` … ``LEM_IV``
     - ``LEM_II``, ``LEM 2``, ``LEMII``, ``model: LEM`` with ``version: II``; a bare ``LEM``
       is LEM I
     - :class:`~pyRadPlan.biology.tissues.LEMTissueProperties` (``version`` field)
   * - ``MKM``, ``mMKM``
     - as written
     - :class:`~pyRadPlan.biology.tissues.MKMTissueProperties`
   * - ``undeclared``
     - empty, ``Generic``, ``not declared``
     - :class:`~pyRadPlan.biology.tissues.TissueProperties`

An unknown name is accepted and described by plain
:class:`~pyRadPlan.biology.tissues.TissueProperties` (α_x/β_x only). A new provenance with
parameters of its own registers its class with
:func:`~pyRadPlan.biology.tissues.register_provenance`.

Data without metadata
~~~~~~~~~~~~~~~~~~~~~

- **No** ``kernelSets``: one set per kind present in the kernels (``alpha_beta`` if
  ``alpha``/``beta`` exist, ``zs``, ``let``), with ids ``alpha_beta``, ``zs`` and ``let``. Its
  provenance is "not declared", unless all tissue classes of the set declare the same one,
  which the set then takes over.
- **No** ``tissueParameters``: one tissue class per α/β kernel column, from the kernels'
  ``alphaX``/``betaX``, matched on α_x/β_x only.

Checking a file
~~~~~~~~~~~~~~~

Load the file and inspect its kernel sets; ``problems`` lists inconsistencies such as missing
kernel fields or columns outside the kernels:

.. code-block:: python

    from pyRadPlan.machines import load_machine

    machine = load_machine("carbon_MyMachine.mat")
    for kernel_set in machine.kernel_sets.values():
        print(kernel_set.id, kernel_set.label, kernel_set.declared, kernel_set.problems)
        for tissue_class in kernel_set.tissue_classes:
            print("   ", tissue_class.name, tissue_class.column)
    print(machine.provided_quantities())

Bundled data
------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Metadata
   * - ``oxygen_Generic.mat``
     - "α/β kernels · LEM I" with the tissue classes ``Generic_LEMax01`` and
       ``Generic_LEMax05`` (α_x/β_x = 0.1/0.05 and 0.5/0.05, ``Dt`` = 30, ``rNucleus`` = 5),
       and "z\* kernels · mMKM": an mMKM dataset with saturation-corrected z\*
       (``variant: z1D_saturation_corrected``) and the tissue class ``Generic_MKM``
       (``rNucleus`` = 3.9, ``rDomain`` = 0.32).
   * - ``carbon_Generic.mat``
     - "α/β kernels · LEM IV" with the tissue classes ``Generic_LEMax01`` and
       ``Generic_LEMax05`` (α_x/β_x = 0.1/0.05 and 0.5/0.05). The LEM cell parameters
       (``Dt``, ``rNucleus``) of this data are not documented and are left unset, so these
       classes match reference-only tissues with those α_x/β_x (the generic default tissue
       among them) and LEM tissues with those α_x/β_x, whatever their ``d_t`` /
       ``r_nucleus``; a LEM tissue labelled with another version than IV gets a warning.
   * - ``RBEtable_LEMI_Scholz06_AX01_BX005.mat``
     - LEM I, α_x = 0.1, β_x = 0.05, ``Dt`` = 30,
       ``rNucleus`` = 5, with its reference.

No bundled machine carries fragment fluence spectra, so ``alpha_beta_tables`` is listed but
unavailable on the bundled machines.

The cell line presets (:data:`~pyRadPlan.biology.tissues.CELL_LINE_PRESETS`: HSG, CHO, T1
and V79 for MKM, HSG, CHO and V79 for LEM I–III, HSG for mMKM) mostly match none of the bundled
tissue classes, so use the machine's tissue classes with the bundled data:

- **carbon, α/β kernels · LEM IV**: there are no LEM IV presets.
- **oxygen, α/β kernels · LEM I**: the LEM I presets have other α_x/β_x (and cell
  parameters) than the generic classes.
- **oxygen, z\* kernels · mMKM**: the two HSG · mMKM presets (Inaniwa et al. 2010, with a
  derived and with the published X-ray reference) have the geometry of ``Generic_MKM`` and
  match it exactly; the MKM presets are MKM parameterizations with other domain geometries.

Each preset names its source, and a cell line may have several presets for one provenance,
one per publication; see :ref:`cell_line_presets` for the publication, table and page each
preset's values are taken from.

The tissue editors of the GUI list the tissue classes of the selected base data and the presets
of its provenance first (marking those that match exactly), and all other presets below,
grouped by provenance, each labelled with its source in short. A preset of another provenance is converted to the provenance of the
base data when applied (see below), and the editors highlight every parameter that still
differs from the closest tissue class of the base data. A model label (or LEM version)
differing from the base data's is only tinted softly, in a warning colour.

Fitting tissues to the base data
--------------------------------

A preset, or a structure's tissue, of another provenance can still serve as a starting
point. :func:`~pyRadPlan.biology.tissues.adopt_provenance` converts it to the provenance of
the base data, keeping the parameters both share (see :ref:`voi_tissue_properties`), and
:func:`~pyRadPlan.biology.tissues.apply_tissue_class` copies a tissue class onto it: the
class's provenance and the parameters identifying it, except those the class leaves
undeclared. Reference parameters the class declares replace only defaulted ones of the tissue,
unless they identify the class (LEM, reference-only classes).

.. _provenance_as_label:

Provenance is a label
~~~~~~~~~~~~~~~~~~~~~

The provenance of a structure's tissue does not have to be the base data's: it labels the
parameters rather than being one of them. A tissue matching a class of the base data by its
parameters resolves to it whatever its label, and the model is evaluated with the base data's
provenance. If the labels differ — e.g. an MKM tissue for the mMKM z\* kernels of the oxygen
machine, or a LEM I tissue for the LEM IV α/β kernels of the carbon machine —
:meth:`~pyRadPlan.biology.models.BiologicalModel.check_tissues` reports a ``provenance_differs``
issue of ``severity="warning"``, e.g. "PTV: The tissue (HSG) is labelled MKM; the base data is
mMKM (z\* kernels · mMKM, z1D_saturation_corrected, …). Evaluated with the base data's model."
Every other issue is an error (``severity="error"``, see
:attr:`~pyRadPlan.biology.models.TissueIssue.is_error`).

Dose engines act on the severity: errors raise with ``calc_bio_dose=True`` or fall back to the
physical dose with ``"auto"`` (invalid reference parameters always raise), while warnings are
logged, reported as engine warnings (which the GUI shows) and the biological dose is computed.
Where several classes or kernel sets match, those of the tissue's own provenance are
preferred.

Relabelling such a tissue is optional and only silences the warning: the warning carries the
base data's provenance as ``fix_provenance`` (only if the tissue's parameter class is that
provenance's, so every parameter is kept), and
:func:`~pyRadPlan.biology.models.apply_provenance_fixes` relabels a copy of the structure set:

.. code-block:: python

    from pyRadPlan.biology.models import apply_provenance_fixes, create_bio_model

    model = create_bio_model("zstar_kernels")
    issues = model.check_tissues(machine, cst)
    [(issue.kind, issue.severity) for issue in issues]  # [('provenance_differs', 'warning')]
    [issue.fix_provenance for issue in issues]           # ['mMKM']
    cst = apply_provenance_fixes(cst, issues)            # the input is not modified

In the GUI, warnings never block or ask: the tissue editors tint the model softly and offer
"Relabel as <model>", the plan widget marks the tissue button with ℹ (⚠ for errors), and the
workflow panel logs the warnings and calculates with the biological model, asking only about
errors.

Comparing a tissue with the base data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~pyRadPlan.biology.models.BiologicalModel.compare_tissues` (or
:func:`~pyRadPlan.biology.models.compare_tissue_parameters`) shows, per structure, the matching or
closest tissue class and each parameter the match compares — declared value, tissue value and
whether they agree, undeclared parameters and the provenance labels left out. ``matched`` tells
whether the parameters match; ``provenance_mismatch`` whether the provenance differs, a
warning if the tissue matched and an error (``type_mismatch``: the parameter classes share no
identifying parameter) if not. It selects the kernel set and the classes exactly as
``check_tissues`` does, so both agree:

.. code-block:: python

    for comparison in model.compare_tissues(machine, cst):
        print(comparison.structure, comparison.matched, comparison.mismatched)
        for parameter in comparison.parameters:
            print("   ", parameter.name, parameter.declared, parameter.value, parameter.matches)
