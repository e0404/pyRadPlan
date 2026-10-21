biology.tissues
===============

.. currentmodule:: pyRadPlan.biology.tissues

Tissue classes
--------------

:class:`TissueProperties` holds the reference photon LQ parameters every RBE model needs.
Each subclass adds the parameters of one model, and declares which of them identify a
tissue class of that model (see :attr:`TissueProperties.identifying_fields`); those are the
fields :meth:`TissueProperties.matches` compares, so a LEM tissue is never matched against
MKM base data.

:class:`LEMTissueProperties` carries the LEM version in a field of its own. Together with
the ``model`` name (e.g. MKM or mMKM) it labels the provenance of the parameters
(:attr:`TissueProperties.provenance_fields`): a soft label that :meth:`TissueProperties.matches`
does not compare, so a LEM I tissue matches a LEM IV class with the same parameters, and an
MKM tissue an mMKM class of the same geometry. :meth:`TissueProperties.find_match` and
:meth:`TissueProperties.lookup` prefer a class of the tissue's own provenance; biological
models warn about a differing label (see :ref:`provenance_as_label`).

.. autosummary::
   :toctree: generated/biology_tissues/

   TissueProperties
   LEMTissueProperties
   LEM_VERSIONS
   MKMTissueProperties

Each class separates the parameters of its model
(:attr:`TissueProperties.model_parameter_fields`) from the reference parameters ``alpha_x`` /
``beta_x`` of the radiation the RBE refers to (:attr:`TissueProperties.reference_fields`).
LEM takes the reference parameters as model inputs as well; :class:`MKMTissueProperties`
derives them (:attr:`TissueProperties.derives_reference`) by the conversion
``alpha_x = alpha_0 + beta_0 * z_1D,ref`` and ``beta_x = beta_0``, with ``z_1D,ref`` the
dose-mean specific energy the reference radiation deposits in a domain, saturation-corrected
for mMKM (:meth:`MKMTissueProperties.estimate_reference_parameters`). Unless given
explicitly, they stay derived from the current model parameters; see
:ref:`mkm_reference_parameters` for using this from scripts and from the GUI.


Reference parameters
--------------------

The reference parameters of a tissue are never unset: one not given (or NaN) is derived by
the model or gets the generic default, and :attr:`TissueProperties.reference_sources` records
which, per parameter (:data:`REFERENCE_SOURCES`; explicit and published values,
:data:`FIXED_REFERENCE_SOURCES`, are never derived again).
:attr:`TissueProperties.reference_source` summarizes the two. Tissue classes of base data are
built by
:meth:`TissueProperties.from_base_data`, which keeps NaN as "not declared". See
:ref:`reference_parameters`.

.. autosummary::
   :toctree: generated/biology_tissues/

   default_reference_parameters
   DEFAULT_ALPHA_X
   DEFAULT_BETA_X
   REFERENCE_FIELDS
   REFERENCE_SOURCES
   FIXED_REFERENCE_SOURCES
   weakest_reference_source
   declared_reference_parameters


Reference radiations and the MKM conversion
-------------------------------------------

The reference radiation an MKM tissue derives its reference parameters for
(``MKMTissueProperties.reference_radiation``) is a registered :class:`ReferenceRadiation`,
described by its dose-mean lineal energy (with the site diameter it was measured for) or a
fixed dose-mean specific energy, and carrying its source as a :class:`ParameterSource`. The
built-in ones (200 kV X-rays, Co-60, 6 MV X-rays) are cited from the MKM literature. The
functions below are the building blocks of the conversion. See
:ref:`mkm_reference_parameters`.

.. autosummary::
   :toctree: generated/biology_tissues/

   ReferenceRadiation
   DEFAULT_REFERENCE_RADIATION
   get_reference_radiation
   register_reference_radiation
   registered_reference_radiations
   specific_energy_from_lineal_energy
   saturation_specific_energy
   saturation_corrected_specific_energy
   GY_PER_KEV_PER_UM3


Converting tissues
------------------

:func:`adopt_provenance` converts a tissue to another provenance, keeping the parameters both
share; :func:`apply_tissue_class` copies a tissue class of base data onto a tissue.

.. autosummary::
   :toctree: generated/biology_tissues/

   adopt_provenance
   ProvenanceAdoption
   apply_tissue_class


Provenance registry
-------------------

The provenance of base data is the radiobiological model (and version) that produced it, e.g.
``LEM_I`` or ``mMKM``. The registry maps the names used in base-data metadata onto the
tissue class holding that model's parameters; :meth:`TissueProperties.from_dict` dispatches
through it. See :ref:`biological_base_data`.

.. autosummary::
   :toctree: generated/biology_tissues/

   Provenance
   get_provenance
   register_provenance
   registered_provenances
   provenance_label
   tissue_provenance
   UNDECLARED_PROVENANCE


Cell line presets
-----------------

Each preset carries the source of its values (:class:`ParameterSource`), which tissues built
from it record as ``cell_line_source``; MKM / mMKM presets also name the reference radiation
they were published for (:attr:`CellLinePreset.reference_radiation`), and a preset may give
the published reference parameters (:attr:`CellLinePreset.published_reference`). A cell line
may have several presets per provenance, told apart by the label of their source; the first
registered is the default of :func:`get_cell_line_preset`, which selects others by their
``source``. A source is any string naming the origin, or one of the sentinels
:data:`UNVERIFIED_SOURCE` (the default) and :data:`UNPUBLISHED_SOURCE`. See
:ref:`cell_line_presets`.

.. autosummary::
   :toctree: generated/biology_tissues/

   CELL_LINE_PRESETS
   CellLinePreset
   get_cell_line_preset
   registered_cell_line_presets
   register_cell_line_preset
   ParameterSource
   UNVERIFIED_SOURCE
   UNPUBLISHED_SOURCE
   SOURCE_SENTINELS
   is_verified_source
