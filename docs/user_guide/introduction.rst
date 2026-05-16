.. _introduction:

Introduction
============

pyRadPlan is a multi-modality radiotherapy treatment planning toolkit in Python, born from the
established MATLAB-based toolkit `matRad <http://www.matRad.org>`_. It is developed by the
`Radiotherapy Optimization group <https://www.dkfz.de/radopt>`_ at the
`German Cancer Research Center (DKFZ) <https://www.dkfz.de>`_.

Mission
-------

pyRadPlan's mission is **AI-ready research treatment planning**. It is primarily aimed at:

- **matRad users** seeking a Python-native equivalent of the matRad workflow.
- **Treatment planning researchers** who need a flexible, extensible platform for algorithmic
  development and experimentation.

The toolkit keeps a clear focus on numerics while remaining easy to integrate with modern AI
tooling. This translates into two foundational design choices that are consistent throughout the
entire codebase:

1. **Pydantic-based data structures** — the main data structures derive from a common
   :class:`~pyRadPlan.core.datamodel.PyRadPlanBaseModel` base class backed by
   `pydantic <https://docs.pydantic.dev/latest/>`_. Pydantic enforces schema validation and
   provides consistent model dumping and matRad conversion hooks. Some objects contain
   SimpleITK images or sparse matrices, so fully JSON-ready serialization may still require
   converting those payloads explicitly.

2. **Backend-agnostic algorithms via the Python Array API** — dose calculation and optimization
   kernels are written against the
   `Python Array API standard <https://data-apis.org/array-api/latest/>`_ rather than NumPy
   directly where practical. The preferred namespace is configured through
   :mod:`pyRadPlan.core.xp_utils`, enabling NumPy, CuPy, or PyTorch-backed paths depending on
   which optional dependencies are installed and supported by the current workflow.

matRad Interoperability
-----------------------

pyRadPlan is designed to interoperate closely with matRad:

- Patient data (CT, structure set) can be loaded directly from ``*.mat`` files produced by
  matRad using :func:`~pyRadPlan.load_patient`.
- All major data structures expose a ``to_matrad()`` method that serializes them back to a
  matRad-compatible representation, enabling round-trip workflows between the two systems.
- Imported matRad data can be used in the Python planning workflow, and exported pyRadPlan
  structures can be handed back to matRad-compatible tooling.

Data Structures Overview
------------------------

The following table lists the main objects in a pyRadPlan workflow and the section of the
concepts guide that explains them in detail.

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Object
     - Type
     - Description
   * - ``ct``
     - :class:`~pyRadPlan.ct.CT`
     - CT image in Hounsfield Units
   * - ``cst``
     - :class:`~pyRadPlan.cst.StructureSet`
     - Segmented structures (targets, OARs) with optimization objectives
   * - ``pln``
     - :class:`~pyRadPlan.plan.Plan`
     - Plan configuration (modality, machine, fractionation, algorithm settings)
   * - ``stf``
     - :class:`~pyRadPlan.stf.SteeringInformation`
     - Beam geometry (gantry/couch angles, spots/bixels)
   * - ``dij``
     - :class:`~pyRadPlan.dij.Dij`
     - Dose influence matrix (voxels × beamlets)
   * - ``fluence``
     - :class:`numpy.ndarray`
     - Optimized beamlet weights / monitor units
