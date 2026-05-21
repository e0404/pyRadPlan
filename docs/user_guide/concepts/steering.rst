.. _concept_steering:

Steering: Beam Geometry
========================

The *steering information* (``stf``) describes the physical geometry of all treatment beams:
their directions, the positions of individual pencil beams / spots, and — for particle therapy —
the energy layers within each beam.

Generating the steering information
------------------------------------

:func:`~pyRadPlan.generate_stf` reads beam directions and other settings from the plan's
``prop_stf`` dictionary and produces a fully validated
:class:`~pyRadPlan.stf.SteeringInformation` object:

.. code-block:: python

    from pyRadPlan import generate_stf, IonPlan

    pln = IonPlan(radiation_mode="protons", machine="Generic")
    pln.prop_stf = {
        "gantry_angles": [0, 90, 270],   # degrees; one beam per angle
        "couch_angles":  [0,  0,   0],
        "bixel_width":   5,              # mm lateral spot spacing
        "generator":     "IMPT",         # default for ions
    }

    stf = generate_stf(ct, cst, pln)

The generator can be selected explicitly via ``prop_stf["generator"]``. Available generators
include:

- **IMPT** — Intensity-modulated particle therapy (ions). Produces energy-layer spot maps that
  cover the target.
- **ionSingleSpot** — A single particle spot, useful for focused tests and debugging.
- **photonIMRT**, **photonOpenFields**, **photonSingleBixel** — Photon steering generators for
  IMRT-like, open-field, and single-bixel setups.
- **VHEE** — Very-high-energy electron steering.

Object hierarchy
-----------------

The steering information is organized as a three-level hierarchy:

.. code-block:: text

    SteeringInformation
    └── Beam  (one per gantry/couch angle pair)
        └── Ray  (one per lateral pencil-beam position)
            └── Beamlet  (one per energy / focus setting)

SteeringInformation
~~~~~~~~~~~~~~~~~~~

:class:`~pyRadPlan.stf.SteeringInformation` is the top-level container.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute
     - Description
   * - ``beams``
     - List of :class:`~pyRadPlan.stf.Beam` objects.
   * - ``num_of_beams``
     - Total number of beams.
   * - ``num_of_rays``
     - Total ray count across all beams.
   * - ``total_number_of_bixels``
     - Total beamlet count. This is the column dimension of the dose influence matrix.
   * - ``bixel_beam_index_map``
     - Integer array mapping every global bixel index to its beam index.

Beam
~~~~

:class:`~pyRadPlan.stf.Beam` represents a single irradiation direction.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute
     - Description
   * - ``gantry_angle``
     - Gantry rotation in degrees (IEC convention).
   * - ``couch_angle``
     - Couch rotation in degrees.
   * - ``bixel_width``
     - Lateral spot spacing in mm.
   * - ``sad``
     - Source-to-axis distance in mm.
   * - ``iso_center``
     - Isocenter coordinates ``[x, y, z]`` in world coordinates.
   * - ``source_point``
     - Source position in world coordinates.
   * - ``rays``
     - List of :class:`~pyRadPlan.stf.Ray` objects.

Ray
~~~

:class:`~pyRadPlan.stf.Ray` represents one lateral pencil-beam direction within a beam.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute
     - Description
   * - ``ray_pos``
     - Ray position in world coordinates (mm).
   * - ``ray_pos_bev``
     - Ray position in beam's-eye-view (BEV) coordinates.
   * - ``beamlets``
     - List of :class:`~pyRadPlan.stf.Beamlet` objects (one per energy layer).

Beamlet
~~~~~~~

:class:`~pyRadPlan.stf.Beamlet` is the smallest unit — a single pencil beam at a fixed
energy.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute
     - Description
   * - ``energy``
     - Beam energy in MeV.
   * - ``weight``
     - Initial fluence weight (updated by the optimizer).
   * - ``num_particles_per_mu``
     - Conversion factor from monitor units to particle count.
   * - ``min_mu`` / ``max_mu``
     - Monitor-unit bounds for the optimizer.
   * - ``range_shifter``
     - Optional range-shifting device description.

Coordinate systems
------------------

pyRadPlan uses the **LPS** (Left-Posterior-Superior) coordinate system for world coordinates,
consistent with DICOM and SimpleITK. Beam's-eye-view (BEV) coordinates follow the IEC 61217
convention.

matRad interoperability
-----------------------

.. code-block:: python

    matrad_stf = stf.to_matrad()  # list of dicts, one per beam
