machines
========

.. currentmodule:: pyRadPlan.machines.base

.. autosummary::
    :toctree: generated/machines/

    Machine
    ExternalBeamMachine
    InternalBeamMachine
    register_machine


Beam limiting devices
---------------------

.. autosummary::
    :toctree: generated/machines/

    BeamLimitingDevice
    MLC
    Jaw


machines.photons
----------------

.. toctree::
   :maxdepth: 2

   machines_photons


machines.particles
------------------

.. toctree::
   :maxdepth: 2

   machines_particles


.. currentmodule:: pyRadPlan.machines

Loading and validating
----------------------
.. autosummary::
    :toctree: generated/machines/

    load_machine
    load_machine_from_mat
    load_from_name
    find_machine_file
    get_available_machines
    validate_machine
    add_machine_file
    remove_machine_file
    user_machine_dir
