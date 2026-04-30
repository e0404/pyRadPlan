.. _user_guide:

User guide
==========
.. toctree::
    :maxdepth: 2
    :caption: Contents:

    introduction
    installation
    quickstart

============
Introduction
============

pyRadPlan is a multi-modality treatment planning toolkit in python born from the established
Matlab-based toolkit `matRad <http://www.matRad.org>`_.
As such, pyRadPlan aims to provide a framework as well as tools for combining dose calculation with
optimization with focus on ion planning.

### Data Structures
pyRadPlan uses a similar datastructure and workflow concept as in matRad, while trying to ensure
that the corresponding datastructures can be easily imported and exported from/to matRad.
This facilitates the application of either algorithms from matRad or native pyRadPlan at any stage
of the treatment planning workflow.

To enforce valid datastructures, we perform validation and serialization with
`pydantic <https://docs.pydantic.dev/latest/>`_.
Datastructures and algorithms rely mostly on `SimpleITK <https://simpleitk.readthedocs.io>`_,
`numpy <https://numpy.org/>`_, and `scipy <https://scipy.org/>`_ for internal data representation
and processing.

============
Installation
============

pyRadPlan can easily be installed from pip

1. Ensure you have Python 3.9 or higher installed. We recommend working in a virtual environment.
2. Install pyRadPlan using pip:
    .. code-block:: bash

        pip install pyRadPlan

============================================
Quickstart: The first minimal treatment plan
============================================

Once you have installed pyRadPlan, you can start using it by importing the package in your Python
script:

.. code-block:: python

    import pyRadPlan

The top-level API exports major workflow functions and data structures for generating a treatment
plan and allows to create a basic treatment plan with just a few lines of code, very similar to
matRad's example `matRad.m <https://github.com/e0404/matRad/blob/master/matRad.m>`_ script:

.. code-block:: python

    from importlib import resources
    import pymatreader
    from pyRadPlan import load_patient, IonPlan, generate_stf, calc_dose_influence, fluence_optimization, plot_slice

    #  Read patient from provided TG119.mat file and validate data
    tg119_path = resources.files("pyRadPlan.data.phantoms").joinpath("TG119.mat")
    ct, cst = load_patient(tg119_path)

    # Create a plan object
    pln = IonPlan(radiation_mode="protons", machine="Generic")

    # Generate Steering Geometry ("stf")
    stf = generate_stf(ct, cst, pln)

    # Calculate Dose Influence Matrix ("dij")
    dij = calc_dose_influence(ct, cst, stf, pln)

    # Fluence Optimization (objectives loaded from "cst")
    fluence = fluence_optimization(ct, cst, stf, dij, pln)

    # Result
    result = dij.compute_result_ct_grid(fluence)

    # Visualize
    plot_slice(ct=ct, cst=cst, overlay=result["physical_dose"], overlay_unit="Gy")
