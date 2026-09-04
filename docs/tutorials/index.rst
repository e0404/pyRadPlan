.. _Tutorials:

Tutorials
===================

The tutorials are generated from the example scripts in the ``examples/`` folder of the
repository. Each script is a `jupytext <https://jupytext.readthedocs.io>`_ notebook and can be
run directly with Python or opened as a notebook in Jupyter.

.. note::

   The notebooks are not executed while building this documentation; the shown outputs were
   produced separately (``python docs/execute_examples.py``) and may lag behind the scripts.
   Examples that need external tools (GUI, Monte Carlo engines, MATLAB/Octave, AI credentials)
   are shown without outputs.

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Examples:

   examples/*
