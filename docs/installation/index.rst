.. _installation:

Installation
============

pyRadPlan requires **Python 3.10 – 3.13**.  Earlier versions are not supported.

.. contents:: On this page
   :local:
   :depth: 2

----

Setting up a virtual environment
---------------------------------

We strongly recommend working inside a dedicated virtual environment to avoid dependency
conflicts with other projects.

.. tab-set::

   .. tab-item:: venv (built-in)

      .. code-block:: bash

         python -m venv .venv

         # Linux / macOS
         source .venv/bin/activate

         # Windows (PowerShell)
         .venv\Scripts\Activate.ps1

      .. note::
         Name the environment ``.venv`` — all formatters and linters in pyRadPlan
         are pre-configured to ignore this folder automatically.

   .. tab-item:: conda / mamba

      .. code-block:: bash

         conda create -n pyradplan python=3.12
         conda activate pyradplan

----

Standard installation (PyPI)
-----------------------------

The recommended way to install pyRadPlan is from the Python Package Index, including
the graphical user interface:

.. code-block:: bash

   pip install "pyRadPlan[gui]"

This pulls in all mandatory dependencies — NumPy, SciPy, SimpleITK, pydantic,
pydantic-ai, and everything else needed to run a full treatment planning workflow on
the CPU with the NumPy backend — plus PySide6 and pyqtgraph for the interactive
matRad-style GUI.  It also provides the ``pyRadPlanGUI`` command:

.. code-block:: bash

   pyRadPlanGUI                    # start with an empty workspace
   pyRadPlanGUI path/to/TG119.mat  # load a matRad patient file on startup

Headless installation
~~~~~~~~~~~~~~~~~~~~~

On servers, compute clusters, CI runners, or in containers you typically neither need
the GUI nor want the sizable Qt runtime it ships with.  Omit the extra to install
the scripting/API core only:

.. code-block:: bash

   pip install pyRadPlan

The GUI degrades gracefully: ``import pyRadPlan`` works as usual, and calling a GUI
entry point without the extra raises an ``ImportError`` explaining how to install it.

----

Installation from source
--------------------------

Clone the repository and install in editable mode so that any changes to the source
are immediately reflected without reinstalling:

.. code-block:: bash

   git clone https://github.com/e0404/pyRadPlan.git
   cd pyRadPlan

   # create and activate a virtual environment first (see above)
   pip install -e .

The ``-e`` flag (editable) creates a link to the source tree rather than copying files
into ``site-packages``.

----

Optional extras
---------------

pyRadPlan provides several ``pip`` extras for optional features.  Install one or more
by appending them in brackets, e.g. ``pip install pyRadPlan[dev,gui]``.

.. list-table::
   :header-rows: 1
   :widths: 15 50 35

   * - Extra
     - What it adds
     - When to use
   * - ``[dev]``
     - pytest, coverage, pytest-cov, pytest-benchmark, pre-commit, sphinx + extensions
       (numpydoc, pydata-sphinx-theme, sphinx-design, autodoc_pydantic,
       sphinx-autodoc-typehints)
     - Development, testing, and documentation builds
   * - ``[gui]``
     - PySide6 ≥ 6.0.1, pyqtgraph ≥ 0.12.0
     - The matRad-style main GUI and interactive visualization widgets — recommended
       for all desktop installs (see :ref:`installation` above); skip on headless
       systems
   * - ``[numba]``
     - numba ≥ 0.61.0
     - JIT-compiled NumPy execution paths for selected hot-path kernels (currently in the
       Siddon ray tracer); enabled by default when installed, see
       :ref:`jit-compiled-kernels`
   * - ``[torch]``, ``[cupy]``, ``[jax]``
     - torch ≥ 2.0 / cupy ≥ 13.0 / jax ≥ 0.4.35
     - Alternate array-API compute backends.  Prefer the vendor-specific install
       commands under :ref:`gpu-backends` when you need a particular CUDA build
   * - ``[profiling]``
     - line_profiler ≥ 4.0
     - Running the line-profiling harnesses in ``benchmark/`` (see
       :ref:`benchmarks-profiling`)
   * - ``[matlab]``
     - matlabengine
     - Calling MATLAB / matRad directly from Python (see :ref:`matlab-octave`)
   * - ``[octave]``
     - oct2py
     - Calling GNU Octave from Python (see :ref:`matlab-octave`)

Example — full developer install from source:

.. code-block:: bash

   pip install -e .[dev,gui]

----

.. _gpu-backends:

GPU / alternate compute backends
---------------------------------

pyRadPlan's numerical algorithms are written against the
`Python Array API standard <https://data-apis.org/array-api/latest/>`_.  At runtime
the library automatically detects which backends are available and selects the most
capable one in this priority order:

1. **CuPy** (NVIDIA CUDA GPU) — highest priority when a GPU is present
2. **PyTorch** (NVIDIA CUDA GPU via ``torch.cuda``)
3. **JAX** (CUDA GPU via JAX's device list)
4. **NumPy** (CPU fallback — always available)

None of these GPU packages are installed automatically because they depend on the CUDA
toolkit version on your machine.  Follow the instructions below for whichever backend
you want.

CuPy (recommended for GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuPy provides a NumPy-compatible GPU array library.  You must install the wheel that
matches your CUDA toolkit version.  Run ``nvcc --version`` or check
``nvidia-smi`` to find your CUDA version, then install the matching wheel:

.. code-block:: bash

   # CUDA 12.x
   pip install cupy-cuda12x

   # CUDA 11.x
   pip install cupy-cuda11x

See the `CuPy installation guide <https://docs.cupy.dev/en/stable/install.html>`_ for
pre-built wheels for other CUDA versions and for conda-based installs.

PyTorch
~~~~~~~

PyTorch can serve as both a CPU and GPU backend.  The correct install command depends
on your OS and CUDA version; use the selector on the
`PyTorch website <https://pytorch.org/get-started/locally/>`_ to get the right
command.  A typical CUDA 12.x install looks like:

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cu124

For a CPU-only PyTorch backend:

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cpu

JAX
~~~

JAX is auto-detected when present.  A CPU build can be installed via the extra
(``pip install pyRadPlan[jax]``); for a CUDA build follow the
`JAX install guide <https://jax.readthedocs.io/en/latest/installation.html>`_, since the
right wheel depends on your CUDA version.

Verifying backend detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

After installing a GPU package, you can confirm that pyRadPlan sees it:

.. code-block:: python

   from pyRadPlan.core.xp_utils import (
       cupy_available, pytorch_gpu_available, jax_gpu_available,
       choose_array_api_namespace,
   )
   print("CuPy GPU:  ", cupy_available())
   print("Torch GPU: ", pytorch_gpu_available())
   print("JAX GPU:   ", jax_gpu_available())
   print("Active ns: ", choose_array_api_namespace())

----

Development setup
-----------------

After installing with ``[dev]``, activate the pre-commit hooks so every commit is
automatically formatted and linted:

.. code-block:: bash

   pre-commit install

Verify the installation:

.. code-block:: bash

   pre-commit --version
   pytest test

The test suite targets Python 3.10–3.13.  Run with coverage:

.. code-block:: bash

   pytest --cov=pyRadPlan test

Building the documentation locally:

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

----

.. _benchmarks-profiling:

Benchmarks and profiling
------------------------

Performance work lives in ``benchmark/`` and is deliberately kept out of the test
suite.  ``pytest`` is configured with ``testpaths = ["test"]``, so a bare ``pytest``
run never picks these up — you have to invoke them explicitly.

``benchmark_*.py``
   `pytest-benchmark <https://pytest-benchmark.readthedocs.io/>`_ harnesses, included
   in the ``[dev]`` extra.  Name the file directly, or override ``python_files`` to run
   the whole folder (the ``benchmark_`` prefix does not match pytest's default
   ``test_*.py`` pattern):

   .. code-block:: bash

      pytest benchmark/benchmark_interp1d.py
      pytest benchmark/ -o python_files="benchmark_*.py"

``profile_*.py``
   Standalone `line_profiler <https://kernprof.readthedocs.io/>`_ scripts with their
   own command-line interface.  They require the ``[profiling]`` extra:

   .. code-block:: bash

      pip install -e .[profiling]
      python benchmark/profile_pencilbeam_photon_forward_lineprof.py --runs 2

When adding new performance work, keep to these two prefixes so the separation from
``test/`` stays obvious.

----

.. _matlab-octave:

MATLAB and Octave interfaces
-----------------------------

pyRadPlan can call into MATLAB or GNU Octave to run matRad algorithms directly.  Both
interfaces are optional.

MATLAB (``matlabengine``)
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``matlabengine`` package version must match your MATLAB release.  Install the
matching version manually *before* running ``pip install pyRadPlan[matlab]`` to avoid
pip pulling an incompatible version:

.. code-block:: bash

   # Example: MATLAB R2024a ships engine 24.1.*
   pip install matlabengine==24.1.*
   pip install pyRadPlan[matlab]   # skips matlabengine — already satisfied

See `MATLAB Engine for Python <https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html>`_
for the exact version matrix.

GNU Octave (``oct2py``)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install pyRadPlan[octave]

``oct2py`` requires Octave to be installed on your system and accessible on the
``PATH``.

----

Troubleshooting
---------------

**"No module named 'pyRadPlan'"** after editable install
   Make sure your virtual environment is activated before running Python.  Run
   ``pip show pyRadPlan`` to confirm the install target.

**ImportError for SimpleITK on Apple Silicon**
   Use ``pip install SimpleITK --pre`` to get a pre-release wheel, or install via
   conda-forge: ``conda install -c conda-forge simpleitk``.

**CuPy import fails with CUDA driver error**
   The ``cupy-cudaXXx`` wheel must match the *driver*-supported CUDA version, not
   just the toolkit.  Run ``nvidia-smi`` and compare "CUDA Version" with the wheel
   you installed.

**PyTorch does not see the GPU**
   Confirm with ``python -c "import torch; print(torch.cuda.is_available())"`` and
   ensure the CUDA wheel matches your driver.

**RuntimeWarning: "jit execution of kernel ... failed" mentioning Triton**
   ``settings.xp.jit_backends`` includes ``"torch"``, but ``torch.compile`` needs a working
   Triton install (not available on all platforms, e.g. plain CPU/Windows PyTorch builds).
   This is a performance-only warning: the affected kernel falls back to its plain Array API
   implementation automatically. Either install a Triton build compatible with your PyTorch
   version, or drop ``"torch"`` from ``settings.xp.jit_backends`` /
   ``PYRADPLAN_XP_JIT_BACKENDS``. See :ref:`jit-compiled-kernels`.

**Pre-commit hook fails on first commit**
   The hook auto-fixes most style issues and then aborts the commit so you can review
   the changes.  Stage the auto-fixed files and commit again:

   .. code-block:: bash

      git add -u
      git commit -m "your message"
