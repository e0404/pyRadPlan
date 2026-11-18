core
====

.. currentmodule:: pyRadPlan.core

.. autosummary::
   :toctree: generated/core/

   PyRadPlanBaseModel
   PyRadPlanError
   Grid
   resample_image
   resample_array
   ArrayBackendGridResampler
   resample_numpy_array


Configurable algorithms
-----------------------

.. autosummary::
   :toctree: generated/core/

   ConfigurableAlgorithm
   AlgorithmConfig
   AlgorithmParameterMetadata


Data directories
----------------

.. autosummary::
   :toctree: generated/core/

   get_data_dir
   get_data_subdir


Base data files
---------------

Machines, RBE tables, stopping power tables and phantoms are found by file name in the
folder shipped with pyRadPlan, in the same folder below the data directory and below each
shared folder of ``pyRadPlan.settings.base_data_path``. See :ref:`base_data_folders`.

.. autosummary::
   :toctree: generated/core/

   BASE_DATA_KINDS
   BaseDataKind
   BaseDataFile
   list_base_data
   find_base_data
   add_base_data_file
   remove_base_data_file
   bundled_base_data_dir
   user_base_data_dir
   shared_base_data_roots
   shared_base_data_dirs
   base_data_dirs
   get_base_data_kind
   check_base_data_file_name


Progress and compute control
----------------------------

.. autosummary::
   :toctree: generated/core/

   ProgressReporter
   ProgressReport
   ProgressLevel
   StatusReport
   ComputeReport
   ComputeControl
   ComputeCancelledError


np2sitk
-------

.. currentmodule:: pyRadPlan.core.np2sitk

.. autosummary::
   :toctree: generated/core/np2sitk/

   linear_indices_to_grid_coordinates
   linear_indices_to_image_coordinates
   linear_indices_to_sitk_mask
   sitk_mask_to_linear_indices


Resampling
----------

.. currentmodule:: pyRadPlan.core.resample

.. autosummary::
   :toctree: generated/core/resample/

   resample_image
   resample_array
   ArrayBackendGridResampler
   resample_numpy_array


For array input, use ``resample_array`` with either ``reference_image`` or
``reference_grid`` and exactly one target specification. Array axes are in
NumPy order: ``(z, y, x)`` for a 3D image. The returned array retains the input
namespace, device and dtype.

.. code-block:: python

   import SimpleITK as sitk
   from pyRadPlan.core.resample import resample_array

   result = resample_array(
       values,
       reference_grid=ct_grid,
       target_grid=dose_grid,
       interpolator=sitk.sitkLinear,
       extrapolate="nearest",
   )

``prefer_native=True`` (default) uses SimpleITK for full NumPy/CuPy volumes;
CuPy input is downloaded and the result uploaded to its original device. This
was faster for the measured larger full-volume workloads. Set
``prefer_native=False`` for supported 3D backend execution without that round
trip; unsupported modes or dtypes then raise ``NotImplementedError``.
Other SimpleITK enums, including BSpline, remain available through the native
path. See :class:`ArrayBackendGridResampler` for prepared, backend-resident
sampling of selected voxels or repeated geometry. ``resample_numpy_array`` is
a deprecated forwarding wrapper.


xp_utils
--------

.. currentmodule:: pyRadPlan.core.xp_utils

.. autosummary::
   :toctree: generated/core/xp_utils/

   cupy_available
   pytorch_gpu_available
   choose_array_api_namespace
   choose_device
   is_on_gpu
   is_host_device
   get_current_stream
   create_stream
   synchronize
   stream_wait_event
   record_event
   elapsed_time
   scatter
   to_numpy
   from_numpy
   to_namespace
   Array
   ArrayNamespace
   quantile
   interp1d
   interpnd
   ArrayBackendGridInterpolator
   ArrayBackend1DInterpolator
   jittable
   JittableKernel
   openblas_has_gemm_race
   warn_on_unreliable_openblas

Reusable grid interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``ArrayBackendGridInterpolator`` when querying the same rectilinear grid
repeatedly. It retains grid preparation and selects its implementation once:

.. code-block:: python

   import numpy as np
   from pyRadPlan.core.xp_utils import ArrayBackendGridInterpolator

   grid = np.array([0.0, 1.0, 3.0])
   values = grid[:, None] + 2 * grid[None, :]
   interpolate = ArrayBackendGridInterpolator(
       (grid, grid), values, bounds_error=True, prefer_native=True
   )
   result = interpolate(np.array([[0.25, 0.5], [2.0, 1.0]]))

The arrays determine the namespace and device. Native SciPy, CuPy, or JAX
implementations are preferred where available; ``prefer_native=False`` selects
the generic Array API implementation (currently 2D/3D, at least two points per
axis). ``interpolate.implementation`` reports the selected implementation.
The default ``bounds_error=False`` clips queries to the boundary; it does not
extrapolate or substitute a fill value. Query arrays must have shape ``(N, ndim)``
and share the grid's namespace and device. No implicit device transfers occur.

Input arrays must remain unchanged while the interpolator is used. Construct a
new object for a different grid or set of values. State belongs to the object;
there is no global cache retaining grid arrays. For a single query batch, use
``interpnd(query_points, points, values, ...)``; the existing import from
``pyRadPlan.core.xp_utils.compat`` remains available.
