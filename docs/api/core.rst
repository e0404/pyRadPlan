core
====

.. currentmodule:: pyRadPlan.core

.. autosummary::
   :toctree: generated/core/

   PyRadPlanBaseModel
   PyRadPlanError
   Grid
   resample_image
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
   resample_numpy_array


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
   jittable
   JittableKernel
   openblas_has_gemm_race
   warn_on_unreliable_openblas
