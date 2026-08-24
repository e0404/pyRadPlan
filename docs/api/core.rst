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
   get_current_stream
   create_stream
   synchronize
   record_event
   elapsed_time
   to_numpy
   from_numpy
   to_namespace
   Array
   ArrayNamespace
   quantile
   interp1d
