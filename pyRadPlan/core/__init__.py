"""Core classes and functions for pyRadPlan."""

from typing import TYPE_CHECKING

from .._lazy import attach_lazy
from ._exceptions import PyRadPlanError
from .datamodel import PyRadPlanBaseModel
from ._configurable import (
    AlgorithmConfig,
    AlgorithmParameterMetadata,
    ConfigurableAlgorithm,
    field_constraints,
    is_nested_instance,
    replace_nested,
)
from ._paths import DEFAULT_DATA_DIR, get_data_dir, get_data_subdir
from ._base_data import (
    BASE_DATA_KINDS,
    BaseDataFile,
    BaseDataKind,
    add_base_data_file,
    base_data_dirs,
    base_data_generation,
    bundled_base_data_dir,
    check_base_data_file_name,
    find_base_data,
    get_base_data_kind,
    list_base_data,
    remove_base_data_file,
    shared_base_data_dirs,
    shared_base_data_roots,
    user_base_data_dir,
)
from ._grids import Grid

# Settings initialization imports core paths. Loading resampling here would import
# xp_utils, which itself needs the settings singleton before it is constructed.
if TYPE_CHECKING:
    from .resample import (
        resample_image as resample_image,
        resample_array as resample_array,
        resample_numpy_array as resample_numpy_array,
        ArrayBackendGridResampler as ArrayBackendGridResampler,
    )
from ._progress import (
    ComputeCancelledError,
    ComputeControl,
    ComputeReport,
    ProgressLevel,
    ProgressReport,
    ProgressReporter,
    StatusReport,
    WarningReport,
    observe_control,
    observe_reports,
)

__all__ = [
    "PyRadPlanBaseModel",
    "PyRadPlanError",
    "Grid",
    "ArrayBackendGridResampler",
    "np2sitk",
    "resample_image",
    "resample_array",
    "resample_numpy_array",
    "DEFAULT_DATA_DIR",
    "get_data_dir",
    "get_data_subdir",
    "BASE_DATA_KINDS",
    "BaseDataFile",
    "BaseDataKind",
    "add_base_data_file",
    "base_data_generation",
    "base_data_dirs",
    "bundled_base_data_dir",
    "check_base_data_file_name",
    "find_base_data",
    "get_base_data_kind",
    "list_base_data",
    "remove_base_data_file",
    "shared_base_data_dirs",
    "shared_base_data_roots",
    "user_base_data_dir",
    "PyRadPlanBaseModel",
    "AlgorithmConfig",
    "AlgorithmParameterMetadata",
    "ConfigurableAlgorithm",
    "field_constraints",
    "is_nested_instance",
    "replace_nested",
    "Grid",
    "ProgressReporter",
    "ProgressReport",
    "ProgressLevel",
    "StatusReport",
    "WarningReport",
    "ComputeReport",
    "ComputeCancelledError",
    "ComputeControl",
    "observe_reports",
    "observe_control",
]

_RESAMPLE_EXPORTS = (
    "resample_image",
    "resample_array",
    "resample_numpy_array",
    "ArrayBackendGridResampler",
)
__getattr__, __dir__, __all__ = attach_lazy(
    __name__,
    {".resample": _RESAMPLE_EXPORTS},
    eager=(name for name in __all__ if name not in _RESAMPLE_EXPORTS),
)
