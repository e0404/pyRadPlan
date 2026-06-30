"""Squared Mimicking Objective."""

from typing import Annotated, Literal, Union

from pydantic import Field

import array_api_compat
import SimpleITK as sitk


from ...core.xp_utils.typing import Array

from ._objective import Objective, ParameterMetadata
from pyRadPlan.core import Grid

# %% Class definition


class SquaredMimicking(Objective):
    """
    Squared Deviating Mimicking (least-squares) objective.

    Attributes
    ----------
    d_ref : sitk.Image or tuple[Array, Grid]
        dose reference image (array)
    """

    name: Literal["Squared Mimicking"] = "Squared Mimicking"

    def default():
        image = sitk.Image([1, 1, 1], sitk.sitkFloat32)
        image.SetPixel(0, 0, 0, 60.0)
        return image

    d_ref: Annotated[
        Union[sitk.Image, tuple[Array, Grid]],
        Field(default_factory=default),
        ParameterMetadata(kind="image_reference"),
    ]

    def compute_objective(self, values: Array) -> Array:
        d_ref = self._resampled_image_reference_cache["d_ref"]
        deviation = values - d_ref
        return (deviation @ deviation) / array_api_compat.size(values)

    def compute_gradient(self, values: Array) -> Array:
        d_ref = self._resampled_image_reference_cache["d_ref"]
        return 2.0 * (values - d_ref) / array_api_compat.size(values)
