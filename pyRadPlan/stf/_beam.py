from typing import Any
import numpy as np
from pydantic import (
    Field,
    model_validator,
    field_validator,
    computed_field,
    field_serializer,
    SerializationInfo,
)

from numpydantic import NDArray, Shape

from pyRadPlan.stf._ray import Ray
from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.util.helpers import models2recarray


class Beam(PyRadPlanBaseModel):
    """
    A class representing a single beam.

    This class extends PyRadPlanBaseModel (Pydantic) and provides functionality to
    handle the steering information, including properties like gantry angle, couch angle, etc.

    Attributes
    ----------
    gantry_angle :
        The gantry angle of the beam in (°).
    couch_angle :
        The couch angle of the beam in (°).
    bixel_width :
        The width of the bixels in (mm).
    radiation_mode :
        The radiation mode of the beam (e.g. photon, proton, carbon).
    machine :
        The machine used for the beam. (e.g. 'Generic')
    sad :
        The source to axis distance in (mm).
    iso_center :
        The isocenter of the beam in (x, y, z) coordinates.
    num_of_rays :
        The number of rays in the beam.
    ray :
        dict containing the properties of each ray.
    source_point_bev :
        The source point in BEV coordinates.
    source_point :
        The source point in (x, y, z) coordinates.
    num_of_bixels_per_ray :
        The number of bixels per ray in an array (1 x num_of_rays).
    longitudinal_spot_spacing :
        The longitudinal spot spacing in (mm).
    total_number_of_bixels :
        The total number of bixels in the beam.

    Methods
    -------
    create_beam():
        Validate and create a Beam object.
    validate_beam():
        Validate and create a Beam object.
    """

    gantry_angle: float = Field(default=0)
    couch_angle: float = Field(default=0)  # , gt=-100, lt=100)
    bixel_width: float = Field(default=5)
    radiation_mode: str = Field(default="protons")
    machine: str = Field(default="Generic")
    sad: float = Field(alias="SAD", default=100000)
    iso_center: NDArray[Shape["3"], np.float64]
    rays: list[Ray] = Field(
        alias="ray"
    )  # alias needed for matRad import. Can also be done in the model_validator

    source_point_bev: NDArray[Shape["3"], np.float64] = Field(
        alias="sourcePoint_bev", default=([0, -10000, 0]), validate_default=True
    )
    source_point: NDArray[Shape["3"], np.float64] = Field(
        default=([0, 0, 0]), validate_default=True
    )
    longitudinal_spot_spacing: float = Field(default=2.0)

    @model_validator(mode="before")
    @classmethod
    def validate_model_input(cls, data: Any) -> Any:
        # isinstance needed to exclude integers
        if isinstance(data, dict):
            if "ray" in data and isinstance(data["ray"], dict):
                ray_dict = data["ray"]

                # these are computed properties in python. Can be removed
                if "totalNumOfBixels" in ray_dict:
                    ray_dict.pop("totalNumOfBixels")
                if "numOfBixelsPerRay" in ray_dict:
                    ray_dict.pop("numOfBixelsPerRay")
                if "numOfRays" in ray_dict:
                    ray_dict.pop("numOfRays")

                length_lists = []

                for v in ray_dict.values():
                    if isinstance(v, list):
                        length_lists.append(len(v))

                    # this exception is necessary if imported stf from matlab is only one beam
                    # TODO: Test if this is necessary
                    if isinstance(v, int or float):
                        length_lists.append(1)

                # TODO: This error message might be confusing in some cases
                if len(set(length_lists)) != 1:
                    raise ValueError(
                        "All values in the 'ray' dictionary must have the same length"
                    )

                if len(set(length_lists)) == 1:
                    ray_list = []

                    for i in range(length_lists[0]):
                        ray = []

                        for key in ray_dict.keys():
                            entry = [key, ray_dict[key][i]]
                            ray.append(entry)

                        ray_list.append(dict(ray))

                    data.pop("ray")
                    data["rays"] = ray_list
        return data

    @field_validator("source_point", "source_point_bev", "iso_center", mode="before")
    @classmethod
    def validate_nparray_dtype(cls, v: Any) -> Any:
        """Validate arrays to have floating point values."""
        v = np.asarray(v, dtype=np.float64)
        return v.reshape((3,))

    @field_validator("rays", mode="after")
    @classmethod
    def validate_rays(cls, v: list[Ray]) -> list[Ray]:
        """
        Validate the rays attribute.

        Pydantic will validate that it is a list but will also allow list of dicts
        """
        try:
            return [Ray.model_validate(ray) for ray in v]
        except TypeError as e:
            raise ValueError(f"Error validating rays: {e}")

    @computed_field
    @property
    def num_of_bixels_per_ray(self) -> np.ndarray:
        return np.array([len(ray.beamlets) for ray in self.rays])

    @computed_field
    @property
    def num_of_rays(self) -> int:
        return len(self.rays)

    @computed_field(alias="totalNumOfBixels")
    @property
    def total_number_of_bixels(self) -> int:
        return int(sum(self.num_of_bixels_per_ray))

    @property
    def bixel_ray_map(self) -> NDArray[Shape["1-*"], np.int64]:
        """Map providing ray index in the beam for each bixel."""
        return np.repeat(np.arange(len(self.rays)), self.num_of_bixels_per_ray)

    # serialization
    @field_serializer("rays")
    def custom_rays_serializer(self, v: list[Ray], info: SerializationInfo) -> Any:
        context = info.context
        if context and context.get("matRad") == "mat-file":
            helper_model = Ray.create_matrad_helper_model()
            override_types = {"range_shifter": np.recarray, "beamlets": None}
            rays_matrad = [helper_model.model_validate(ray) for ray in v]
            rays_recarray = models2recarray(
                rays_matrad, override_types=override_types, serialization_context=context
            )
            # override_types = get_type_hints(Beamlet)
            # rays_recarray = models2recarray(v, serialization_context=context)
            return rays_recarray
        return [ray.model_dump(by_alias=info.by_alias) for ray in v]


def create_beam():
    """Validate and create a Beam object."""
    # TODO
    pass


def validate_beam():
    """Validate and create a Beam object."""
    # TODO
    pass
