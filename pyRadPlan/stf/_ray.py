"""Defines a class representing a single ray.

The ray is pointing from the beam source to a position in the patient.
"""

import functools
from typing import Any, Union, Optional
import numpy as np
from pydantic import (
    create_model,
    Field,
    model_validator,
    field_validator,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    field_serializer,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    computed_field,
    ValidationError,
)
from numpydantic import NDArray, Shape
from pyRadPlan.util.helpers import dl2ld, ld2dl
from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.stf._beamlet import Beamlet


class Ray(PyRadPlanBaseModel):
    """
    A class representing a single ray.

    This class extends PyRadPlanBaseModel (pydantic) and provides functionality to
    handle the ray information, including properties like ray positions, energy, focus index, etc.

    Attributes
    ----------
    ray_pos_bev : np.ndarray
        The ray positions in BEV coordinates.
    ray_pos : np.ndarray
        The ray positions in (x, y, z) coordinates.
    target_point_bev : np.ndarray
        The target point in BEV coordinates.
    target_point : np.ndarray
        The target point in (x, y, z) coordinates.
    beamlets : list[Beamlet]
        The beamlets in the ray.
    """

    beamlets: list[Beamlet]

    ray_pos_bev: NDArray[Shape["3"], np.float64] = Field(alias="rayPos_bev")
    ray_pos: NDArray[Shape["3"], np.float64]

    target_point: Optional[NDArray[Shape["3"], np.float64]] = Field(default=None)
    target_point_bev: Optional[NDArray[Shape["3"], np.float64]] = Field(
        alias="targetPoint_bev", default=None
    )

    @field_validator("ray_pos_bev", "target_point_bev", "ray_pos", "target_point", mode="wrap")
    @classmethod
    def validate_nparray_dtype(
        cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
    ) -> NDArray[Shape["3"], np.float64]:
        """Validate / convert arrays to have floating point values."""

        if v is not None:
            if not isinstance(v, np.ndarray):
                v = np.array(v, dtype=float)
            if not np.issubdtype(v.dtype, np.floating):
                v = v.astype(float)
            v = v.reshape((3,))
        return handler(v, info)

    # Custom validator to convert list to np.ndarray
    @model_validator(mode="wrap")
    @classmethod
    def sanitize_beamlet_structure(
        cls, data: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
    ) -> Any:
        """
        Sanitize the beamlet structure in the ray.

        It may be structured differently (e.g. when coming from matRad.).
        """
        # if isinstance(data, list):
        #     for i in range(len(data)):
        #         if isinstance(data[i], int) or isinstance(data[i], float):
        #             data[i] = [float(data[i])]
        #         if isinstance(data[i], list):
        #             data[i] = np.array(data[i])

        try:
            return handler(data, info)
        except ValidationError:
            if isinstance(data, dict):
                # We obtain some meta information about the Beamlet model
                beamlet_fields = Beamlet.model_fields
                beamlet_field_aliases = {
                    field.validation_alias: key for key, field in beamlet_fields.items()
                }

                # Beamlets may be structured differently in the ray, so we need to extract them
                # into a dictionary containing a list of values (or other dicts) for each beamlet
                # property
                beamlet_subdict = {}
                to_remove = []

                for key in data:
                    if key in beamlet_fields.keys() or key in beamlet_field_aliases.keys():
                        if key in beamlet_field_aliases.keys():
                            update_key = beamlet_field_aliases[key]
                        else:
                            update_key = key
                        value = data[key]

                        if isinstance(value, dict):
                            try:
                                value = dl2ld(
                                    value, type_check=True
                                )  # dict_of_lists to list_of_dicts
                            except TypeError:
                                value = [value]
                                # This is an exception for the case where the beamlet is a single
                                # beamlet

                        # now, if it is not a list, we make it a list
                        if not isinstance(value, list) and not isinstance(value, np.ndarray):
                            value = [value]

                        beamlet_subdict.update({update_key: value})

                        to_remove.append(key)

                # Sanitze data to not have beamlet properties as arrays in the ray
                for key in to_remove:
                    data.pop(key)

                # correct indexing for focus_ix:
                if "focus_ix" in beamlet_subdict:
                    beamlet_subdict["focus_ix"] = [ix - 1 for ix in beamlet_subdict["focus_ix"]]

                try:
                    beamlets = dl2ld(beamlet_subdict, type_check=False)
                except TypeError as exc:
                    raise TypeError(f"Beamlet information not consistent in Ray: {exc}") from exc

                data["beamlets"] = beamlets

            return handler(data)
        except Exception as exc:
            raise exc

    @field_serializer("beamlets", mode="wrap")
    def custom_beamlets_serializer(
        self, v: list[Beamlet], handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> Union[dict[str, Any], list]:
        context = info.context
        if context and context.get("matRad") == "mat-file":
            beamlets_dump = [beamlet.to_matrad(context=context["matRad"]) for beamlet in v]

            # Convert the list of dictionaries to a dictionary of lists and return
            beamlets_dump = ld2dl(beamlets_dump, type_check=False)

            for field in beamlets_dump:
                first_element = beamlets_dump[field][0]
                if isinstance(first_element, dict):
                    field_dump = ld2dl(beamlets_dump[field], type_check=False)
                    beamlets_dump[field] = np.rec.fromarrays(
                        field_dump.values(), names=list(field_dump)
                    )
            return beamlets_dump
        return handler(v, info)

    def to_matrad(self, context: Union[str, dict] = "mat-file") -> Any:
        """Serialize rays for matRad structure."""

        model_dump = super().to_matrad(context=context)

        for key in model_dump["beamlets"]:
            model_dump[key] = model_dump["beamlets"][key]

        model_dump.pop("beamlets")
        return model_dump

    @classmethod
    def create_matrad_helper_model(cls):
        """Create a helper model for matRad serialization."""

        def get_property_list(self: Ray, name: str):
            return [getattr(beamlet, name) for beamlet in self.beamlets]

        beamlet_fields = Beamlet.model_fields
        prop_lambdas = {}
        for field in beamlet_fields:
            prop_lambdas[field] = computed_field(
                functools.partial(get_property_list, name=field),
                return_type=list[beamlet_fields[field].annotation],
                alias=beamlet_fields[field].serialization_alias,
            )

        return create_model("RayMatRadHelper", __base__=Ray, **prop_lambdas)
