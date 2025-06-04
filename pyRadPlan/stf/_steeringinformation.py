from typing import Any, Union
from typing_extensions import Self
import numpy as np
from pydantic import (
    SerializationInfo,
    model_validator,
    field_serializer,
    ValidatorFunctionWrapHandler,
    ValidationInfo,
    ValidationError,
)
from numpydantic import NDArray, Shape
from pyRadPlan.stf._beam import Beam
from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.util.helpers import models2recarray


class SteeringInformation(PyRadPlanBaseModel):
    """
    A class representing the Steering Information (stf).

    This class extends PyRadPlanBaseModel (based on pydantic) and provides functionality to
    handle single beams, including their properties.
    These are defined in the corresponding class (_Beam.py).

    Attributes
    ----------
    beams : List[Beam] - list consisting of Beam objects (pydantic)
        beam class object containing the properties of the beam.

    Methods
    -------
    validate_model_input(data: Any) -> Any
        Validates the input data before creating the model instance.

    to_matrad() -> dict
        Creates a dictionary ready to save the stf model to a mat-file that can be read.
    """

    beams: list[Beam]

    # Validation
    @model_validator(mode="wrap")
    @classmethod
    def validate_model_input(
        cls, data: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
    ) -> Self:
        """
        Validate the input data for creating the model instance.

        Will first try to run pydantics handler, and if it fails with a
        ValidationError, it will
        try to convert the data to the right format.
        """
        try:
            return handler(data, info)
        except ValidationError:
            # Check if import is from matlab
            # If from matlab but stf already choosen, the code is executed as normal
            matrad_format = ["__header__", "__version__", "__globals__"]
            if all(key in data for key in matrad_format):
                # struct of matlab usual in 4. position. #TODO: are there exceptions?
                data = data[list(data.keys())[3]]

            if isinstance(data, dict) and "beams" not in data:
                # This code is needed to pass the right format of the stf from matRad
                try:
                    length_lists = [len(v) for v in data.values()]
                except (TypeError, ValueError):
                    return handler({"beams": [data]}, info)  # TODO: only works if its a list??

                if len(set(length_lists)) == 1 and len(length_lists) > 1:
                    beam_list = []
                    length_lists = []

                    for v in data.values():
                        if isinstance(v, list):
                            length_lists.append(len(v))

                        # this exception is necessary if imported stf from matlab is only one beam
                        if isinstance(v, int or float):
                            length_lists.append(1)

                    for i in range(length_lists[0]):
                        beam = []

                        for key in data.keys():
                            entry = [key, data[key][i]]
                            beam.append(entry)

                        beam_list.append(dict(beam))

                    return handler({"beams": beam_list}, info)

            if isinstance(data, list):
                return handler({"beams": data}, info)

            return handler(data, info)
        except Exception as exc:
            raise exc

    @field_serializer("beams")
    def custom_beams_serializer(self, v: list[Beam], info: SerializationInfo) -> Any:
        """Serialize the beams fields in various contexts."""
        context = info.context
        if context and context.get("matRad") == "mat-file":
            override_types = {"rays": np.recarray}
            beams_recarray = models2recarray(
                v, override_types=override_types, serialization_context=context
            )
            return beams_recarray
        return [
            beam.model_dump(
                by_alias=info.by_alias,
            )
            for beam in v
        ]

    def to_matrad(self, context: str = "mat-file") -> Any:
        export = super().to_matrad(context=context)
        return export["beams"]

    @property
    def num_of_beams(self) -> int:
        return len(self.beams)

    @property
    def num_of_rays(self) -> int:
        return sum([beam.num_of_rays for beam in self.beams])

    @property
    def total_number_of_bixels(self) -> int:
        return sum([beam.total_number_of_bixels for beam in self.beams])

    @property
    def bixel_beam_index_map(self) -> NDArray[Shape["1-*"], np.int64]:
        """Mapping of bixels to their respective beam index."""
        tmp_map = np.zeros(self.total_number_of_bixels, dtype=np.int64)
        start = 0
        for b, beam in enumerate(self.beams):
            tmp_map[start : start + beam.total_number_of_bixels] = b
            start += beam.total_number_of_bixels
        return tmp_map

    @property
    def bixel_ray_index_per_beam_map(self) -> NDArray[Shape["1-*"], np.int64]:
        """Mapping of bixels to the ray index in the individual beams."""
        tmp_map = np.zeros(self.total_number_of_bixels, dtype=np.int64)
        start = 0
        for beam in self.beams:
            tmp_map[start : start + beam.total_number_of_bixels] = beam.bixel_ray_map
            start += beam.total_number_of_bixels
        return tmp_map

    @property
    def bixel_index_per_beam_map(self) -> NDArray[Shape["1-*"], np.int64]:
        """Mapping of bixels to their bixel index in the respective beam."""
        tmp_map = np.zeros(self.total_number_of_bixels, dtype=np.int64)
        start = 0
        for beam in self.beams:
            tmp_map[start : start + beam.total_number_of_bixels] = np.arange(
                beam.total_number_of_bixels
            )
            start += beam.total_number_of_bixels
        return tmp_map


def create_stf(
    stf: Union[dict[str, Any], SteeringInformation, None] = None, **kwargs
) -> SteeringInformation:
    """
    Create a Steering Information object.

    Parameters
    ----------
    stf : Union[dict[str, Any], None]
        dictionary containing the data to create the stf object.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    SteeringInformation
        A SteeringInformation class object.

    Raises
    ------
    ValueError
        If the radiation mode is unknown or empty.
    """
    if stf:
        # If data is already a Stf object, return it directly
        if isinstance(stf, SteeringInformation):
            return stf
        return SteeringInformation.model_validate(stf)

    return SteeringInformation(**kwargs)  # not tested


def validate_stf(
    stf: Union[dict[str, Any], SteeringInformation, None] = None, **kwargs
) -> SteeringInformation:
    """
    Validate a Steering Information object.

    Synonym to create_stf but should be used in validation context.

    Parameters
    ----------
    stf : Union[dict[str, Any], None]
        dictionary containing the data to create the stf object.

    Returns
    -------
    SteeringInformation
        A validated SteeringInformation class object.
    """
    return create_stf(stf, **kwargs)
