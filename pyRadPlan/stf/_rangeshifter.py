from typing import Any
import numpy as np
from pydantic import Field, field_serializer, SerializerFunctionWrapHandler, FieldSerializationInfo
from pyRadPlan.core import PyRadPlanBaseModel


class RangeShifter(PyRadPlanBaseModel):
    """
    Class representing a range shifter.

    This class extends Pydantic's BaseModel and provides functionality to
    handle the range shifter information, including properties like id, equivalent thickness,
    and source to range shifter distance.

    Attributes
    ----------
    id : int
        The identifier for the range shifter.
    eq_thickness : float
        The equivalent thickness of the range shifter.
    source_rashi_distance : float
        The distance from the source to the range shifter.
    """

    id: int = Field(alias="ID", default=0)
    eq_thickness: float = Field(default=0.0)
    source_rashi_distance: float = Field(default=0.0)

    @field_serializer("*", mode="wrap")
    def _field_typing(
        self, v: Any, handler: SerializerFunctionWrapHandler, info: FieldSerializationInfo
    ) -> Any:
        """Serialize with possible matRad context."""

        context = info.context
        if context and context.get("matRad") == "mat-file":
            return np.float64(v)  # Ensure double for MATLAB/matRad

        return handler(v, info)
