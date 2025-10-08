from typing import Optional, Annotated
from datetime import datetime
from pydantic import (
    Field,
    StringConstraints,
    field_validator,
)

from pyRadPlan.core import PyRadPlanBaseModel


class Machine(PyRadPlanBaseModel):
    """Base class for Machine objects.

    Defines minimum meta-data a machine must hold:

    Attributes
    ----------
    radiation_mode : str
        The radiation mode of the machine.
    description : str
        The description of the machine.
    machine : str
        The name of the machine.
    """

    radiation_mode: str = Field()
    description: str = Field(default="")
    name: Annotated[str, StringConstraints(min_length=1)] = Field(
        alias="machine", default="Generic"
    )
    created_on: Optional[datetime] = Field(default=None)
    last_modified: Optional[datetime] = Field(default=None)
    created_by: Optional[str] = Field(default="")
    last_modified_by: Optional[str] = Field(default="")
    data_type: Optional[str] = Field(default="-")
    version: Annotated[str, StringConstraints(pattern=r"^\d+\.\d+\.\d+$")] = Field(
        default="0.0.1", validate_default=True
    )

    # Abstract property handled by the individual machines:
    _possible_radiation_modes: list[str]

    @field_validator("created_on", "last_modified", mode="before")
    @classmethod
    def validate_datetime_variants(cls, v):
        # If it is a string, we try some additional formats in addition to
        # pydantics accepted datetime values
        # For example, matRad macines use the format "%d-%b-%Y" for some dates
        if isinstance(v, str):
            try_formats = ["%d-%b-%Y"]

            for fmt in try_formats:
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    pass

        return v
