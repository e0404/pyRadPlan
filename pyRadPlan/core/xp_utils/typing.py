"""Typing for Array API Arrays compatible with pydantic validation."""

from __future__ import annotations
from typing import TYPE_CHECKING, Annotated, Any
import array_api_compat
from pydantic_core import core_schema, PydanticCustomError

# --- Static type (only visible to type checkers) ---
if TYPE_CHECKING:
    # Use the official Protocol for IDE/mypy/pyright.
    # Import ONLY for type checking (won't run at runtime).
    from array_api._2024_12 import Array as ArrayType  # or _2023_12, etc.
    import array_api._2024_12 as array_api_types  # type: ignore[import]

    ArrayNamespace = array_api_types.ArrayNamespace  # type: ignore[no-redef]
    ArrayNamespaceFull = array_api_types.ArrayNamespaceFull  # type: ignore[no-redef]
else:
    # Runtime placeholder (harmless stand-in)
    class ArrayType:  # noqa: D401
        """Runtime placeholder for the Array Protocol."""

        pass

    class ArrayNamespace:  # noqa: D401
        """Runtime placeholder for the ArrayNamespace Protocol."""

        pass

    class ArrayNamespaceFull:  # noqa: D401
        """Runtime placeholder for the ArrayNamespaceFull Protocol."""

        pass


# --- Runtime nominal type for Pydantic v2 ---
class ArrayAPIArray:
    """
    Nominal runtime type representing an Array API–compatible array.

    Pydantic will use our validator; static checkers will see ArrayProto via Annotated.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        """Override to provide a custom core schema for array validation."""

        def validate(value: Any):
            if array_api_compat.is_array_api_obj(value):
                return value

            raise PydanticCustomError("array_api", "Value is not an Array API–compatible array")

        return core_schema.no_info_plain_validator_function(validate)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema_, handler):
        """Override to generate a sensible JSON schema."""
        js = handler(core_schema_)
        js.update({"title": "ArrayAPIArray", "type": "object"})
        return js


Array = Annotated[ArrayType, ArrayAPIArray]  # <- export this
