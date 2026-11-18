"""Basic Model for all pyRadPlan Datastructures."""

from typing import Any, Union
import numpy as np
from pydantic import (
    AliasGenerator,
    BaseModel,
    ConfigDict,
)
from pydantic.alias_generators import to_camel
from copy import deepcopy
import json

import array_api_compat
import SimpleITK as sitk
import scipy.sparse as sp

from ._serialization import ARRAY_TAG, decode_tree, to_json_tree


def _has_array_tag(obj: Any) -> bool:
    """Whether a tree of dicts, lists and tuples contains a tagged array leaf.

    A tree holding an array or image is native data (e.g. a python-mode dump), not a JSON
    dump, so the scan stops there; this keeps validating large native trees cheap.
    """
    stack = [obj]
    while stack:
        item = stack.pop()
        if isinstance(item, (np.ndarray, sitk.Image)):
            return False
        if isinstance(item, dict):
            if ARRAY_TAG in item:
                return True
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return False


def decode_input(data: Any) -> Any:
    """Decode the tagged array leaves of a JSON-mode dump (see :meth:`model_dump`).

    Data without tagged leaves is returned unchanged. The ``validate_*`` factories and
    :meth:`PyRadPlanBaseModel.model_validate` call this; plain constructors do not.
    """
    return decode_tree(data) if _has_array_tag(data) else data


def _images_equal(a: sitk.Image, b: sitk.Image) -> bool:
    """Whether two images share geometry, pixel type and voxel values."""
    if (
        a.GetDimension() != b.GetDimension()
        or a.GetSize() != b.GetSize()
        or a.GetPixelID() != b.GetPixelID()
    ):
        return False
    if not (
        np.allclose(a.GetSpacing(), b.GetSpacing())
        and np.allclose(a.GetOrigin(), b.GetOrigin())
        and np.allclose(a.GetDirection(), b.GetDirection())
    ):
        return False
    return bool(np.array_equal(sitk.GetArrayViewFromImage(a), sitk.GetArrayViewFromImage(b)))


class PyRadPlanBaseModel(BaseModel):
    """
    Base class for all pyRadPlan data structures.

    Especially useful for structures that should be matRad compatible.
    Extends Pydantic's BaseModel to use pydantic validation and serialization.

    Attributes
    ----------
    model_config : ConfigDict
        Configuration for the model, including alias generation, population by
        name, arbitrary types allowed, assignment validation, and attribute
        creation from dictionary.
    """

    model_config = ConfigDict(
        alias_generator=AliasGenerator(alias=to_camel),
        populate_by_name=True,  # Allows both snake_case and camelCase attributes
        arbitrary_types_allowed=True,  # Allows arbitrary types in the model (will be casted)
        validate_assignment=True,  # Validate assignment of values to fields
        # (not only during construction)
        from_attributes=True,  # Allows to create a model from a dictionary
    )

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> Any:
        """
        Validate an object, decoding the tagged array leaves of a JSON-mode dump first.

        See :meth:`pydantic.BaseModel.model_validate` for the arguments.
        """
        return super().model_validate(decode_input(obj), **kwargs)

    @classmethod
    def model_validate_json(cls, json_data: Union[str, bytes, bytearray], **kwargs: Any) -> Any:
        """
        Validate a JSON string written by :meth:`model_dump_json`.

        See :meth:`pydantic.BaseModel.model_validate_json` for the arguments.
        """
        return cls.model_validate(json.loads(json_data), **kwargs)

    def model_dump(self, *, mode: str = "python", **kwargs: Any) -> dict[str, Any]:
        """
        Dump the model to a dictionary.

        In ``mode="json"`` arrays, sparse matrices and images are replaced by tagged
        dictionaries (see :mod:`pyRadPlan.core._serialization`), so the result can be passed
        to :func:`json.dumps` and read back with ``model_validate``. All other arguments are
        those of :meth:`pydantic.BaseModel.model_dump`.
        """
        if mode != "json":
            return super().model_dump(mode=mode, **kwargs)
        return to_json_tree(super().model_dump(mode="python", **kwargs))

    def model_dump_json(self, *, indent: Union[int, None] = None, **kwargs: Any) -> str:
        """
        Dump the model to a standard JSON string (see :meth:`model_dump`).

        Non-finite floats are written as strings and tagged leaves, not as ``NaN`` constants.
        """
        return json.dumps(self.model_dump(mode="json", **kwargs), indent=indent, allow_nan=False)

    def __eq__(self, other: Any) -> bool:
        """
        Specialized __eq__ method to compare two pyRadPlanBaseModel instances.

        It first tries to compare the instances using the super().__eq__ method.
        If this fails, it compares the dictionaries. This is due to some issues
        comparing numpy arrays within the models.
        """
        # SimpleITK's image comparison returns an image (always truthy) or raises for
        # different dimensions, so models holding images are compared entry by entry.
        if any(isinstance(v, sitk.Image) for v in self.__dict__.values()):
            return self._eq_dicts(other)
        try:
            return super().__eq__(other)
        except (ValueError, TypeError, RuntimeError):
            return self._eq_dicts(other)

    def _eq_dicts(self, other: Any) -> bool:
        """Compare the field dictionaries entry by entry (arrays, images and nested dicts)."""
        if not isinstance(other, PyRadPlanBaseModel):
            return False
        if self.__dict__.keys() != other.__dict__.keys():
            return False
        stack = [(self.__dict__, other.__dict__)]
        while stack:
            dict_a, dict_b = stack.pop()
            if dict_a.keys() != dict_b.keys():
                return False
            for key in dict_a:
                if isinstance(dict_a[key], dict) and isinstance(dict_b[key], dict):
                    stack.append((dict_a[key], dict_b[key]))
                elif not self._eq_dict_entry(dict_a, dict_b, key):
                    return False
        return True

    def __ne__(self, other: Any) -> bool:
        """
        Specialized __ne__ method to compare two PyRadPlanBaseModel instances.

        This method returns the negation of the __eq__ method.
        """
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        else:
            return True

    def _eq_dict_entry(self, dict_a: dict, dict_b: dict, key: Any) -> bool:
        """Compare two dictionary entries for equality."""

        if isinstance(dict_a[key], sitk.Image) or isinstance(dict_b[key], sitk.Image):
            return (
                isinstance(dict_a[key], sitk.Image)
                and isinstance(dict_b[key], sitk.Image)
                and _images_equal(dict_a[key], dict_b[key])
            )
        if (
            isinstance(dict_a[key], np.ndarray)
            and isinstance(dict_b[key], np.ndarray)
            and dict_a[key].dtype == object
            and dict_b[key].dtype == object
        ):
            return self._eq_object_arrays(dict_a[key], dict_b[key])
        elif array_api_compat.is_array_api_obj(dict_a[key]) and array_api_compat.is_array_api_obj(
            dict_b[key]
        ):
            try:
                xp = array_api_compat.array_namespace(dict_a[key], dict_b[key])
            except (ValueError, TypeError):
                return False
            if dict_a[key].shape != dict_b[key].shape or not xp.all(dict_a[key] == dict_b[key]):
                return False
        elif dict_a[key] != dict_b[key]:
            return False

        return True

    def _eq_object_arrays(self, obj_array_a: np.ndarray, obj_array_b: np.ndarray) -> bool:
        """Compare two object arrays for equality."""
        assert obj_array_a.dtype == object and obj_array_b.dtype == object

        if obj_array_a.shape != obj_array_b.shape:
            return False

        for a, b in zip(obj_array_a.flat, obj_array_b.flat):
            if a is None and b is None:
                continue
            elif sp.issparse(a) and sp.issparse(b):
                if a.shape != b.shape or a.dtype != b.dtype or (a != b).nnz != 0:
                    return False
            elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                if a.shape != b.shape or a.dtype != b.dtype or not np.all(a == b):
                    return False
            elif array_api_compat.is_array_api_obj(a) and array_api_compat.is_array_api_obj(b):
                try:
                    xp = array_api_compat.array_namespace(a, b)
                except (ValueError, TypeError):
                    return False
                if not xp.all(a == b):
                    return False
            elif any(a != b):
                return False

        return True

    def to_matrad(self, context: Union[str, dict] = "mat-file") -> Any:
        """
        Perform matRad compatible serialization.

        Parameters
        ----------
        context : str, optional
            The context in which the datastructure should be serialized,
            by default 'mat-file'.

        Returns
        -------
        Any
            A datastructre compatible with matRad in the given context

        Notes
        -----
        Currently, the only supported context is 'mat-file'. In the future,
        this could be extended to support other contexts, such as direct
        calling via the matlab engine or oct2py.
        """
        self_copy = deepcopy(self)
        if isinstance(context, dict):
            if "matRad" not in context:
                context.update({"matRad": "mat-file"})
        else:
            context = {"matRad": context}

        if context["matRad"] != "mat-file":
            raise ValueError(f"Context {context} not supported")

        # Standard is a model_dump using above alias and context information
        return self_copy.model_dump(by_alias=True, context=context)
