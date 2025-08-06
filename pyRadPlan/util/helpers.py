"""Test Helper functions for pyRadPlan."""

from typing import Union, get_type_hints, get_origin, get_args
import warnings
import numpy as np
from pydantic import BaseModel
from pydantic.fields import FieldInfo, ComputedFieldInfo
from pyRadPlan.core import PyRadPlanBaseModel
import scipy.sparse as sp


def dl2ld(dict_of_lists: dict[str, list], type_check: bool = True) -> list[dict]:
    """Convert a dictionary of lists to a list of dictionaries.

    Parameters
    ----------
    dict_of_lists : dict
        The dictionary of lists to convert.
    type_check : bool, optional
        Whether to perform type checking, by default True

    Returns
    -------
    list[dict]
        A list of dictionaries.
    """

    if type_check:
        # Check if dict_of_lists is a dictionary
        if not isinstance(dict_of_lists, dict):
            raise TypeError("The input must be a dictionary.")
        # Check if all values in the dictionary are lists
        if not all(isinstance(value, list) for value in dict_of_lists.values()):
            raise TypeError("All values in the dictionary must be lists.")
        # CHeck if all lists have the same length
        if len(set(map(len, dict_of_lists.values()))) > 1:
            raise TypeError("All lists in the dictionary must have the same length.")

    # Empty input returns empty array
    if dict_of_lists == {}:
        return []

    # Convert the dictionary of lists to a list of dictionaries
    list_of_dicts = [dict(zip(dict_of_lists, t)) for t in zip(*dict_of_lists.values())]

    return list_of_dicts


def ld2dl(list_of_dicts: list[dict], type_check: bool = True) -> dict[str, list]:
    """Convert a list of dictionaries to a dictionary of lists.

    Parameters
    ----------
    list_of_dicts : list
        The list of dictionaries to convert.
    type_check : bool, optional
        Whether to perform type checking, by default True

    Returns
    -------
    dict
        A dictionary of lists.
    """

    if type_check:
        # Check if list_of_dicts is a list
        if not isinstance(list_of_dicts, list):
            raise TypeError("The input must be a list.")
        # Check if all elements in the list are dictionaries
        if not all(isinstance(element, dict) for element in list_of_dicts):
            raise TypeError("All elements in the list must be dictionaries.")

    # Empty input returns empty array
    if len(list_of_dicts) == 0:
        return {}

    # Convert the list of dictionaries to a dictionary of lists
    dict_of_lists = {key: [element[key] for element in list_of_dicts] for key in list_of_dicts[0]}

    return dict_of_lists


def models2recarray(
    models: list[BaseModel],
    serialization_context: Union[dict, str] = None,
    override_types: dict = None,
    by_alias: bool = False,
) -> np.recarray:
    """Convert a list of PyRadPlanBaseModel instances to a numpy recarray.

    Parameters
    ----------
    models : list
        The list of PyRadPlanBaseModel instances to convert.
    serialization_context : str, optional
        The context in which the datastructure should be serialized, by default None.
    override_types : dict, optional
        A dictionary of types to override the automatically obtained types, by default None.
        If the context contains "matRad" and the models are of type PyRadPlanBaseModel, the models
        will be converted to matRad compatible structures.
    by_alias : bool, optional
        Whether to use the alias names for serialization, by default False.

    Returns
    -------
    np.recarray
        A numpy recarray.
    """

    # Check if models is a list
    if not isinstance(models, list):
        raise TypeError("The input must be a list.")

    # Check if models is not empty
    if not models:
        raise ValueError("The input list must not be empty.")

    first_model_type = type(models[0])

    # Check if the type is a subclass of a BaseModel
    if not issubclass(first_model_type, BaseModel):
        raise TypeError("All elements in the list must be subclasses of BaseModel.")

    # Check if all modelss in the list are of the same type
    if not all(isinstance(model, first_model_type) for model in models):
        raise TypeError("All models in the list must be of the same type.")

    if "matRad" in serialization_context and isinstance(models[0], PyRadPlanBaseModel):
        models_dump = [model.to_matrad(context=serialization_context) for model in models]
        by_alias = True
    else:
        models_dump = [
            model.model_dump(by_alias=by_alias, context=serialization_context) for model in models
        ]

    model_fields = models[0].model_fields | models[0].model_computed_fields
    model_field_types = get_type_hints(models[0].__class__)
    for cfield in models[0].model_computed_fields:
        model_field_types.update({cfield: model_fields[cfield].return_type})

    # Debug
    if "machine" in model_fields:
        pass

    # Type management
    # Override manual types
    if override_types:
        model_field_types.update(override_types)

        # Remove model_field and model_field_types where the override_types is None
        model_fields = {
            field: model_fields[field]
            for field in model_fields
            if model_field_types[field] is not None
        }
        model_field_types = {
            field: model_field_types[field]
            for field in model_field_types
            if model_field_types[field] is not None
        }

    # convert dtypes
    for field in model_field_types:
        # str -> 'U'
        if model_field_types[field] is str:
            model_field_types[field] = object

        # list management
        if get_origin(model_field_types[field]) is list:
            model_field_types[field] = np.ndarray

        # manage optional types
        if get_origin(model_field_types[field]) is Union:
            t_args = get_args(model_field_types[field])
            if type(None) in t_args and len(t_args) == 2:
                model_field_types[field] = t_args[0]
            else:
                warnings.warn(
                    f"Field {field} has a Union type that is not supported. Will use object."
                )
                model_field_types[field] = object

    aliases = {}
    for field in model_fields:
        if by_alias:
            if isinstance(model_fields[field], FieldInfo):
                aliases[field] = model_fields[field].serialization_alias
            elif isinstance(model_fields[field], ComputedFieldInfo):
                aliases[field] = model_fields[field].alias
            else:  # Sanity Check
                raise TypeError(f"Field {field} is not a FieldInfo or ComputedFieldInfo object.")
        else:
            aliases[field] = field

    # Create a dtype for the structured armodel
    models_dtype = np.dtype([(aliases[field], model_field_types[field]) for field in model_fields])

    models_recarray = np.recarray((len(models),), dtype=models_dtype)

    for i, model_dict in enumerate(models_dump):
        for field in model_fields:
            fname = aliases[field]
            value = model_dict[fname]

            if model_field_types[field] == np.str_:
                value = np.str_(value)
            elif model_field_types[field] == np.ndarray:
                value = np.asarray(value)
            models_recarray[fname][i] = value

    return models_recarray


def swap_orientation_sparse_matrix(
    sparse_matrix: sp.csc_matrix, original_shape, axes
) -> sp.csc_matrix:
    """
    Swap the specified axes of a sparse matrix.

    Parameters
    ----------
    sparse_matrix : sp.csc_matrix
        The sparse matrix to swap axes.
    original_shape : tuple
        The original shape of the matrix.
    axes : tuple
        The axes to swap.

    Returns
    -------
    sp.csc_matrix
        The sparse matrix with swapped axes.
    """
    row_indices, _ = sparse_matrix.nonzero()

    if axes in ((0, 1), (1, 0)):
        j, i, k = np.unravel_index(row_indices, original_shape)
        new_shape = (original_shape[1], original_shape[0], original_shape[2])
    elif axes in ((0, 2), (2, 0)):
        k, j, i = np.unravel_index(row_indices, original_shape)
        new_shape = (original_shape[2], original_shape[1], original_shape[0])
    elif axes in ((1, 2), (2, 1)):
        i, k, j = np.unravel_index(row_indices, original_shape)
        new_shape = (original_shape[0], original_shape[2], original_shape[1])
    else:
        raise ValueError("Invalid axes for swapping")

    new_indices = np.ravel_multi_index((i, j, k), new_shape)
    num_rows = np.prod(original_shape)
    permutation = sp.csc_matrix(
        (np.ones_like(row_indices), (new_indices, row_indices)),
        shape=(num_rows, num_rows),
        dtype=bool,
    )

    reordered_sparse_matrix = permutation @ sparse_matrix
    return reordered_sparse_matrix
