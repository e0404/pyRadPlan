from abc import ABC
from typing import Any, Union
from typing_extensions import Annotated, Self
import warnings
from pydantic import (
    Field,
    field_validator,
    model_validator,
    computed_field,
    StringConstraints,
)

import numpy as np
import SimpleITK as sitk

from pyRadPlan.core import PyRadPlanBaseModel, np2sitk
from pyRadPlan.ct import CT

# Default overlap priorities
DEFAULT_OVERLAPS = {"TARGET": 0, "OAR": 5, "HELPER": 10, "EXTERNAL": 15}


class VOI(PyRadPlanBaseModel, ABC):
    """
    Represents a Volume of Interest (VOI).

    Parameters
    ----------
    name : str
        The name of the VOI.
    ct_image : CT
        The CT image where the VOI is defined.
    mask : np.ndarray or sitk.Image
        Boolean mask (using 0,1) for referencing of voxels
        (Multiple allocations possible for robust scenarios)
    alpha_x : float, optional
        The alpha_x value. Defaults to 0.1.
    beta_x : float, optional
        The beta_x value. Defaults to 0.05.
    overlap_priority : int
        The overlap priority of the VOI. Lowest number is overlapping higher numbers.
    """

    name: str
    ct_image: CT
    mask: sitk.Image
    alpha_x: float = Field(default=0.1)
    beta_x: float = Field(default=0.05)
    voi_type: Annotated[str, StringConstraints(strip_whitespace=True, to_upper=True)]

    overlap_priority: int = Field(
        alias="Priority", default_factory=lambda data: DEFAULT_OVERLAPS[data["voi_type"]]
    )

    # TODO: it would be nicer if this was a list of optimization.Objective, but that would create a
    # circular import. Forward type hinting does not work directly due to pydantic. If someone has
    # a better idea how to solve this, please do so.
    objectives: list[Any] = Field(default=[], description="List of objective function definitions")

    @field_validator("mask", mode="before")
    @classmethod
    def validate_mask_type(cls, v: Any) -> Any:
        """
        Validate the mask type.

        Parameters
        ----------
        v : Any
            The mask value to be validated.

        Returns
        -------
        sitk.Image
            The validated mask.

        Raises
        ------
        ValueError
            If the mask type is not supported.
        """
        if isinstance(v, np.ndarray):
            if v.dtype in ["bool", "int"]:
                v = v.astype("uint8")
            if v.dtype != "uint8":
                raise ValueError(
                    f"{v.dtype} is not supported for index mask. Please use uint8 or boolean mask."
                )

            if v.ndim == 3:
                return sitk.GetImageFromArray(v, False)

            if v.ndim == 4:
                mask = []
                for i in range(v.shape[0]):
                    mask.append(sitk.GetImageFromArray(v[i], False))
                v = sitk.JoinSeries(mask)
                return v

            raise ValueError("Dimensionality not supported!")

        if isinstance(v, sitk.Image):
            if sitk.GetArrayViewFromImage(v).dtype != "uint8":
                raise ValueError(
                    f"""{sitk.GetArrayViewFromImage(v).dtype} is not supported
                      for index mask. Please use uint8."""
                )
            return v

        raise ValueError("mask must be either passed as numpy array or SimpleITK image")

    @model_validator(mode="after")
    def validate_mask(self):
        """
        Check if the given indices are valid for the CT image.

        Raises
        ------
        ValueError
            If the mask is not a sitk.Image.
        ValueError
            If the dimensions of the mask do not match the CT image.
        """
        if not isinstance(self.mask, sitk.Image):
            raise ValueError("Sanity check failed - mask is not a SimpleITK image")

        # check dimensions of sitk image
        dims = self.mask.GetSize()
        if dims != self.ct_image.cube_hu.GetSize():
            raise ValueError(
                f"Mask provided with dimensions {dims}, "
                f"but ct has dimensions {self.ct_image.cube_hu.GetSize()}"
            )

        # set image parameters for mask in accordance to ct image
        self.mask.SetOrigin(self.ct_image.cube_hu.GetOrigin())
        self.mask.SetSpacing(self.ct_image.cube_hu.GetSpacing())
        self.mask.SetDirection(self.ct_image.cube_hu.GetDirection())
        return self

    @computed_field
    @property
    def indices(self) -> np.ndarray:
        """
        Return the indices of the voxels in the mask using Fortran/SITK convention.

        Returns
        -------
        np.ndarray
            The indices of the voxels.
        """
        return np2sitk.sitk_mask_to_linear_indices(self.mask, order="sitk")

    @computed_field
    @property
    def indices_numpy(self) -> np.ndarray:
        """
        Return the indices of the voxels in the mask using C/numpy convention.

        Returns
        -------
        np.ndarray
            The indices of the voxels.
        """
        return np2sitk.sitk_mask_to_linear_indices(self.mask, order="numpy")

    @computed_field
    @property
    def _numpy_mask(self) -> np.ndarray:
        """
        Returns the mask as a numpy array.

        Returns
        -------
        np.ndarray
            The mask as a numpy array.
        """
        return sitk.GetArrayViewFromImage(self.mask)

    @computed_field
    @property
    def num_of_scenarios(self) -> int:
        """
        Returns the number of scenarios.

        Returns
        -------
        int
            The number of scenarios.
        """

        if self.mask.GetDimension() == 4:
            return self.mask.GetSize()[3]

        return 1

    def get_indices(self, order="sitk") -> np.ndarray:
        """
        Return the indices of the voxels in the mask.

        Parameters
        ----------
        order : str, optional
            The order of the indices. Defaults to "sitk".

        Returns
        -------
        np.ndarray
            The indices of the voxels.
        """
        if order == "numpy":
            return self.indices_numpy
        if order == "sitk":
            return self.indices
        raise ValueError(f"Unknown order: {order}")

    def scenario_indices(self, order_type="numpy") -> Union[np.ndarray, list[np.ndarray]]:
        """
        Return the flattened indices of the individual scenarios.

        Parameters
        ----------
        order_type : str, optional
            The order type. Defaults to "numpy".

        Returns
        -------
        List[np.ndarray]
            The flattened indices of the individual scenarios.
        """
        if order_type == "numpy":
            _order = "C"
        elif order_type == "sitk":
            _order = "F"
        else:
            raise ValueError(f"Unknown order type: {order_type}")

        arr = sitk.GetArrayViewFromImage(self.mask)
        if len(arr.shape) == 3:
            return np.ravel_multi_index(np.argwhere(arr).T, dims=arr.shape, order=_order)
        if len(arr.shape) == 4:
            return [
                np.ravel_multi_index(np.argwhere(arr[i]).T, dims=arr[i].shape, order=_order)
                for i in range(arr.shape[0])
            ]

        raise ValueError("Sanity check failed - mask has invalid dimensions")

    def masked_ct(self, order_type="numpy") -> Union[sitk.Image, np.ndarray]:
        """
        Return the masked CT image, either as a numpy array or a SimpleITK image.

        Parameters
        ----------
        order_type : str, optional
            The order type. Defaults to "numpy".

        Returns
        -------
        sitk.Image or np.ndarray
            The masked CT image.
        """

        if order_type not in ["numpy", "sitk"]:
            raise ValueError(f"Invalid order type requested: {order_type}")

        if len(self.mask.GetSize()) == 3:
            masked_ct = sitk.Mask(self.ct_image.cube_hu, self.mask)
        elif len(self.mask.GetSize()) == 4:
            masked_ct = [
                sitk.Mask(self.ct_image.cube_hu[:, :, :, i], self.mask[:, :, :, i])
                for i in range(self.mask.GetSize()[-1])
            ]
            masked_ct = sitk.JoinSeries(masked_ct)
        else:
            raise ValueError("Sanity check failed - mask has invalid dimensions")

        if order_type == "numpy":
            return sitk.GetArrayFromImage(masked_ct)
        if order_type == "sitk":
            return masked_ct

        raise ValueError(f"Sanity check failed -- Invalid order type requested: {order_type}")

    @computed_field
    @property
    def scenario_ct_data(self) -> Union[list[np.ndarray], np.ndarray]:
        """
        Returns a list of CT data for the individual scenarios.

        Returns
        -------
        List[np.ndarray]
            The CT data for the individual scenarios.
        """

        mask_np = sitk.GetArrayFromImage(self.mask).astype("bool")
        ct_np = sitk.GetArrayFromImage(self.ct_image.cube_hu)

        if len(self.mask.GetSize()) == 3:
            return ct_np[mask_np]

        if len(self.mask.GetSize()) == 4:
            return [ct_np[i][mask_np[i]] for i in range(mask_np.shape[0])]

        raise ValueError("Sanity Check failed -- Unsupported dimensionality of stored mask")

    def to_matrad(self, context: str = "mat-file") -> Any:
        """
        Create an object that can be interpreted by matRad in the given context.

        Returns
        -------
        Any
            VOI as list to write cell arrays.
        """

        if context != "mat-file":
            raise ValueError(f"Context {context} not supported")

        voi_list = [0]  # We store an ID which will be changed by cst if exported from there
        voi_list.append(self.name)
        voi_list.append(self.voi_type)
        if self.num_of_scenarios == 1:
            index_lists = np.ndarray(shape=(1,), dtype=object)
            mask_array = sitk.GetArrayFromImage(self.mask)
            mask_array = np.swapaxes(mask_array, 1, 2)
            indices = np.argwhere(mask_array.ravel(order="C") > 0) + 1
            index_lists[0] = np.array(indices, dtype=float)

        else:
            index_lists = self.scenario_indices(order_type="numpy")
            for i, index_list in enumerate(index_lists):
                index_lists[i] = index_list.astype(float)

        voi_list.append(index_lists)

        property_dict = {
            "alphaX": self.alpha_x,
            "betaX": self.beta_x,
            "Priority": self.overlap_priority,
        }
        voi_list.append(property_dict)

        # Will not be populated in here but in cst if exported from there
        objective_dict = {}
        voi_list.append([objective_dict])

        return voi_list

    def resample_on_new_ct(self, new_ct: CT) -> Self:
        """
        Resample on new CT image.

        Parameters
        ----------
        new_ct : CT
            The new CT image to resample the VOI on.

        Returns
        -------
        Self
            The resampled VOI.
        """

        if not isinstance(new_ct, CT):
            raise ValueError("new_ct must be a CT object")

        if self.mask.GetDimension() == 3:
            new_mask = sitk.Resample(self.mask, new_ct.cube_hu)
        elif self.mask.GetDimension() == 4:
            new_mask = []
            for i in range(self.mask.GetSize()[-1]):
                new_mask.append(sitk.Resample(self.mask[:, :, :, i], new_ct.cube_hu))
            new_mask = sitk.JoinSeries(new_mask)
        else:
            raise ValueError("Sanity check failed -- mask has invalid dimensions")

        resampled_voi = self.model_copy(update={"mask": new_mask, "ct_image": new_ct})

        if len(resampled_voi.indices) == 0:
            warnings.warn("Resampling created an empty structure")

        return resampled_voi


class OAR(VOI):
    """
    Represents an organ at risk (OAR).

    Attributes
    ----------
    Inherits all attributes from Plan.

    Methods
    -------
    voi_type : str
        Returns the voi_type as 'OAR'.
    """

    voi_type: str = "OAR"

    @field_validator("voi_type", mode="after")
    @classmethod
    def validate_voi_type(cls, v: str) -> str:
        """
        Validate the voi type for an OAR.

        Parameters
        ----------
        v : str
            The voi type to be validated.

        Returns
        -------
        str
            The validated voi type.

        Raises
        ------
        ValueError
            If the voi type is not "OAR".
        """

        if v != "OAR":
            raise ValueError('VOI type for OAR must be "OAR"')
        return v


class Target(VOI):
    """
    Represents a target VOI.

    Attributes
    ----------
    Inherits all attributes from Plan.

    Methods
    -------
    voi_type : str
        Returns the voi_type as 'TARGET'.
    """

    voi_type: str = "TARGET"

    @field_validator("voi_type", mode="after")
    @classmethod
    def validate_voi_type(cls, v: str) -> str:
        """
        Validate the voi type for a Target.

        Parameters
        ----------
        v : str
            The voi type to be validated.

        Returns
        -------
        str
            The validated voi type.

        Raises
        ------
        ValueError
            If the voi type is not "OAR".
        """

        if v != "TARGET":
            raise ValueError('VOI type for a Target must be "TARGET"')
        return v


class HelperVOI(VOI):
    """
    Represents a helper VOI.

    Attributes
    ----------
    Inherits all attributes from Plan.

    Methods
    -------
    voi_type : str
        Returns the voi_type as 'HELPER'.
    """

    voi_type: str = "HELPER"

    @field_validator("voi_type", mode="after")
    @classmethod
    def validate_voi_type(cls, v: str) -> str:
        """
        Validate the voi type for a HelperVOI.

        Parameters
        ----------
        v : str
            The voi type to be validated.

        Returns
        -------
        str
            The validated voi type.

        Raises
        ------
        ValueError
            If the voi type is not "HELPER".
        """
        if v != "HELPER":
            raise ValueError('VOI type for a HelperVOI must be "HELPER"')
        return v


class ExternalVOI(VOI):
    """
    Represents an external contour limiting voxels to be considered for planning (EXTERNAL).

    Attributes
    ----------
    Inherits all attributes from Plan.

    Methods
    -------
    voi_type : str
        Returns the voi_type as 'EXTERNAL'.
    """

    voi_type: str = "EXTERNAL"

    @field_validator("voi_type", mode="after")
    @classmethod
    def validate_voi_type(cls, v: str) -> str:
        """
        Validate the voi type for an EXTERNAL contour.

        Parameters
        ----------
        v : str
            The voi type to be validated.

        Returns
        -------
        str
            The validated voi type.

        Raises
        ------
        ValueError
            If the voi type is not "EXTERNAL".
        """

        if v != "EXTERNAL":
            raise ValueError('VOI type for EXTERNAL must be "EXTERNAL"')
        return v


__VOITYPES__ = {"OAR": OAR, "TARGET": Target, "HELPER": HelperVOI, "EXTERNAL": ExternalVOI}


def create_voi(data: Union[dict[str, Any], VOI, None] = None, **kwargs) -> VOI:
    """
    Create a VOI object.

    Parameters
    ----------
    data : Union[dict[str, Any], VOI, None]
        Dictionary containing the data to create the VOI object.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    VOI
        A VOI object.
    """

    if data:
        # If data is already a VOI object, return it directly
        if isinstance(data, VOI):
            return data

        # obtain voi type if we have a dict including camelCase check
        voi_type = data.get("voi_type", data.get("voiType", None))

        if voi_type in __VOITYPES__:
            return __VOITYPES__[voi_type].model_validate(data)

        raise ValueError(f"Invalid VOI type: {voi_type}")

    voi_type = kwargs.get("voi_type", "")

    if voi_type in __VOITYPES__:
        return __VOITYPES__[voi_type](**kwargs)

    raise ValueError(f"Invalid VOI type: {voi_type}")


def validate_voi(data: Union[dict[str, Any], VOI, None] = None, **kwargs) -> VOI:
    """
    Validate and create a VOI object.

    Synonym to create_voi but should be used in validation context.

    Parameters
    ----------
    voi : Union[dict[str, Any], VOI, None], optional
        Dictionary containing the data to create the VOI object, by default None.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    VOI
        A validated VOI object.
    """
    return create_voi(data, **kwargs)
