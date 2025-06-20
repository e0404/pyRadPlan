import logging
from typing import Union, Optional, Any
from typing_extensions import Self
from pydantic import (
    Field,
    computed_field,
    field_validator,
    model_validator,
)
from numpydantic import NDArray, Shape

import pint
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.cst import StructureSet

logger = logging.getLogger(__name__)

ureg = pint.UnitRegistry()


class DVH(PyRadPlanBaseModel):
    """
    Dose Volume Histogram (DVH) class.

    Note
    ----
    The DVH class can be used for other quantities as well.
    The bins correspond to the left edge of the bin, thus we have as many bin points as dose points.

    """

    diff_volume: NDArray[Shape["*"], np.float64] = Field(
        alias="volumePoints", description="Volume percentages for each point"
    )

    unit: pint.Unit = Field(default=ureg.gray, description="Unit of the DVH bins")
    name: str = Field(default="DVH", description="Name of the DVH")

    bin_edges: NDArray[Shape["*"], np.float64] = Field(
        alias="doseGrid", description="Left bin edges for the DVH"
    )

    @field_validator("unit", mode="before")
    @classmethod
    def validate_unit(cls, v: Any) -> pint.Unit:
        """
        Validate the unit of the DVH bins.

        Parameters
        ----------
        v : Any
            The unit to validate.

        Returns
        -------
            pint.Unit: Validated unit of the DVH bins.
        """

        if isinstance(v, pint.Unit):
            return v

        try:
            return ureg(v)
        except pint.UndefinedUnitError:
            raise ValueError(f"Invalid unit: {v}")

    @model_validator(mode="after")
    def check_shapes(self) -> Self:
        """
        Validate the shapes of the DVH data.

        Returns
        -------
            DVH: The DVH object itself if validation passes.
        """
        if (
            self.diff_volume.ndim != self.bin_edges.ndim
            or self.diff_volume.ndim != 1
            or self.bin_edges.size != self.diff_volume.size + 1
        ):
            raise ValueError("The shape of diff_volume must be consistent with shape of bins.")
        return self

    @computed_field
    @property
    def cum_volume(self) -> NDArray[Shape["*"], np.float64]:
        """
        Get the cumulative DVH Volume.

        Returns
        -------
            NDArray: Cumulative DVH volume as a numpy array
        """
        return np.cumsum(self.diff_volume[::-1])[::-1]

    @computed_field
    @property
    def cumulative(self) -> NDArray[Shape["2,*"], np.float64]:
        """
        Get the cumulative DVH.

        Gives a 2xnum_points array with the first row being the dose / quantity
        and the second row being the cumulative volume percentages.

        Returns
        -------
            NDArray: Cumulative DVH

        Notes
        -----
        The x coordinate (dose/quantity) corresponds to the histograms left
        bin edge, as this corresponds to the meaning: Volume covered by
        "at least" the respective dose/quantity value.
        """
        return np.vstack((self.bins, self.cum_volume))

    @computed_field
    @property
    def differential(self) -> NDArray[Shape["2,*"], np.float64]:
        """
        Get the differential DVH.

        Gives a 2xnum_points array with the first row being the dose
        and the second row being the differential volume percentages.

        Returns
        -------
            NDArray: Differential DVH

        Notes
        -----
        The x ccoordinate (dose/quantity) corresponds to the histograms
        bin center for the differntial DVH. This is because the differential
        DVH captures the relative volume within a bin, thus the center
        represents the dose/quantity value for that bin best.
        Be aware of this difference to the cumulative DVH.

        """
        return np.vstack((self.bin_centers, self.diff_volume))

    @computed_field
    @property
    def num_points(self) -> int:
        """
        Get the number of points in the DVH.

        Returns
        -------
            int: Number of points in the DVH
        """
        return len(self.diff_volume)

    @computed_field
    @property
    def bins(self) -> NDArray[Shape["*"], np.float64]:
        """
        Get the DVH bin left edges.

        This corresponds to the dose / quantity thresholds for the cumulative volume percentages.

        Returns
        -------
            NDArray: DVH bins as a numpy array
        """
        return self.bin_edges[:-1]

    @computed_field
    @property
    def has_regular_bins(self) -> bool:
        """
        Check if the DVH has regular bins.

        Returns
        -------
            bool: True if the DVH has regular bins, False otherwise
        """
        return np.all(np.diff(self.bin_edges) == self.bin_edges[1] - self.bin_edges[0])

    @computed_field
    @property
    def bin_centers(self) -> NDArray[Shape["*"], np.float64]:
        """
        Get the centers of the DVH bins.

        Returns
        -------
            NDArray: Centers of the DVH bins as a numpy array
        """
        return (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

    def get_vx(self, x: Union[float, NDArray]) -> float:
        """
        Get the volume percentage for a given dose (or other quantity) value.

        Parameters
        ----------
        x : float
            The dose/quantity value to get the volume percentage for.

        Returns
        -------
            float: Volume percentage corresponding to the given dose/quantity value.
        """
        if x < self.bins[0] or x > self.bins[-1]:
            raise ValueError("Dose value is out of bounds of the DVH bins.")

        # Interpolate to find the volume percentage
        return np.interp(x, self.bins, self.cum_volume)

    def get_dy(self, y: Union[float, NDArray]) -> float:
        """
        Get the dose (or other quantity) value for a given volume percentage.

        Parameters
        ----------
        y : float
            The volume percentage to get the dose/quantity value for.

        Returns
        -------
            float: Dose/quantity value corresponding to the given volume percentage.
        """
        if y < 0 or y > 100:
            raise ValueError("Volume percentage must be between 0 and 100.")

        # Interpolate to find the dose value
        return np.interp(y / 100, self.cum_volume, self.bins)

    @classmethod
    def compute(
        cls,
        quantity: Union[sitk.Image, NDArray],
        mask: Optional[Union[sitk.Image, NDArray]] = None,
        num_points: int = 1000,
        max_value: Optional[float] = None,
        min_value: float = 0.0,
        **kwargs,
    ) -> Self:
        """
        Create a DVH from a dose_like quantity and mask.

        Parameters
        ----------
        quantity : Union[sitk.Image, NDArray]
            The dose-like quantity to update the DVH with. Can be a SimpleITK image or a numpy array.
        mask : Optional[Union[sitk.Image, NDArray]], optional
            The mask to apply to the dose quantity. If None, the whole image / array will be used.
            If provided, it must either have the same shape as the quantity.
            Defaults to None.
        num_points : int, optional
            The number of points in the DVH. Defaults to 100.
        max_value : Optional[float], optional
            The maximum value for the DVH bins. If None, it will be set to 105% of the maximum value in the quantity.
        min_value : float, optional
            The minimum value for the DVH bins. Defaults to 0.0.
        **kwargs: Additional arguments passed to the DVH Model.

        Returns
        -------
            DVH: DVH Object
        """

        # manage quantity
        if isinstance(quantity, sitk.Image):
            # Convert SimpleITK image to numpy array
            quantity = sitk.GetArrayFromImage(quantity)

        # First check if mask is provided and is valid
        if mask is not None:
            if isinstance(mask, sitk.Image):
                # Convert SimpleITK mask to numpy array
                mask = sitk.GetArrayViewFromImage(mask)

            # Mask has either same dimensionality as quantity, or is a linear mask
            if mask.shape != quantity.shape:
                raise ValueError("Mask must have the same shape as the quantity.")

            q_array = quantity[mask.astype(np.bool)].flat
        else:
            q_array = quantity.flat

        if max_value is None:
            max_value = np.max(q_array) * 1.05

        # Use bin edges that include the last point
        hist, edges = np.histogram(
            q_array, bins=num_points, range=(min_value, max_value), density=False
        )
        # Remove the last bin
        hist = hist / len(q_array) * 100

        return DVH(diff_volume=hist, bin_edges=edges, **kwargs)

    def plot(self, ax=None, line_width=2, plot_legend=True, **kwargs):
        """Plot the DVH curve.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure and axes. Defaults to None.
        line_width : float, optional
            Width of the plotted lines. Defaults to 2.
        plot_legend : bool, optional
            Whether to show the legend. Defaults to True.
        **kwargs:
            Additional arguments passed to matplotlib's plot function.

        Returns
        -------
            matplotlib.axes.Axes: The axes containing the plot
        """
        if ax is None:
            _, ax = plt.subplots()

        # Plot with or without legend
        if plot_legend:
            ax.plot(
                self.bins,
                self.cum_volume,
                linewidth=line_width,
                label=self.name,
                **kwargs,
            )
        else:
            ax.plot(self.bins, self.cum_volume, linewidth=line_width, **kwargs)

        # Set labels and legend
        ax.set_xlabel("Dose [Gy]")
        ax.set_ylabel("Volume [%]")
        if plot_legend:
            ax.legend()

        # Set axis limits with 5% padding
        ax.set_xlim(0, np.max(self.bins) * 1.05)
        ax.set_ylim(0, np.max(self.cumulative) * 1.05)

        return ax


class DVHCollection(PyRadPlanBaseModel):
    """Collection of DVH curves for multiple structures."""

    dvhs: list[DVH] = Field(description="List of DVH curves")

    @classmethod
    def from_structure_set(
        cls,
        cst: StructureSet,
        dose: Union[NDArray, sitk.Image],
    ) -> Self:
        """
        Create DVH collection from a structure set.

        Parameters
        ----------
        cst : StructureSet
            The structure set containing the VOIs (Volumes of Interest).
        dose : Union[NDArray, sitk.Image]
            The dose-like quantity to compute the DVH from. Can be a SimpleITK image or a numpy array.

        Returns
        -------
            DVHCollection: Collection of DVH curves for all structures
        """
        dvhs = [DVH.compute(quantity=dose, mask=voi.mask, name=voi.name) for voi in cst.vois]
        return cls(dvhs=dvhs)

    def plot(
        self,
        structures: Union[list[str], None] = None,
        ax=None,
        line_width=2,
        plot_legend=True,
        **kwargs,
    ):
        """Plot DVH curves for selected or all structures.

        Parameters
        ----------
        structures : Union[list[str], None], optional
            List of structure names to plot. If None, all structures are plotted. Defaults to None.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure and axes. Defaults to None
        line_width : float, optional
            Width of the plotted lines. Defaults to 2.
        plot_legend : bool, optional
            Whether to show the legend. Defaults to True.
        **kwargs:
            Additional arguments passed to matplotlib's plot function.


        Returns
        -------
            matplotlib.axes.Axes: The axes containing the plot
        """
        if ax is None:
            _, ax = plt.subplots()

        # Filter DVHs based on structure names if provided
        dvhs_to_plot = self.dvhs
        if structures is not None:
            dvhs_to_plot = [dvh for dvh in self.dvhs if dvh.name in structures]
            if not dvhs_to_plot:
                raise ValueError("None of the specified structures found in DVH collection")

        # Plot each DVH
        for dvh in dvhs_to_plot:
            dvh.plot(ax=ax, line_width=line_width, plot_legend=plot_legend, **kwargs)

        return ax
