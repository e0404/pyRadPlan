import csv
import logging
import os
from typing import Literal, Union, Optional, Any
from typing_extensions import Self
from pydantic import (
    Field,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from numpydantic import NDArray, Shape

import pint
import numpy as np
import SimpleITK as sitk

from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.core._units import ureg
from pyRadPlan.cst import StructureSet
from pyRadPlan.result import QuantityDistribution

logger = logging.getLogger(__name__)

DVHKind = Literal["cumulative", "differential"]


def _unit_symbol(unit: pint.Unit) -> str:
    from pyRadPlan.analysis._qi import format_unit_symbol  # noqa: PLC0415 - circular import

    return format_unit_symbol(unit)


def _write_csv(path: Union[str, os.PathLike], rows: list[list[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def _number(value: float) -> str:
    return repr(float(value))


def dose_at_volume(
    bins: NDArray, cum_volume: NDArray, y: Union[float, NDArray]
) -> Union[float, NDArray]:
    """Return the dose (or other quantity) value covering *y* percent of the volume.

    Parameters
    ----------
    bins : NDArray
        Left bin edges of a cumulative DVH.
    cum_volume : NDArray
        Cumulative volume percentages at *bins*, decreasing with dose.
    y : float or NDArray
        Volume percentage(s) between 0 and 100.

    Returns
    -------
    float or NDArray
        The largest bin value still covering at least *y* percent of the volume.
    """
    y = np.asarray(y, dtype=np.float64)
    if np.any(y < 0) or np.any(y > 100):
        raise ValueError("Volume percentage must be between 0 and 100.")

    # Dy is the largest quantity value still covering at least y percent of the
    # volume, matching the "at least" reading of the cumulative DVH documented on
    # :attr:`DVH.cumulative`. cum_volume decreases with dose, so the search runs
    # over the reversed (increasing) array and the index is mapped back.
    #
    # This is deliberately not an interpolation across the cumulative curve: a
    # uniformly irradiated region is a plateau in the DVH, and interpolating over
    # it reports the dose at the wrong end of that plateau.
    reversed_volume = np.asarray(cum_volume)[::-1]
    last = reversed_volume.size - 1
    index = last - np.searchsorted(reversed_volume, y, side="left")
    index = np.clip(index, 0, last)
    values = np.asarray(bins)[index]
    return float(values) if values.ndim == 0 else values


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
    total_volume: Optional[float] = Field(
        default=None, description="Absolute volume of the structure in cm3, if known"
    )

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
            return ureg.Unit(v)
        except pint.UndefinedUnitError:
            raise ValueError(f"Invalid unit: {v}")

    @field_serializer("unit")
    def _serialize_unit(self, v: pint.Unit) -> str:
        return str(v)

    @field_serializer("has_regular_bins")
    def _serialize_has_regular_bins(self, v: Any) -> bool:
        return bool(v)

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
        bin center for the differential DVH. This is because the differential
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
        return dose_at_volume(self.bins, self.cum_volume, y)

    @classmethod
    def compute(
        cls,
        quantity: Union[QuantityDistribution, sitk.Image, NDArray],
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
        quantity : Union[QuantityDistribution, sitk.Image, NDArray]
            The dose-like quantity to update the DVH with, e.g. ``result.physical_dose``.
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

        Notes
        -----
        For a SimpleITK quantity, `total_volume` (cm3) is the number of voxels in the
        mask times the voxel volume, unless passed explicitly.
        """

        voxel_volume = None
        if isinstance(quantity, QuantityDistribution):
            quantity = quantity.values
        if isinstance(quantity, sitk.Image):
            voxel_volume = float(np.prod(quantity.GetSpacing())) / 1000.0
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
        if voxel_volume is not None:
            kwargs.setdefault("total_volume", len(q_array) * voxel_volume)

        # np.histogram takes the bin dtype from the quantity, so a float32 input --
        # what a DICOM-imported dose is -- would produce float32 edges, which the
        # model declares as float64 and rejects. Widening float32 is exact, so this
        # only changes the storage dtype, never a bin edge.
        return DVH(
            diff_volume=np.asarray(hist, dtype=np.float64),
            bin_edges=np.asarray(edges, dtype=np.float64),
            **kwargs,
        )

    def _curve(self, kind: DVHKind) -> tuple[Any, Any]:
        if kind == "cumulative":
            return self.bins, self.cum_volume
        if kind == "differential":
            return self.bin_centers, self.diff_volume
        raise ValueError("kind must be 'cumulative' or 'differential'.")

    def to_rows(
        self, kind: DVHKind = "cumulative", absolute_volume: bool = False
    ) -> list[list[str]]:
        """Return the DVH as a table of strings, header first.

        Parameters
        ----------
        kind : {"cumulative", "differential"}, optional
            Cumulative DVH at the left bin edges (default) or differential DVH at the bin
            centers.
        absolute_volume : bool, optional
            Add a ``volume [cm3]`` column. Requires `total_volume`.

        Returns
        -------
        list of list of str
            Columns ``dose [<unit>]``, ``volume [%]`` and optionally ``volume [cm3]``. Numbers
            are written with full precision.
        """
        x, y = self._curve(kind)
        header = [f"dose [{_unit_symbol(self.unit)}]", "volume [%]"]
        columns = [x, y]
        if absolute_volume:
            if self.total_volume is None:
                raise ValueError(f"DVH '{self.name}' does not know its absolute volume.")
            header.append("volume [cm3]")
            columns.append(y / 100.0 * self.total_volume)
        rows = [header]
        rows.extend([_number(v) for v in values] for values in zip(*columns))
        return rows

    def to_csv(
        self,
        path: Union[str, os.PathLike],
        kind: DVHKind = "cumulative",
        absolute_volume: bool = False,
    ) -> None:
        """Write the DVH as CSV (see :meth:`to_rows` for the columns).

        Parameters
        ----------
        path : str or os.PathLike
            Output file.
        kind : {"cumulative", "differential"}, optional
            DVH type to write.
        absolute_volume : bool, optional
            Add a ``volume [cm3]`` column.
        """
        _write_csv(path, self.to_rows(kind, absolute_volume))

    def to_json(self, path: Union[str, os.PathLike], indent: Optional[int] = None) -> None:
        """Write the DVH as lossless JSON (see ``model_dump_json``).

        Parameters
        ----------
        path : str or os.PathLike
            Output file.
        indent : int, optional
            Indentation of the JSON.
        """
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.model_dump_json(indent=indent))

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
            import matplotlib.pyplot as plt  # noqa: PLC0415 - optional at runtime

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
        dose: Union[QuantityDistribution, NDArray, sitk.Image],
    ) -> Self:
        """
        Create DVH collection from a structure set.

        Parameters
        ----------
        cst : StructureSet
            The structure set containing the VOIs (Volumes of Interest).
        dose : Union[QuantityDistribution, NDArray, sitk.Image]
            The dose-like quantity to compute the DVH from, e.g. ``result.physical_dose``.

        Returns
        -------
            DVHCollection: Collection of DVH curves for all structures
        """
        dvhs = [DVH.compute(quantity=dose, mask=voi.mask, name=voi.name) for voi in cst.vois]
        return cls(dvhs=dvhs)

    def to_rows(
        self,
        kind: DVHKind = "cumulative",
        absolute_volume: bool = False,
        layout: Literal["auto", "wide", "long"] = "auto",
    ) -> list[list[str]]:
        """Return the DVHs as a table of strings, header first.

        Parameters
        ----------
        kind : {"cumulative", "differential"}, optional
            Cumulative DVH at the left bin edges (default) or differential DVH at the bin
            centers.
        absolute_volume : bool, optional
            Also write the volume in cm3 (requires `DVH.total_volume` for every DVH).
        layout : {"auto", "wide", "long"}, optional
            ``"wide"``: one ``dose [<unit>]`` column followed by a ``<name> [%]`` column per
            structure (and a ``<name> [cm3]`` column each with ``absolute_volume``); needs
            identical dose points, units and unique names. ``"long"``: columns ``structure``,
            ``dose [<unit>]``, ``volume [%]`` (and ``volume [cm3]``), one row per DVH point.
            ``"auto"`` uses wide when possible and long otherwise.

        Returns
        -------
        list of list of str
            The table; empty for an empty collection.
        """
        if layout not in ("auto", "wide", "long"):
            raise ValueError("layout must be 'auto', 'wide' or 'long'.")
        if not self.dvhs:
            return []
        if absolute_volume and any(dvh.total_volume is None for dvh in self.dvhs):
            raise ValueError("Not every DVH knows its absolute volume.")

        curves = [dvh._curve(kind) for dvh in self.dvhs]
        shared = (
            len({dvh.name for dvh in self.dvhs}) == len(self.dvhs)
            and len({dvh.unit for dvh in self.dvhs}) == 1
            and all(
                len(x) == len(curves[0][0]) and np.array_equal(x, curves[0][0]) for x, _ in curves
            )
        )
        if layout == "wide" and not shared:
            raise ValueError(
                "Wide layout needs identical dose points, units and unique names; use 'long'."
            )

        if layout == "long" or not shared:
            unit = self.dvhs[0].unit
            rows = [["structure", *self.dvhs[0].to_rows(kind, absolute_volume)[0]]]
            for dvh, (x, y) in zip(self.dvhs, curves):
                try:
                    scale = ureg.Quantity(1.0, dvh.unit).to(unit).magnitude
                except pint.DimensionalityError as err:
                    raise ValueError(
                        f"DVH '{dvh.name}' is in {dvh.unit:~}, incompatible units with the "
                        f"{unit:~} of the first DVH; export the collections separately."
                    ) from err
                columns = [x * scale, y]
                if absolute_volume:
                    columns.append(y / 100.0 * dvh.total_volume)
                rows.extend([dvh.name, *(_number(v) for v in values)] for values in zip(*columns))
            return rows

        header = [f"dose [{_unit_symbol(self.dvhs[0].unit)}]"]
        columns = [curves[0][0]]
        for dvh, (_, y) in zip(self.dvhs, curves):
            header.append(f"{dvh.name} [%]")
            columns.append(y)
            if absolute_volume:
                header.append(f"{dvh.name} [cm3]")
                columns.append(y / 100.0 * dvh.total_volume)
        rows = [header]
        rows.extend([_number(v) for v in values] for values in zip(*columns))
        return rows

    def to_csv(
        self,
        path: Union[str, os.PathLike],
        kind: DVHKind = "cumulative",
        absolute_volume: bool = False,
        layout: Literal["auto", "wide", "long"] = "auto",
    ) -> None:
        """Write the DVHs as CSV (see :meth:`to_rows` for the layouts).

        Parameters
        ----------
        path : str or os.PathLike
            Output file.
        kind : {"cumulative", "differential"}, optional
            DVH type to write.
        absolute_volume : bool, optional
            Also write the volume in cm3.
        layout : {"auto", "wide", "long"}, optional
            Table layout.
        """
        _write_csv(path, self.to_rows(kind, absolute_volume, layout))

    def to_json(self, path: Union[str, os.PathLike], indent: Optional[int] = None) -> None:
        """Write the collection as lossless JSON (see ``model_dump_json``).

        Parameters
        ----------
        path : str or os.PathLike
            Output file.
        indent : int, optional
            Indentation of the JSON.
        """
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.model_dump_json(indent=indent))

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
            import matplotlib.pyplot as plt  # noqa: PLC0415 - optional at runtime

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
