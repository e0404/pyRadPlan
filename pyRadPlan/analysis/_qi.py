"""Quality indicators (QI) for dose distributions on structure sets."""

from typing import Annotated, Any, Callable, ClassVar, Iterator, Literal, Optional, Union
from typing_extensions import Self

import numpy as np
import pint
import SimpleITK as sitk
import matplotlib.pyplot as plt
from numpydantic import NDArray
from pydantic import Field, field_serializer, field_validator

from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.cst import StructureSet
from pyRadPlan.analysis._dvh import ureg

DoseLike = Union[sitk.Image, NDArray]
MaskLike = Union[sitk.Image, NDArray]


def _to_array(x: DoseLike) -> np.ndarray:
    return sitk.GetArrayFromImage(x) if isinstance(x, sitk.Image) else np.asarray(x)


def _validate_single_scenario(arr: np.ndarray, name: str) -> None:
    """Raise when a quantity or mask contains multiple robust scenarios."""
    if arr.ndim == 4:
        raise ValueError(
            f"{name} appears to contain multiple scenarios. "
            "Select one scenario before computing QIs."
        )


def _validate_sitk_compatibility(quantity: DoseLike, mask: Optional[MaskLike]) -> None:
    """Validate SimpleITK dimensions and spatial metadata before masking."""
    if isinstance(quantity, sitk.Image) and quantity.GetDimension() != 3:
        raise ValueError("QI computation supports one 3D SimpleITK quantity at a time.")

    if isinstance(mask, sitk.Image) and mask.GetDimension() != 3:
        raise ValueError("QI computation supports one 3D SimpleITK mask at a time.")

    if not isinstance(quantity, sitk.Image) or not isinstance(mask, sitk.Image):
        return

    if quantity.GetSize() != mask.GetSize():
        raise ValueError(
            f"Mask size {mask.GetSize()} does not match quantity size {quantity.GetSize()}."
        )

    metadata_matches = (
        np.allclose(quantity.GetOrigin(), mask.GetOrigin())
        and np.allclose(quantity.GetSpacing(), mask.GetSpacing())
        and np.allclose(quantity.GetDirection(), mask.GetDirection())
    )
    if not metadata_matches:
        raise ValueError("Mask image geometry does not match quantity image geometry.")


def _apply_mask(arr: np.ndarray, mask: Optional[MaskLike]) -> np.ndarray:
    """Return a 1-D array of values inside the mask (or all values if mask is None)."""
    _validate_single_scenario(arr, "Quantity")
    if mask is None:
        return arr.ravel()
    mask_arr = _to_array(mask).astype(bool)
    _validate_single_scenario(mask_arr, "Mask")
    if mask_arr.shape != arr.shape:
        raise ValueError(f"Mask shape {mask_arr.shape} does not match quantity shape {arr.shape}.")
    return arr[mask_arr]


def _extract_voxels(quantity: DoseLike, mask: Optional[MaskLike]) -> np.ndarray:
    """Return a 1-D array of voxel values inside the mask (or all voxels if mask is None)."""
    _validate_sitk_compatibility(quantity, mask)
    return _apply_mask(_to_array(quantity), mask)


def _coerce_unit(v: Any) -> pint.Unit:
    """Coerce a ``pint.Unit`` or unit string into a ``pint.Unit`` on the shared registry."""
    if isinstance(v, pint.Unit):
        return v
    if isinstance(v, str):
        try:
            return ureg.Unit(v)
        except pint.UndefinedUnitError as e:
            raise ValueError(f"Invalid unit: {v}") from e
    raise ValueError(f"Unsupported unit type: {type(v).__name__}")


def _validate_ref_vol(ref_vol: Any) -> float:
    """Validate a D_x reference volume percentage."""
    try:
        ref_vol_float = float(ref_vol)
    except (TypeError, ValueError) as e:
        raise ValueError("Reference volume must be finite and between 0 and 100 percent.") from e
    if not np.isfinite(ref_vol_float) or ref_vol_float < 0.0 or ref_vol_float > 100.0:
        raise ValueError("Reference volume must be finite and between 0 and 100 percent.")
    return ref_vol_float


def _validate_ref_dose(ref_dose: Any) -> float:
    """Validate a V_x reference dose."""
    try:
        ref_dose_float = float(ref_dose)
    except (TypeError, ValueError) as e:
        raise ValueError("Reference dose must be finite.") from e
    if not np.isfinite(ref_dose_float):
        raise ValueError("Reference dose must be finite.")
    return ref_dose_float


def _threshold_in_quantity_unit(
    ref_dose: float,
    ref_unit: Optional[pint.Unit],
    quantity_unit: Optional[pint.Unit],
) -> tuple[float, pint.Unit, pint.Unit]:
    """Return the threshold magnitude in the unit used by the dose values."""
    ref_dose = _validate_ref_dose(ref_dose)
    quantity_unit = _coerce_unit(ureg.gray if quantity_unit is None else quantity_unit)
    ref_unit = _coerce_unit(quantity_unit if ref_unit is None else ref_unit)
    try:
        threshold = float((ref_dose * ref_unit).to(quantity_unit).magnitude)
    except pint.DimensionalityError as e:
        raise ValueError(
            f"Reference dose unit '{ref_unit}' is not compatible with quantity unit "
            f"'{quantity_unit}'."
        ) from e
    return threshold, ref_unit, quantity_unit


class QI(PyRadPlanBaseModel):
    """A single quality indicator value.

    Subclasses provide a `compute_from(quantity, mask, ...)` constructor that
    reduces a dose-like quantity over a VOI mask. Subclasses also set
    ``metric_prefix`` and override the ``metric`` property when the metric id
    depends on parameters (e.g. DX, VX). Each subclass declares a unique
    ``qi_type`` literal so collections of QIs serialize and reload faithfully.
    """

    metric_prefix: ClassVar[str] = ""

    qi_type: str = Field(default="qi", description="Discriminator tag identifying the QI type")
    value: float = Field(description="Quality indicator value")
    unit: pint.Unit = Field(default=ureg.gray, description="Unit of the value")

    @field_validator("unit", mode="before")
    @classmethod
    def _validate_unit(cls, v: Any) -> pint.Unit:
        return _coerce_unit(v)

    @field_serializer("unit")
    def _serialize_unit(self, v: pint.Unit) -> str:
        return str(v)

    @property
    def metric(self) -> str:
        return self.metric_prefix


class _VoxelReduction(QI):
    """Common scaffolding for QIs computed by reducing voxel values in a VOI."""

    _reducer: ClassVar[Callable[[np.ndarray], float]]

    @classmethod
    def _from_voxels(cls, voxels: np.ndarray, unit: Optional[pint.Unit] = None) -> Self:
        value = float(cls._reducer(voxels)) if voxels.size > 0 else float("nan")
        return cls(value=value, unit=ureg.gray if unit is None else unit)

    @classmethod
    def compute_from(
        cls,
        quantity: DoseLike,
        mask: Optional[MaskLike] = None,
        unit: Optional[pint.Unit] = None,
    ) -> Self:
        return cls._from_voxels(_extract_voxels(quantity, mask), unit=unit)


class Mean(_VoxelReduction):
    """Mean dose over the VOI."""

    metric_prefix: ClassVar[str] = "mean"
    _reducer: ClassVar[Callable[[np.ndarray], float]] = staticmethod(np.mean)

    qi_type: Literal["mean"] = "mean"


class Std(_VoxelReduction):
    """Standard deviation of the dose over the VOI."""

    metric_prefix: ClassVar[str] = "std"
    _reducer: ClassVar[Callable[[np.ndarray], float]] = staticmethod(np.std)

    qi_type: Literal["std"] = "std"


class Max(_VoxelReduction):
    """Maximum dose over the VOI."""

    metric_prefix: ClassVar[str] = "max"
    _reducer: ClassVar[Callable[[np.ndarray], float]] = staticmethod(np.max)

    qi_type: Literal["max"] = "max"


class Min(_VoxelReduction):
    """Minimum dose over the VOI."""

    metric_prefix: ClassVar[str] = "min"
    _reducer: ClassVar[Callable[[np.ndarray], float]] = staticmethod(np.min)

    qi_type: Literal["min"] = "min"


class DX(QI):
    """Minimum dose covering at least the given reference volume percentage (e.g. D50)."""

    metric_prefix: ClassVar[str] = "D"

    qi_type: Literal["dx"] = "dx"
    ref_vol: float = Field(ge=0.0, le=100.0, description="Reference volume in % (0-100)")

    @field_validator("ref_vol", mode="before")
    @classmethod
    def _validate_ref_vol(cls, v: Any) -> float:
        return _validate_ref_vol(v)

    @property
    def metric(self) -> str:
        return f"{self.metric_prefix}{self.ref_vol:g}"

    @classmethod
    def _from_voxels(
        cls,
        voxels: np.ndarray,
        unit: Optional[pint.Unit] = None,
        ref_vol: float = 50.0,
    ) -> Self:
        ref_vol = _validate_ref_vol(ref_vol)
        value = float(np.percentile(voxels, 100 - ref_vol)) if voxels.size > 0 else float("nan")
        return cls(value=value, unit=ureg.gray if unit is None else unit, ref_vol=ref_vol)

    @classmethod
    def compute_from(
        cls,
        quantity: DoseLike,
        mask: Optional[MaskLike] = None,
        unit: Optional[pint.Unit] = None,
        ref_vol: float = 50.0,
    ) -> Self:
        return cls._from_voxels(_extract_voxels(quantity, mask), unit=unit, ref_vol=ref_vol)


class VX(QI):
    """Volume percentage receiving at least the given reference dose (e.g. V20Gy)."""

    metric_prefix: ClassVar[str] = "V"

    qi_type: Literal["vx"] = "vx"
    ref_dose: float = Field(description="Reference dose")
    ref_unit: pint.Unit = Field(default=ureg.gray, description="Unit of the reference dose")

    @field_validator("ref_dose", mode="before")
    @classmethod
    def _validate_ref_dose(cls, v: Any) -> float:
        return _validate_ref_dose(v)

    @field_validator("ref_unit", mode="before")
    @classmethod
    def _validate_ref_unit(cls, v: Any) -> pint.Unit:
        return _coerce_unit(v)

    @field_serializer("ref_unit")
    def _serialize_ref_unit(self, v: pint.Unit) -> str:
        return str(v)

    @property
    def metric(self) -> str:
        return f"{self.metric_prefix}{self.ref_dose:g}{self.ref_unit:~}"

    @classmethod
    def _from_voxels(
        cls,
        voxels: np.ndarray,
        unit: Optional[pint.Unit] = None,
        ref_dose: float = 0.0,
        ref_unit: Optional[pint.Unit] = None,
        quantity_unit: Optional[pint.Unit] = None,
    ) -> Self:
        threshold, ref_unit, _ = _threshold_in_quantity_unit(ref_dose, ref_unit, quantity_unit)
        if voxels.size == 0:
            value = float("nan")
        else:
            value = float((voxels >= threshold).sum() / voxels.size * 100.0)
        return cls(
            value=value,
            unit=ureg.percent if unit is None else unit,
            ref_dose=float(ref_dose),
            ref_unit=ref_unit,
        )

    @classmethod
    def compute_from(
        cls,
        quantity: DoseLike,
        mask: Optional[MaskLike] = None,
        unit: Optional[pint.Unit] = None,
        ref_dose: float = 0.0,
        ref_unit: Optional[pint.Unit] = None,
        quantity_unit: Optional[pint.Unit] = None,
    ) -> Self:
        return cls._from_voxels(
            _extract_voxels(quantity, mask),
            unit=unit,
            ref_dose=ref_dose,
            ref_unit=ref_unit,
            quantity_unit=quantity_unit,
        )


# Discriminated union so dicts of QIs round-trip to their concrete subclasses.
AnyQI = Annotated[Union[Mean, Std, Max, Min, DX, VX], Field(discriminator="qi_type")]


class StructureQIs(PyRadPlanBaseModel):
    """Quality indicators for a single structure, keyed by metric id."""

    name: str = Field(description="Structure name")
    metrics: dict[str, AnyQI] = Field(
        default_factory=dict, description="Metrics keyed by metric id (e.g. 'mean', 'D50')"
    )

    def __getitem__(self, metric: str) -> QI:
        return self.metrics[metric]

    def __contains__(self, metric: object) -> bool:
        return metric in self.metrics

    def __iter__(self) -> Iterator[str]:
        return iter(self.metrics)

    def keys(self):
        return self.metrics.keys()

    def values(self):
        return self.metrics.values()

    def items(self):
        return self.metrics.items()


class QICollection(PyRadPlanBaseModel):
    """Quality indicators for every structure in a structure set, keyed by structure name."""

    structures: dict[str, StructureQIs] = Field(
        default_factory=dict, description="Structures keyed by name"
    )

    def __getitem__(self, structure: str) -> StructureQIs:
        return self.structures[structure]

    def __contains__(self, structure: object) -> bool:
        return structure in self.structures

    def __iter__(self) -> Iterator[StructureQIs]:
        return iter(self.structures.values())

    def __len__(self) -> int:
        return len(self.structures)

    @staticmethod
    def _default_ref_doses(dose: DoseLike) -> list[float]:
        """Five evenly-spaced reference doses in (0, max_dose]."""
        dose_arr = _to_array(dose)
        _validate_single_scenario(dose_arr, "Dose")
        max_dose = float(np.max(dose_arr))
        if max_dose <= 0.0:
            return []
        doses = [round(float(d), 1) for d in np.linspace(max_dose / 5, max_dose, 5)]
        return list(dict.fromkeys(doses))

    @classmethod
    def from_structure_set(
        cls,
        cst: StructureSet,
        dose: DoseLike,
        ref_vols: Optional[list[float]] = None,
        ref_doses: Optional[list[float]] = None,
        dose_unit: Optional[pint.Unit] = None,
        ref_unit: Optional[pint.Unit] = None,
    ) -> Self:
        """Compute the standard QIs for every VOI in ``cst``.

        Parameters
        ----------
        cst : StructureSet
            Structure set whose VOIs will be analyzed.
        dose : DoseLike
            Dose distribution. SimpleITK image or numpy array.
        ref_vols : list[float], optional
            Reference volumes (in %) for DX metrics. Defaults to ``[2, 5, 50, 95, 98]``.
        ref_doses : list[float], optional
            Reference doses for VX metrics. Defaults to five evenly-spaced
            doses in ``(0, max(dose)]`` derived from the dose distribution.
        dose_unit : pint.Unit, optional
            Unit of the values in ``dose``. Defaults to gray.
        ref_unit : pint.Unit, optional
            Unit of the ``ref_doses`` thresholds. Defaults to ``dose_unit``.
        """
        if ref_vols is None:
            ref_vols = [2, 5, 50, 95, 98]
        if ref_doses is None:
            ref_doses = cls._default_ref_doses(dose)

        dose_unit = _coerce_unit(ureg.gray if dose_unit is None else dose_unit)
        ref_unit = _coerce_unit(dose_unit if ref_unit is None else ref_unit)

        structures: dict[str, StructureQIs] = {}
        for voi in cst.vois:
            if voi.name in structures:
                raise ValueError(f"Duplicate VOI name in structure set: {voi.name}")
            voxels = _extract_voxels(dose, voi.mask)
            metrics: dict[str, QI] = {}
            for reducer_cls in (Mean, Std, Max, Min):
                qi = reducer_cls._from_voxels(voxels, unit=dose_unit)
                metrics[qi.metric] = qi
            for ref_vol in ref_vols:
                qi = DX._from_voxels(voxels, unit=dose_unit, ref_vol=float(ref_vol))
                metrics[qi.metric] = qi
            for ref_dose in ref_doses:
                qi = VX._from_voxels(
                    voxels,
                    ref_dose=float(ref_dose),
                    ref_unit=ref_unit,
                    quantity_unit=dose_unit,
                )
                metrics[qi.metric] = qi
            structures[voi.name] = StructureQIs(name=voi.name, metrics=metrics)

        return cls(structures=structures)

    def plot(
        self,
        structures: Optional[list[str]] = None,
        metrics: Optional[list[str]] = None,
        ax=None,
        **kwargs,
    ):
        """Render the QIs as a matplotlib table.

        Parameters
        ----------
        structures : list[str], optional
            Structure names to include. ``None`` shows all.
        metrics : list[str], optional
            Metric ids to include (e.g. ``["mean", "D50"]``). ``None`` shows all.
        ax : matplotlib.axes.Axes, optional
            Target axes. A new figure is created if not provided.
        **kwargs:
            Forwarded to ``ax.table``.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))

        selected = self._select_structures(structures)
        metric_keys = self._select_metric_keys(selected, metrics)
        cell_text = self._build_rows(selected, metric_keys)
        col_labels = [self._format_col_label(m, self._first_qi(selected, m)) for m in metric_keys]
        row_labels = [s.name for s in selected]

        table = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
            **kwargs,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax.axis("off")
        ax.set_title("Quality Indicators", fontsize=12, fontweight="bold", pad=20)
        return ax

    def _select_structures(self, names: Optional[list[str]]) -> list[StructureQIs]:
        if names is None:
            return list(self.structures.values())
        selected = [self.structures[n] for n in names if n in self.structures]
        if not selected:
            raise ValueError("None of the specified structures found in QI collection")
        return selected

    @staticmethod
    def _select_metric_keys(
        selected: list[StructureQIs], metrics: Optional[list[str]]
    ) -> list[str]:
        # Union of metric ids across structures, preserving first-seen order.
        available: list[str] = []
        for s in selected:
            for m in s.metrics:
                if m not in available:
                    available.append(m)
        if metrics is None:
            return available
        return [m for m in available if m in metrics]

    @staticmethod
    def _first_qi(selected: list[StructureQIs], metric: str) -> Optional[QI]:
        for s in selected:
            if metric in s.metrics:
                return s.metrics[metric]
        return None

    @staticmethod
    def _build_rows(selected: list[StructureQIs], metric_keys: list[str]) -> list[list[str]]:
        rows: list[list[str]] = []
        for s in selected:
            row: list[str] = []
            for m in metric_keys:
                qi = s.metrics.get(m)
                row.append("-" if qi is None or np.isnan(qi.value) else f"{qi.value:.2f}")
            rows.append(row)
        return rows

    @staticmethod
    def _format_col_label(metric: str, qi: Optional[QI]) -> str:
        if qi is None or not isinstance(qi.unit, pint.Unit):
            return metric
        unit_str = str(qi.unit)
        if unit_str == "percent":
            return f"{metric} [%]"
        return f"{metric} [{qi.unit:~}]"
