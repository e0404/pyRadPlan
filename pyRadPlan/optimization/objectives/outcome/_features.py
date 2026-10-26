"""Feature extraction for feature-based outcome models.

Feature-based outcome models (logistic regression, SVM, small MLPs, ...) do not
consume dose volumes but a handful of scalar descriptors computed per segment.
This module provides those descriptors together with their analytic dose
gradient, so an objective can chain ``dp/dfeature`` back onto the dose vector.

Two kinds of feature exist:

* *dose* features (mean dose, Dx, ...) are recomputed from the segment's dose
  vector on every iteration and carry a non-zero gradient.
* *geometry* features (volume, sphericity, ...) depend only on the segment mask,
  are computed once on the CT grid, and have a zero dose gradient.

Names and formulas follow pyanno4rt so that a model trained there yields the
same values here.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Optional

import array_api_compat
import numpy as np
import SimpleITK as sitk

from ....ai.modelhub import OutcomeFeature
from ....core import Grid
from ....core.resample import resample_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dose features
# ---------------------------------------------------------------------------


def _one_hot(values, xp, index):
    """Sub-gradient of an order statistic: unit weight on the selected voxel."""
    grad = xp.zeros_like(values)
    grad[index] = 1.0
    return grad


def _mean(values, xp, argument=None):
    return xp.mean(values)


def _mean_gradient(values, xp, argument=None):
    return xp.full_like(values, 1.0 / array_api_compat.size(values))


def _maximum(values, xp, argument=None):
    return xp.max(values)


def _maximum_gradient(values, xp, argument=None):
    return _one_hot(values, xp, xp.argmax(values))


def _minimum(values, xp, argument=None):
    return xp.min(values)


def _minimum_gradient(values, xp, argument=None):
    return _one_hot(values, xp, xp.argmin(values))


def _deviation(values, xp, argument=None):
    return xp.std(values)


def _deviation_gradient(values, xp, argument=None):
    std = xp.std(values)
    if float(std) == 0.0:
        return xp.zeros_like(values)
    return (values - xp.mean(values)) / (array_api_compat.size(values) * std)


def _dx_position(values, argument) -> int:
    """Index into the sorted dose vector for a Dx feature (MATLAB-style rounding)."""
    n = array_api_compat.size(values)
    position = n * (1.0 - float(argument) / 100.0)
    # numpy rounds halves to even, MATLAB (and hence pyanno4rt) rounds them up.
    rounded = np.ceil(position) if (position - np.floor(position)) == 0.5 else np.round(position)
    return int(np.clip(rounded, 0, n - 1))


def _dx(values, xp, argument):
    return xp.sort(values)[_dx_position(values, argument)]


def _dx_gradient(values, xp, argument):
    return _one_hot(values, xp, xp.argsort(values)[_dx_position(values, argument)])


def _vx(values, xp, argument):
    exceeding = xp.astype(values >= float(argument), values.dtype)
    return xp.sum(exceeding) / array_api_compat.size(values)


def _zero_gradient(values, xp, argument=None):
    return xp.zeros_like(values)


#: Dose-dependent features: ``name -> (value, gradient)``.
DOSE_FEATURES: dict[str, tuple[Callable, Callable]] = {
    "Dose Mean": (_mean, _mean_gradient),
    "Dose Maximum": (_maximum, _maximum_gradient),
    "Dose Minimum": (_minimum, _minimum_gradient),
    "Dose Deviation": (_deviation, _deviation_gradient),
    "Dose Dx": (_dx, _dx_gradient),
    # Vx counts voxels above a threshold; its derivative vanishes almost everywhere.
    "Dose Vx": (_vx, _zero_gradient),
}


# ---------------------------------------------------------------------------
# Geometry features
# ---------------------------------------------------------------------------


def segment_volume(mask: np.ndarray, resolution: np.ndarray) -> float:
    """Segment volume in mm^3."""
    return float(np.sum(mask) * np.prod(resolution))


def segment_area(mask: np.ndarray, resolution: np.ndarray) -> float:
    """Segment surface area in mm^2, counting mask transitions along each axis.

    Notes
    -----
    Following pyanno4rt, the first slice along each axis contributes its full
    cross-section instead of a difference, which closes the surface at the
    leading but not at the trailing end. The in-plane voxel area here is the
    product of the two resolutions orthogonal to the axis; pyanno4rt pairs the
    resolutions differently, which matters only for anisotropic grids.
    """
    area = 0.0
    mask = np.asarray(mask, dtype=np.int8)
    for axis in range(mask.ndim):
        if mask.shape[axis] < 2:
            continue
        slices = np.moveaxis(mask, axis, 0)
        transitions = float(slices[0].sum()) + float(np.abs(np.diff(slices[1:], axis=0)).sum())
        area += transitions * float(np.prod(np.delete(resolution, axis)))
    return area


def segment_eigenvalues(mask: np.ndarray, resolution: np.ndarray) -> np.ndarray:
    """Eigenvalues of the segment's spatial covariance matrix (mm^2)."""
    indices = np.nonzero(mask)
    if len(indices[0]) == 0:
        raise ValueError("Cannot compute geometry features for an empty segment.")

    coordinates = np.stack(
        [index.astype(np.float64) * spacing for index, spacing in zip(indices, resolution)]
    )
    coordinates -= coordinates.mean(axis=1, keepdims=True)
    return np.linalg.eigvals(coordinates @ coordinates.T).real


def segment_sphericity(mask: np.ndarray, resolution: np.ndarray) -> float:
    """Sphericity, i.e. the surface area of an equal-volume sphere over the segment's area."""
    area = segment_area(mask, resolution)
    if area == 0.0:
        raise ValueError("Cannot compute sphericity for a segment with zero surface area.")
    return float(np.pi ** (1 / 3) * (6.0 * segment_volume(mask, resolution)) ** (2 / 3) / area)


def segment_eccentricity(mask: np.ndarray, resolution: np.ndarray) -> float:
    """Eccentricity derived from the segment's principal spatial moments."""
    eigenvalues = segment_eigenvalues(mask, resolution)
    largest = float(np.max(eigenvalues))
    if largest <= 0.0:
        return 0.0
    return float(1.0 - np.sqrt(max(float(np.min(eigenvalues)), 0.0) / largest))


#: Dose-independent features: ``name -> value(mask, resolution)``.
GEOMETRY_FEATURES: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "Segment Volume": segment_volume,
    "Segment Area": segment_area,
    "Segment Sphericity": segment_sphericity,
    "Segment Eccentricity": segment_eccentricity,
}


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """Turn a dose vector on the dose grid into an outcome model's feature vector.

    Parameters
    ----------
    features : Iterable[OutcomeFeature]
        Model inputs, in the order the model expects them.
    ct : CT
        Planning CT; geometry features are evaluated on its grid, matching how
        the model was trained.
    cst : StructureSet
        Structure set holding the segments the features refer to.
    segment_map : dict[str, str], optional
        Mapping from the feature's segment name to the VOI name in ``cst``.
        Names are otherwise matched case-insensitively.
    """

    def __init__(
        self,
        features: Iterable[OutcomeFeature],
        ct: Any,
        cst: Any,
        segment_map: Optional[dict[str, str]] = None,
    ) -> None:
        self.features = list(features)
        segment_map = segment_map or {}
        self._masks = {
            segment: self._find_mask(cst, segment_map.get(segment, segment), segment)
            for segment in dict.fromkeys(feature.segment for feature in self.features)
        }
        self._static = self._compute_geometry(ct)
        self.dose_feature_indices = tuple(
            position
            for position, feature in enumerate(self.features)
            if feature.function in DOSE_FEATURES
        )
        #: Dose-grid voxel indices per segment, filled by :meth:`set_dose_grid`.
        self.segment_indices: dict[str, np.ndarray] = {}

    @staticmethod
    def _find_mask(cst: Any, wanted: str, segment: str) -> sitk.Image:
        """Resolve a feature's segment name to a VOI mask in ``cst``."""
        for voi in cst.vois:
            if voi.name == wanted:
                return voi.mask
        for voi in cst.vois:
            if voi.name.casefold() == wanted.casefold():
                return voi.mask
        available = [voi.name for voi in cst.vois]
        raise ValueError(
            f"Segment '{segment}' (looked up as '{wanted}') is not in the structure set. "
            f"Available VOIs: {available}. Use 'segment_map' to map the names."
        )

    def _compute_geometry(self, ct: Any) -> dict[int, float]:
        """Evaluate all dose-independent features once, on the CT grid."""
        # sitk arrays are (z, y, x), so the (x, y, z) resolution has to be reversed.
        resolution = np.asarray(ct.grid.resolution_vector, dtype=np.float64)[::-1]

        static = {}
        for position, feature in enumerate(self.features):
            if feature.function in DOSE_FEATURES:
                continue
            if feature.function not in GEOMETRY_FEATURES:
                raise ValueError(
                    f"Unknown feature function '{feature.function}'. Supported: "
                    f"{sorted(DOSE_FEATURES)} and {sorted(GEOMETRY_FEATURES)}."
                )
            mask = sitk.GetArrayViewFromImage(self._masks[feature.segment]) > 0
            static[position] = float(GEOMETRY_FEATURES[feature.function](mask, resolution))

        if static:
            logger.info(
                "Geometry features: %s",
                {self.features[position].name: value for position, value in static.items()},
            )
        return static

    def set_dose_grid(self, dose_grid: Grid) -> None:
        """Resample the segment masks onto the optimization dose grid."""
        self.segment_indices = {}
        for segment, mask in self._masks.items():
            resampled = resample_image(
                input_image=mask,
                interpolator=sitk.sitkNearestNeighbor,
                target_grid=dose_grid,
                extrapolate=0,
            )
            # numpy (C) order, matching the dose vector the optimizer hands to objectives
            indices = np.flatnonzero(sitk.GetArrayFromImage(resampled).reshape(-1) > 0.5)
            if indices.size == 0:
                raise ValueError(f"Segment '{segment}' is empty on the dose grid.")
            self.segment_indices[segment] = indices

    def _segment_dose(self, dose, xp, segment: str):
        if not self.segment_indices:
            raise RuntimeError("FeatureExtractor has no dose grid; call set_dose_grid() first.")
        indices = xp.asarray(self.segment_indices[segment], device=array_api_compat.device(dose))
        return indices, dose[indices]

    def compute_values(self, dose, xp) -> np.ndarray:
        """Return the feature vector of shape ``(n_features,)``."""
        values = np.empty(len(self.features), dtype=np.float64)
        for position, feature in enumerate(self.features):
            if position in self._static:
                values[position] = self._static[position]
                continue
            value_fn, _ = DOSE_FEATURES[feature.function]
            _, segment_dose = self._segment_dose(dose, xp, feature.segment)
            values[position] = float(value_fn(segment_dose, xp, feature.argument))
        return values

    def compute_dose_gradients(self, dose, xp):
        """Yield ``(feature_index, voxel_indices, d feature / d dose)`` per dose feature."""
        for position in self.dose_feature_indices:
            feature = self.features[position]
            _, gradient_fn = DOSE_FEATURES[feature.function]
            indices, segment_dose = self._segment_dose(dose, xp, feature.segment)
            yield position, indices, gradient_fn(segment_dose, xp, feature.argument)
