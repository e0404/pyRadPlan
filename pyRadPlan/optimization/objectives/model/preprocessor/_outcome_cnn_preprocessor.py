"""Config-driven preprocessor for CNN-based outcome objectives.

Reads all preprocessing parameters (dimensions, spacing, windowing,
normalization, mask collapsing, input ordering) from the model's JSON
config.  Uses SimpleITK for one-time CT/mask resampling and
``torch.nn.functional.grid_sample`` for differentiable per-iteration
dose resampling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple
import logging

import numpy as np
import SimpleITK as sitk
from pyRadPlan.core import xp_utils

if TYPE_CHECKING:
    from pyRadPlan.core import Grid

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    F = None  # type: ignore

logger = logging.getLogger(__name__)


class OutcomeCnnPreprocessor:
    """Config-driven preprocessor for CNN outcome objectives.

    All preprocessing parameters are read from the model JSON config:

    - ``model_preprocessing.input_dimensions`` — target grid size (X, Y, Z)
    - ``model_preprocessing.input_spacing`` — voxel spacing in mm
    - ``model_preprocessing.input_physical_extent`` — physical extent (optional,
      defaults to ``spacing * dimensions``)
    - ``model_preprocessing.center`` — ``"target"`` centres on target COM
    - ``model_preprocessing.modality.dose`` — normalization, extraction
    - ``model_preprocessing.modality.ct`` — HU window
    - ``model_preprocessing.modality.mask`` — collapse flag
    - ``model_input.type_order`` — channel concatenation order

    Parameters
    ----------
    config : dict
        Full JSON config loaded alongside the model weights.
    device : torch.device, optional
        Compute device.  Defaults to CUDA if available, else CPU.
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional["torch.device"] = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for OutcomeCnnPreprocessor.")

        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Extract preprocessing parameters from config ---
        preproc = config.get("model_preprocessing", {})

        self.input_dimensions: Tuple[int, ...] = tuple(preproc["input_dimensions"])
        self.input_spacing: Tuple[float, ...] = tuple(preproc["input_spacing"])
        # TODO: Maybe validate extent against calculated one.
        self.input_physical_extent: Tuple[float, ...] = tuple(
            preproc.get(
                "input_physical_extent",
                [s * d for s, d in zip(self.input_spacing, self.input_dimensions)],
            )
        )
        self.center_mode: str = preproc.get("center", "target")

        # Per-modality config
        modality = preproc.get("modality", {})
        self.dose_config: dict = modality.get("dose", {})
        self.ct_config: dict = modality.get("ct", {})
        self.mask_config: dict = modality.get("mask", {})

        # Model input channel ordering
        model_input = config.get("model_input", {})
        self.type_order: list[str] = model_input.get("type_order", ["dose", "ct", "mask"])

        # Prepared tensors — set by prepare()
        self._ct_tensor: Optional[torch.Tensor] = None  # (1, C, Z, Y, X)
        self._mask_tensor: Optional[torch.Tensor] = None  # (1, C, Z, Y, X)

        # Grid metadata
        self._model_grid: Optional[dict] = None
        self._dose_grid: Optional[dict] = None
        self._grid_sample_coords: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(
        self,
        ct,
        cst,
        cst_masks: Optional[list[str]] = None,
    ) -> None:
        """One-time preparation of CT, masks and grid geometry.

        Parameters
        ----------
        ct : CT
            CT object (``ct.cube_hu`` is a SimpleITK image).
        cst : StructureSet
            Structure set with VOIs and ``target_center_of_mass()``.
        cst_masks : list[str], optional
            VOI names to use as mask channels.  When *None*, all VOIs are
            used (subject to the ``collaps`` flag in the config).
        """
        # 1. Centre point
        # TODO: Define what happens if "body" or "OAR" is defined.
        if self.center_mode == "target":
            center = cst.target_center_of_mass()
        else:
            raise ValueError(f"Unknown center mode: {self.center_mode}")

        # 2. Model grid geometry
        ct_direction = tuple(np.asarray(ct.grid.direction).flatten())
        self._model_grid = self._compute_model_grid(center, ct_direction)

        # 3. Default dose grid = CT grid
        self._dose_grid = {
            "size": ct.grid.dimensions,
            "origin": tuple(ct.grid.origin),
            "spacing": tuple(ct.grid.resolution_vector),
            "direction": ct_direction,
        }

        # 4. Resample CT and masks (SimpleITK, once)
        resampler = self._make_sitk_resampler()
        self._prepare_ct(ct.cube_hu, cst, resampler)
        self._prepare_masks(cst, cst_masks, resampler)

        # 5. Precompute dose→model grid_sample coordinates
        self._precompute_grid_coords()

        logger.info(
            "OutcomeCnnPreprocessor ready: model grid %s @ %smm, device=%s",
            self._model_grid["size"],
            self._model_grid["spacing"],
            self.device,
        )

    def build_input_tensor(
        self,
        dose_values,
        requires_grad: bool = False,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", ...]:
        """Build model input tensors (dose, CT, mask) separately.

        Parameters
        ----------
        dose_values
            1-D Fortran-order flat dose array **or** 3-D ``(X, Y, Z)`` array.
            Accepts any array namespace (numpy, cupy, torch).
        requires_grad : bool
            If *True*, the returned tuple includes ``dose_leaf`` — the leaf
            tensor in dose-grid space with ``requires_grad=True``.

        Returns
        -------
        tuple
            ``(dose_model, ct_tensor, mask_tensor)`` when
            ``requires_grad=False``, or
            ``(dose_model, ct_tensor, mask_tensor, dose_leaf)`` when *True*.
        """
        if self._grid_sample_coords is None:
            raise RuntimeError("Preprocessor not prepared — call prepare() first.")

        # --- Dose: any array → torch (Z, Y, X) ---
        dose_src = self._dose_to_tensor(dose_values)
        if requires_grad:
            dose_src = dose_src.requires_grad_(True)

        # Resample to model grid (differentiable)
        dose_5d = dose_src.unsqueeze(0).unsqueeze(0)  # (1, 1, Z, Y, X)
        dose_model = F.grid_sample(
            dose_5d,
            self._grid_sample_coords,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # (1, 1, mZ, mY, mX)

        # Dose normalization
        norm_value = self.dose_config.get("normalization_value")
        if norm_value is not None:
            dose_model = dose_model / norm_value

        # Dose extraction (element-wise masking)
        if self.dose_config.get("extract", False):
            dose_model = dose_model * self._mask_tensor

        if requires_grad:
            return dose_model, self._ct_tensor, self._mask_tensor, dose_src
        return dose_model, self._ct_tensor, self._mask_tensor

    def set_dose_grid(self, dose_grid: "Grid") -> None:
        """Update the dose grid and recompute sampling coordinates.

        Parameters
        ----------
        dose_grid : Grid
            The (possibly different) dose grid used during optimisation.
        """
        if hasattr(dose_grid, "dimensions"):
            self._dose_grid = {
                "size": dose_grid.dimensions,
                "origin": tuple(dose_grid.origin),
                "spacing": tuple(dose_grid.resolution_vector),
                "direction": tuple(np.asarray(dose_grid.direction).flatten()),
            }
        else:
            # SimpleITK image fallback
            self._dose_grid = {
                "size": dose_grid.GetSize(),
                "origin": dose_grid.GetOrigin(),
                "spacing": dose_grid.GetSpacing(),
                "direction": dose_grid.GetDirection(),
            }
        self._precompute_grid_coords()
        logger.debug(
            "Dose grid updated: %s @ %smm", self._dose_grid["size"], self._dose_grid["spacing"]
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_model_grid(self, center: np.ndarray, direction: tuple) -> dict:
        """Compute model grid origin so it is centred on *center*."""
        size = self.input_dimensions
        spacing = self.input_spacing

        dir_mat = np.asarray(direction, dtype=np.float64).reshape(3, 3)
        center_idx = np.array([(s - 1) / 2.0 for s in size])
        offset = dir_mat @ (np.array(spacing) * center_idx)
        origin = tuple((np.asarray(center, dtype=np.float64) - offset).tolist())

        return {
            "size": size,
            "origin": origin,
            "spacing": spacing,
            "direction": direction,
        }

    def _make_sitk_resampler(self) -> sitk.ResampleImageFilter:
        mg = self._model_grid
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize([int(s) for s in mg["size"]])
        resampler.SetOutputSpacing([float(v) for v in mg["spacing"]])
        resampler.SetOutputOrigin([float(v) for v in mg["origin"]])
        resampler.SetOutputDirection([float(v) for v in mg["direction"]])
        return resampler

    # --- CT ---

    def _prepare_ct(self, ct_itk: sitk.Image, cst, resampler: sitk.ResampleImageFilter) -> None:
        window = self.ct_config.get("window", [-1024, 3071])
        lo, hi = float(window[0]), float(window[1])

        # Mask outside body
        body = next((v for v in cst.vois if v.name.upper() == "BODY"), None)
        if body is not None:
            ct_itk = sitk.Mask(ct_itk, body.mask, outsideValue=lo, maskingValue=0)

        ct_clamped = sitk.Clamp(ct_itk, lowerBound=lo, upperBound=hi)

        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(lo)
        ct_resampled = resampler.Execute(ct_clamped)

        # Normalize to [0, 1]
        ct_np = sitk.GetArrayFromImage(ct_resampled).astype(np.float32)  # (Z, Y, X)
        ct_np = (ct_np - lo) / (hi - lo)

        self._ct_tensor = (
            torch.from_numpy(ct_np).unsqueeze(0).unsqueeze(0).to(self.device)
        )  # (1, 1, Z, Y, X)

    # --- Masks ---

    def _prepare_masks(
        self,
        cst,
        cst_masks: Optional[list[str]],
        resampler: sitk.ResampleImageFilter,
    ) -> None:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)

        # Resolve VOIs
        if cst_masks is not None:
            vois = []
            for name in cst_masks:
                voi = next((v for v in cst.vois if v.name.lower() == name.lower()), None)
                if voi is None:
                    available = [v.name for v in cst.vois]
                    raise ValueError(f"VOI '{name}' not found. Available: {available}")
                vois.append(voi)
        else:
            vois = list(cst.vois)

        # Resample each mask
        arrays = [
            sitk.GetArrayFromImage(resampler.Execute(v.mask)).astype(np.float32) for v in vois
        ]

        # Collapse if requested
        if self.mask_config.get("collaps", False) and len(arrays) > 1:
            combined = arrays[0]
            for a in arrays[1:]:
                combined = np.maximum(combined, a)
            arrays = [combined]

        masks_np = np.stack(arrays, axis=0)  # (C, Z, Y, X)
        self._mask_tensor = (
            torch.from_numpy(masks_np).unsqueeze(0).to(self.device)
        )  # (1, C, Z, Y, X)

    # --- Grid-sample coordinate precomputation ---

    def _precompute_grid_coords(self) -> None:
        """Compute normalised ``[-1, 1]`` coordinates for ``grid_sample``.

        Maps every voxel centre in the *model* grid to the corresponding
        normalised position in the *dose* grid.  The result is stored as
        ``self._grid_sample_coords`` with shape ``(1, mZ, mY, mX, 3)``.
        """
        mg = self._model_grid
        dg = self._dose_grid
        mx, my, mz = mg["size"]

        # Model voxel indices → physical coordinates
        ix = torch.arange(mx, dtype=torch.float32)
        iy = torch.arange(my, dtype=torch.float32)
        iz = torch.arange(mz, dtype=torch.float32)
        gz, gy, gx = torch.meshgrid(iz, iy, ix, indexing="ij")  # (mZ, mY, mX)

        indices = torch.stack(
            [gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1
        )  # (N, 3) — x, y, z

        m_origin = torch.tensor(mg["origin"], dtype=torch.float32)
        m_spacing = torch.tensor(mg["spacing"], dtype=torch.float32)
        m_dir = torch.tensor(mg["direction"], dtype=torch.float32).reshape(3, 3)

        phys = m_origin + (indices * m_spacing) @ m_dir.T  # (N, 3)

        # Physical coordinates → dose voxel indices
        d_origin = torch.tensor(dg["origin"], dtype=torch.float32)
        d_spacing = torch.tensor(dg["spacing"], dtype=torch.float32)
        d_dir = torch.tensor(dg["direction"], dtype=torch.float32).reshape(3, 3)
        d_dir_inv = torch.linalg.inv(d_dir)

        dose_idx = ((phys - d_origin) @ d_dir_inv.T) / d_spacing  # (N, 3)

        # Normalise to [-1, 1] (align_corners=True convention)
        d_size = torch.tensor(dg["size"], dtype=torch.float32)
        normalised = 2.0 * dose_idx / (d_size - 1) - 1.0  # (N, 3)

        # grid_sample 5-D: grid[..., 0]=W(X), grid[..., 1]=H(Y), grid[..., 2]=D(Z)
        self._grid_sample_coords = normalised.reshape(1, int(mz), int(my), int(mx), 3).to(
            self.device
        )

    # --- Dose helper ---

    def _dose_to_tensor(self, dose_values) -> "torch.Tensor":
        """Convert dose array (any array namespace) to a (Z, Y, X) tensor on device."""

        t = xp_utils.to_namespace(torch, dose_values).to(dtype=torch.float32)
        if t.ndim == 1:
            dx, dy, dz = self._dose_grid["size"]
            # Fortran-order reshape (X,Y,Z) == C-order reshape (Z,Y,X)
            t = t.reshape(dz, dy, dx)
        else:
            # Input is (X,Y,Z) → permute to (Z,Y,X)
            t = t.permute(2, 1, 0)
        return t.contiguous().to(self.device).detach()
