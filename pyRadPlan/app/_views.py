"""View products derived from the workspace: volumes, structure masks, beams and DVHs.

Every array is delivered on the CT grid in ``(z, y, x)`` order (the order of
``sitk.GetArrayFromImage``), so all volumes of a session share one grid and frontends
never resample. Products are cached by the identity of the objects they derive from and
dropped when the workspace reports a change of those objects.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from ._results import _is_volume, result_components

#: Result keys preferred as the initially shown quantity (Qt ``default_quantity_key``).
DEFAULT_QUANTITY_KEYS: tuple[str, ...] = (
    "rbe_x_dose",
    "physical_dose",
    "Physical quantity",
    "physical_quantity",
)

#: Units of result keys that are no registered quantity identifiers (matRad names).
_EXTRA_UNITS = {"physicalDose": "Gy", "RBExDose": "Gy (RBE)", "LET": "keV/µm"}

#: Planes of the ``"bits"`` mask encoding hold this many structures each.
BITS_PER_PLANE = 32


def resolve_structure(vois: Sequence[Any], ref: Any) -> int:
    """Return the position of the structure *ref* (a position or a unique name) in *vois*.

    Names need not be unique; a name shared by several structures is refused, so a
    frontend has to address those by position.
    """
    if isinstance(ref, bool) or not isinstance(ref, (int, str)):
        raise TypeError(f"A structure is addressed by position or name, not {ref!r}")
    if isinstance(ref, int):
        if not 0 <= ref < len(vois):
            raise KeyError(f"No structure at position {ref}")
        return ref
    matches = [i for i, voi in enumerate(vois) if voi.name == ref]
    if not matches:
        raise KeyError(f"Unknown structure {ref!r}")
    if len(matches) > 1:
        raise KeyError(
            f"Structure name {ref!r} is ambiguous (positions {matches}); use a position"
        )
    return matches[0]


def default_quantity(keys: Sequence[str]) -> Optional[str]:
    """Return the quantity to show first: RBE-weighted dose, else dose, else the first."""
    for candidate in DEFAULT_QUANTITY_KEYS:
        if candidate in keys:
            return candidate
    return keys[0] if keys else None


def quantity_unit(key: str) -> str:
    """Return the unit symbol of a result key; ``""`` if unknown."""
    if key == "ct":
        return "HU"
    from pyRadPlan.quantities import get_available_quantities  # noqa: PLC0415

    quantity = get_available_quantities().get(key)
    if quantity is None:
        return _EXTRA_UNITS.get(key, "")
    try:
        return f"{quantity.unit:~P}"
    except Exception:  # noqa: BLE001 - a unit is cosmetic
        return str(quantity.unit)


def component_info(result: Any, key: str) -> dict[str, Any]:
    """Quantity, snapshot tag, component kind and labels of a result component key.

    Grouped as the result catalog groups (``gui/widgets/_display_names.py``
    ``group_result_keys``), so the labels match ``state.results``: ``label`` is the Qt
    selector label (``"dose (tag), beam 0"``), ``meta_label`` / ``unit`` the colorbar and
    DVH axis label and unit (Qt ``QUANTITY_META``, ``"Quantity"`` / ``""`` if unknown).
    """
    from pyRadPlan.quantities._display_names import group_result_keys  # noqa: PLC0415

    from ._results import quantity_meta  # noqa: PLC0415

    keys = list(result_components(result))
    if key not in keys:
        keys.append(key)
    for group in group_result_keys(keys):
        for (component, member), kind in zip(group.components, group.kinds):
            if member != key:
                continue
            meta_label, unit = quantity_meta(group.base)
            label = group.label if component == "full" else f"{group.label}, {component}"
            return {
                "key": key,
                "base": group.base,
                "tag": group.tag,
                "kind": kind or "other",
                "label": label,
                "meta_label": meta_label,
                "unit": unit,
            }
    raise KeyError(f"No volume quantity {key!r} in the result")


def _image_3d(image: Any, phase: int = 0) -> Any:
    if image.GetDimension() == 4:
        return image[:, :, :, int(phase)]
    return image


def grid_info(ct: Any) -> dict[str, Any]:
    """Shape ``(nz, ny, nx)``, spacing / origin ``(x, y, z)`` and 3x3 direction of the CT."""
    image = ct.cube_hu
    dim = image.GetDimension()
    size = image.GetSize()[:3]
    direction = image.GetDirection()
    if dim == 4:
        direction = [direction[i * 4 + j] for i in range(3) for j in range(3)]
    return {
        "shape": [int(size[2]), int(size[1]), int(size[0])],
        "spacing": [float(v) for v in image.GetSpacing()[:3]],
        "origin": [float(v) for v in image.GetOrigin()[:3]],
        "direction": [float(v) for v in direction],
    }


def result_quantities(result: Any) -> list[str]:
    """Return the result keys holding one 3-D volume (no beam/scenario lists, no weights)."""
    if not isinstance(result, dict):
        return []
    return [key for key, value in result.items() if _is_volume(value)]


def result_entry(result: Any, key: str) -> Any:
    """Return the 3-D volume of a full or component key, ``None`` if there is none."""
    return result_components(result).get(key)


def _quantity_on_ct_grid(value: Any, ct: Any) -> np.ndarray:
    import SimpleITK as sitk  # noqa: PLC0415

    reference = _image_3d(ct.cube_hu)
    nx, ny, nz = reference.GetSize()
    if isinstance(value, sitk.Image):
        image = value
        if (
            image.GetSize() != reference.GetSize()
            or not np.allclose(image.GetSpacing(), reference.GetSpacing())
            or not np.allclose(image.GetOrigin(), reference.GetOrigin())
            or not np.allclose(image.GetDirection(), reference.GetDirection())
        ):
            from pyRadPlan.core.resample import resample_image  # noqa: PLC0415

            image = resample_image(
                image, interpolator=sitk.sitkLinear, target_image=reference, extrapolate=0
            )
        return sitk.GetArrayFromImage(image)
    array = np.asarray(value)
    if array.shape != (nz, ny, nx):
        raise ValueError(
            f"Result array of shape {array.shape} does not match the CT grid "
            f"(z, y, x) = {(nz, ny, nx)}"
        )
    return array


def _finite_range(array: np.ndarray) -> list[Optional[float]]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return [None, None]
    return [float(finite.min()), float(finite.max())]


def _cumulative_dvh(
    values: np.ndarray, num_points: int, max_value: Optional[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Left bin edges and cumulative volume in percent (:meth:`pyRadPlan.analysis.DVH.compute`).

    ``max_value=None`` bins from 0 to 105 % of the highest of *values*, as the Qt
    analysis window does per structure.
    """
    if values.size == 0:
        edges = np.linspace(0.0, max_value or 1.0, num_points + 1)
        return edges[:-1], np.zeros(num_points)
    from pyRadPlan.analysis import DVH  # noqa: PLC0415

    dvh = DVH.compute(values, num_points=num_points, max_value=max_value)
    return dvh.bins, dvh.cum_volume


def _bbox(mask: np.ndarray) -> Optional[list[int]]:
    """``[z0, y0, x0, z1, y1, x1]`` (ends exclusive) of the set voxels, ``None`` if empty."""
    lower, upper = [], []
    for axis in range(3):
        others = tuple(a for a in range(3) if a != axis)
        hits = np.flatnonzero(mask.any(axis=others))
        if hits.size == 0:
            return None
        lower.append(int(hits[0]))
        upper.append(int(hits[-1]) + 1)
    return lower + upper


def _plane_dtype(count: int) -> np.dtype:
    if count <= 8:
        return np.dtype(np.uint8)
    if count <= 16:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


# ----------------------------------------------------------------------
# Beam geometry in LPS millimetres (the Qt viewer maps it to its voxel space)
# ----------------------------------------------------------------------


def _as_angle_array(value: Any) -> np.ndarray:
    """Flatten a ``prop_stf`` angle entry (list, scalar, array or None) to 1-D floats."""
    if value is None:
        return np.empty(0)
    return np.atleast_1d(np.asarray(value, dtype=float)).ravel()


def plan_beam_angles(pln: Any) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Gantry and couch angle of every beam of *pln*, ``None`` if it defines no beams."""
    return _beam_angles(getattr(pln, "prop_stf", None) or {})


def plan_iso_centers(pln: Any, cst: Any) -> Optional[np.ndarray]:
    """Isocenters of *pln* in mm, shape ``(k, 3)``, resolved as the stf generators do.

    ``None`` when the plan sets none and there is no target to derive it from.
    """
    return _iso_centers((getattr(pln, "prop_stf", None) or {}).get("iso_center"), cst)


def _beam_angles(prop_stf: Mapping[str, Any]) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Gantry and couch angle of every beam (a single couch angle applies to all)."""
    gantry_angles = _as_angle_array(prop_stf.get("gantry_angles"))
    if gantry_angles.size == 0:
        return None
    couch_angles = _as_angle_array(prop_stf.get("couch_angles"))
    if couch_angles.size == 0:
        couch_angles = np.zeros_like(gantry_angles)
    elif couch_angles.size == 1:
        couch_angles = np.full_like(gantry_angles, couch_angles[0])
    if couch_angles.size != gantry_angles.size:
        return None
    return gantry_angles, couch_angles


def _same_angles(a: Sequence[float], b: Sequence[float]) -> bool:
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bool(np.allclose((difference + 180.0) % 360.0 - 180.0, 0.0, atol=1e-6))


def _source_direction(gantry: float, couch: float) -> np.ndarray:
    from pyRadPlan.geometry import lps  # noqa: PLC0415

    # The source sits at -y in the beam's eye view (Beam.derive_source_points).
    rotation = lps.get_beam_rotation_matrix(float(gantry), float(couch))
    return np.asarray(rotation @ np.array([0.0, -1.0, 0.0]), dtype=float)


def _orbit(couch: float) -> dict[str, list[float]]:
    u = _source_direction(0.0, couch)
    w = _source_direction(90.0, couch)
    return {
        "orbit_u": [float(v) for v in u],
        "orbit_w": [float(v) for v in w],
        "orbit_axis": [float(v) for v in np.cross(u, w)],
    }


def _iso_centers(iso_center: Any, cst: Any) -> Optional[np.ndarray]:
    from pyRadPlan.stf import resolve_iso_centers  # noqa: PLC0415

    try:
        return resolve_iso_centers(iso_center, cst)
    except Exception:  # noqa: BLE001 - the isocenter is optional (no target, no cst)
        return None


def _vec(values: Any) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=float).ravel()]


_RESOLVE = object()


def beam_geometry(
    pln: Any,
    stf: Any,
    cst: Any,
    draft: Optional[Mapping[str, Any]] = None,
    *,
    iso_centers: Any = _RESOLVE,
) -> dict[str, Any]:
    """Beam axes of the plan (or of an unapplied *draft* of its angles) in LPS millimetres.

    The geometry both the Qt viewer and the web viewer draw: the stf beams are used while
    their angles equal the plan's (or the draft's), else the angles themselves; with plan
    angles the plan isocenters take precedence over the stf's, as Recalculate Dose moves
    the beams to them. Beams with plan angles carry their gantry orbit (``orbit_u`` /
    ``orbit_w``, the source directions at gantry 0 and 90 degrees, and
    ``orbit_axis = u x w``), so a viewer can rotate them.

    Parameters
    ----------
    draft
        ``{gantry_angles, couch_angles?, iso_center?}`` of the plan form; a given
        ``iso_center`` (``None``: automatic) replaces the plan's. Draft beams whose angles
        differ from the applied plan are ``pending``.
    iso_centers
        The isocenters of the plan (with the draft applied), if the caller resolved them
        already (:func:`plan_iso_centers`); resolved here when omitted.
    """
    prop_stf = dict(getattr(pln, "prop_stf", None) or {})
    applied = _beam_angles(prop_stf)
    if draft is not None:
        prop_stf["gantry_angles"] = draft.get("gantry_angles")
        prop_stf["couch_angles"] = draft.get("couch_angles")
        if "iso_center" in draft:
            prop_stf["iso_center"] = draft["iso_center"]
    angles = _beam_angles(prop_stf)
    if iso_centers is _RESOLVE:
        iso_centers = _iso_centers(prop_stf.get("iso_center"), cst)

    def plan_iso(i: int) -> np.ndarray:
        return iso_centers[min(i, len(iso_centers) - 1)]

    beams: list[dict[str, Any]] = []
    source = None
    stf_beams = getattr(stf, "beams", None)
    if stf_beams and (
        angles is None
        or (
            len(stf_beams) == angles[0].size
            and _same_angles([b.gantry_angle for b in stf_beams], angles[0])
            and _same_angles([b.couch_angle for b in stf_beams], angles[1])
        )
    ):
        source = "stf"
        use_plan_iso = angles is not None and iso_centers is not None
        for i, beam in enumerate(stf_beams):
            direction = np.asarray(beam.source_point, dtype=float)
            norm = float(np.linalg.norm(direction))
            if norm == 0.0:
                continue
            gantry, couch = float(beam.gantry_angle), float(beam.couch_angle)
            entry = {
                "index": i,
                "gantry": gantry,
                "couch": couch,
                "iso_center": _vec(plan_iso(i) if use_plan_iso else beam.iso_center),
                "source_direction": _vec(direction / norm),
            }
            if angles is not None:
                entry.update(_orbit(couch))
            beams.append(entry)
    elif angles is not None and iso_centers is not None:
        source = "plan"
        for i, (gantry, couch) in enumerate(zip(*angles)):
            beams.append(
                {
                    "index": i,
                    "gantry": float(gantry),
                    "couch": float(couch),
                    "iso_center": _vec(plan_iso(i)),
                    "source_direction": _vec(_source_direction(gantry, couch)),
                    **_orbit(couch),
                }
            )
    for entry in beams:
        i = entry["index"]
        entry["name"] = f"#{i}"
        entry["label"] = f"Beam #{i}: gantry {entry['gantry']:g}°, couch {entry['couch']:g}°"
        entry["pending"] = draft is not None and not (
            applied is not None
            and i < applied[0].size
            and _same_angles([entry["gantry"], entry["couch"]], [applied[0][i], applied[1][i]])
        )
    return {
        "source": source,
        "iso_centers": [] if iso_centers is None else [_vec(iso) for iso in iso_centers],
        "beams": beams,
    }


class ViewCache:
    """Cached view products of one workspace.

    Every product is stored with the identities of the workspace objects it was derived
    from (the CT, a result entry, a structure mask, ...). A lookup rebuilds it when one of
    them was replaced; :meth:`invalidate` drops products whose sources were replaced, so
    no stale arrays are kept alive. Products survive changes that do not touch their
    sources, e.g. an objective edit keeps every mask and an optimization keeps the CT.

    Parameters
    ----------
    get
        Returns the workspace object stored under a key.
    """

    def __init__(self, get: Callable[[str], Any]) -> None:
        self._get = get
        self._cache: dict[tuple, tuple[tuple, Any]] = {}

    def _identity(self, key: tuple) -> Optional[tuple]:
        """Identities of the current sources of *key*; ``None`` if a source is gone."""
        ct = self._get("ct")
        if ct is None:
            return None
        kind = key[0]
        if kind == "volume":
            name = key[1]
            if name == "ct":
                return (ct, ct.cube_hu)
            source = result_entry(self._get("result"), name)
            return None if source is None else (ct, source)
        cst = self._get("cst")
        if cst is None:
            return None
        masks = [voi.mask for voi in cst.vois]

        def selected(indices: tuple) -> Optional[list]:
            if any(not 0 <= i < len(masks) for i in indices):
                return None
            return [masks[i] for i in indices]

        if kind in ("mask", "mask_info"):
            chosen = selected((key[1],))
            return None if chosen is None else (ct, *chosen)
        if kind == "masks":
            chosen = selected(key[2])
            return None if chosen is None else (ct, *chosen)
        if kind in ("dvh", "dvh_curves"):
            source = result_entry(self._get("result"), key[1])
            chosen = selected(key[2])
            if source is None or chosen is None:
                return None
            return (ct, source, *chosen)
        return None

    @staticmethod
    def _same(a: tuple, b: Optional[tuple]) -> bool:
        return b is not None and len(a) == len(b) and all(x is y for x, y in zip(a, b))

    def invalidate(self) -> None:
        """Drop the products whose sources were replaced or removed."""
        for key, (identity, _) in list(self._cache.items()):
            if not self._same(identity, self._identity(key)):
                del self._cache[key]

    def clear(self) -> None:
        """Drop every product."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def _cached(self, key: tuple, build: Callable[[], Any]) -> Any:
        identity = self._identity(key)
        hit = self._cache.get(key)
        if hit is not None and self._same(hit[0], identity):
            return hit[1]
        value = build()
        if identity is not None:
            self._cache[key] = (identity, value)
        return value

    def _require_ct(self) -> Any:
        ct = self._get("ct")
        if ct is None:
            raise LookupError("No CT loaded")
        return ct

    def volume_array(self, name: str, phase: int = 0) -> np.ndarray:
        """Return the float32 ``(z, y, x)`` array of ``"ct"`` or a result component key."""
        import SimpleITK as sitk  # noqa: PLC0415

        ct = self._require_ct()
        if name == "ct":

            def build() -> np.ndarray:
                image = _image_3d(ct.cube_hu, phase)
                return sitk.GetArrayFromImage(image).astype(np.float32, copy=False)

        else:
            source = result_entry(self._get("result"), name)
            if source is None:
                raise KeyError(f"No volume quantity {name!r} in the result")

            def build() -> np.ndarray:
                array = _quantity_on_ct_grid(source, ct)
                return np.ascontiguousarray(array, dtype=np.float32)

        return self._cached(("volume", name, int(phase)), build)

    def ct_int16(self, phase: int = 0) -> Optional[np.ndarray]:
        """Return the CT as int16 when all its values are integral and fit, else ``None``."""
        values = self.volume_array("ct", phase)

        def build() -> Optional[np.ndarray]:
            info = np.iinfo(np.int16)
            finite = np.isfinite(values).all()
            if not finite or values.size == 0:
                return None
            if values.min() < info.min or values.max() > info.max:
                return None
            if not np.array_equal(values, np.round(values)):
                return None
            return values.astype(np.int16)

        return self._cached(("volume", "ct", int(phase), "int16"), build)

    def volume(self, name: str, phase: int = 0, encoding: str = "float32") -> dict[str, Any]:
        """Return a ``Volume`` (see ``web/PROTOCOL.md``) with its data as a numpy array.

        *encoding* ``"auto"`` sends the CT as int16 when that is lossless.
        """
        if encoding not in ("float32", "auto"):
            raise ValueError(f"Unknown volume encoding {encoding!r}; use 'float32' or 'auto'")
        data = self.volume_array(name, phase)
        payload = data
        if encoding == "auto" and name == "ct":
            compact = self.ct_int16(phase)
            if compact is not None:
                payload = compact
        unit = quantity_unit(name)
        if unit == "":
            unit = component_info(self._get("result"), name)["unit"]
        return {
            "name": name,
            "unit": unit,
            "data": payload,
            "grid": grid_info(self._get("ct")),
            "range": _finite_range(data),
        }

    def _voi(self, index: int) -> Any:
        cst = self._get("cst")
        if cst is None:
            raise LookupError("No structure set loaded")
        return cst.vois[index]

    def mask_array(self, index: int) -> np.ndarray:
        """Return the uint8 ``(z, y, x)`` mask of the structure at *index* (nominal), CT grid."""
        import SimpleITK as sitk  # noqa: PLC0415

        ct = self._require_ct()
        voi = self._voi(index)

        def build() -> np.ndarray:
            mask = voi._nominal_mask_3d()
            reference = _image_3d(ct.cube_hu)
            if mask.GetSize() != reference.GetSize():
                from pyRadPlan.core.resample import resample_image  # noqa: PLC0415

                mask = resample_image(
                    mask,
                    interpolator=sitk.sitkNearestNeighbor,
                    target_image=reference,
                    extrapolate=0,
                )
            return (sitk.GetArrayViewFromImage(mask) != 0).astype(np.uint8)

        return self._cached(("mask", index), build)

    def mask_info(self, index: int) -> dict[str, Any]:
        """``num_voxels`` and ``bbox`` (``[z0, y0, x0, z1, y1, x1]`` or ``None``) of a mask."""
        mask = self.mask_array(index)

        def build() -> dict[str, Any]:
            return {"num_voxels": int(np.count_nonzero(mask)), "bbox": _bbox(mask)}

        return self._cached(("mask_info", index), build)

    def structure_indices(self, refs: Optional[Sequence[Any]]) -> list[int]:
        """Positions of the structures *refs* (positions or unique names; all when ``None``)."""
        cst = self._get("cst")
        if cst is None:
            raise LookupError("No structure set loaded")
        vois = list(cst.vois)
        if refs is None:
            return list(range(len(vois)))
        return [resolve_structure(vois, ref) for ref in refs]

    def structure_masks(
        self, names: Optional[Sequence[Any]] = None, encoding: str = "uint8"
    ) -> dict[str, Any]:
        """Masks of the structures (all when *names* is omitted) on the CT grid.

        Encodings (``web/PROTOCOL.md``, ``view.structure_masks``): ``"uint8"`` one volume
        per structure (v0); ``"bits"`` up to 32 structures per bit plane; ``"crops"`` each
        mask cropped to its bounding box.
        """
        ct = self._require_ct()
        indices = self.structure_indices(names)
        vois = [self._voi(i) for i in indices]
        grid = grid_info(ct)
        if encoding == "uint8":
            masks = [
                {
                    "name": voi.name,
                    "index": i,
                    "color": list(voi.color),
                    "data": self.mask_array(i),
                }
                for i, voi in zip(indices, vois)
            ]
            return {"grid": grid, "masks": masks}
        if encoding == "crops":
            masks = []
            for i, voi in zip(indices, vois):
                info = self.mask_info(i)
                bbox = info["bbox"]
                data = None
                if bbox is not None:
                    z0, y0, x0, z1, y1, x1 = bbox
                    data = np.ascontiguousarray(self.mask_array(i)[z0:z1, y0:y1, x0:x1])
                masks.append(
                    {"name": voi.name, "index": i, "color": list(voi.color), **info, "data": data}
                )
            return {"grid": grid, "encoding": "crops", "masks": masks}
        if encoding == "bits":
            names_key = tuple(indices)

            def build() -> list[np.ndarray]:
                planes = []
                for start in range(0, len(vois), BITS_PER_PLANE):
                    chunk = indices[start : start + BITS_PER_PLANE]
                    dtype = _plane_dtype(len(chunk))
                    plane = np.zeros(grid["shape"], dtype=dtype)
                    for bit, i in enumerate(chunk):
                        plane |= self.mask_array(i).astype(dtype) << dtype.type(bit)
                    planes.append(plane)
                return planes

            planes = self._cached(("masks", "bits", names_key), build)
            masks = [
                {
                    "name": voi.name,
                    "index": index,
                    "color": list(voi.color),
                    "plane": n // BITS_PER_PLANE,
                    "bit": n % BITS_PER_PLANE,
                    **self.mask_info(index),
                }
                for n, (index, voi) in enumerate(zip(indices, vois))
            ]
            return {"grid": grid, "encoding": "bits", "planes": planes, "masks": masks}
        raise ValueError(f"Unknown mask encoding {encoding!r}; use 'uint8', 'bits' or 'crops'")

    def dvh_curves(
        self,
        quantity: str,
        indices: Sequence[int],
        num_points: int,
        range_mode: str = "shared",
    ) -> list[dict[str, Any]]:
        """Cumulative DVH curve and stats per structure (``None`` stats when empty).

        ``range_mode="shared"`` bins every curve on one axis from 0 to 105 % of the
        highest value inside any of the structures; ``"per_structure"`` bins each from 0
        to 105 % of its own maximum (:meth:`pyRadPlan.analysis.DVH.compute`, Qt analysis
        window).
        """
        if range_mode not in ("shared", "per_structure"):
            raise ValueError(f"Unknown DVH range {range_mode!r}; use 'shared' or 'per_structure'")

        def build() -> list[dict[str, Any]]:
            values = self.volume_array(quantity)
            masked = [values[self.mask_array(i).astype(bool)] for i in indices]
            shared_max: Optional[float] = None
            if range_mode == "shared":
                highest = max((float(v.max()) for v in masked if v.size), default=0.0)
                shared_max = highest * 1.05 if highest > 0 else 1.0
            curves = []
            for inside in masked:
                dose, volume = _cumulative_dvh(inside, int(num_points), shared_max)
                empty = inside.size == 0
                curves.append(
                    {
                        "dose": dose.astype(np.float32),
                        "volume": volume.astype(np.float32),
                        "stats": {
                            "mean": None if empty else float(inside.mean()),
                            "min": None if empty else float(inside.min()),
                            "max": None if empty else float(inside.max()),
                        },
                    }
                )
            return curves

        names = tuple(indices)
        if range_mode == "shared":
            key = ("dvh", quantity, names, int(num_points))
        else:
            key = ("dvh_curves", quantity, names, int(num_points), range_mode)
        return self._cached(key, build)

    def dvh(
        self,
        quantity: Optional[str] = None,
        names: Optional[Sequence[Any]] = None,
        num_points: int = 500,
        range_mode: str = "shared",
    ) -> dict[str, Any]:
        """Cumulative DVHs of the structures for a result quantity (v0 ``analysis.dvh``).

        All curves share one dose axis from 0 to 105 % of the highest value inside any
        of the structures, binned as :meth:`pyRadPlan.analysis.DVH.compute` bins, unless
        *range_mode* is ``"per_structure"``.
        """
        self._require_ct()
        result = self._get("result")
        keys = result_quantities(result)
        quantity = quantity or default_quantity(keys)
        if quantity is None or result_entry(result, quantity) is None:
            raise KeyError(f"No volume quantity {quantity!r} in the result")
        indices = self.structure_indices(names)
        vois = [self._voi(i) for i in indices]
        curves = self.dvh_curves(quantity, indices, num_points, range_mode)
        info = component_info(result, quantity)
        return {
            "quantity": quantity,
            "unit": quantity_unit(quantity) or info["unit"],
            "label": info["meta_label"],
            "unit_info": unit_info(info["unit"]),
            "curves": [
                {"name": voi.name, "index": i, "color": list(voi.color), **curve}
                for i, voi, curve in zip(indices, vois, curves)
            ],
        }


def unit_info(unit: str) -> dict[str, Any]:
    """``{symbol, dimensionless}`` of a display unit (``pyRadPlan.analysis._units.safe_unit``).

    ``symbol`` is what the Qt DVH markers append to a value: ``"%"`` for percent, ``""``
    for dimensionless units, else the pint symbol (``"Gy"`` for ``"Gy (RBE)"``).
    """
    from pyRadPlan.analysis._qi import format_unit_symbol  # noqa: PLC0415
    from pyRadPlan.analysis._units import safe_unit  # noqa: PLC0415

    resolved = safe_unit(unit)
    if str(resolved) == "percent":
        symbol = "%"
    elif resolved.dimensionless:
        symbol = ""
    else:
        symbol = format_unit_symbol(resolved)
    return {"symbol": symbol, "dimensionless": bool(resolved.dimensionless)}
