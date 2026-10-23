"""Tissue parameter lookups: how per-voxel tissue classes select from per-class kernels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import array_api_compat
import numpy as np

from pyRadPlan.core.xp_utils import device_cache_key

from pyRadPlan.biology.tissues._base import TissueProperties


class TissueParameterLookup(ABC):
    """
    Maps per-voxel tissue classes onto rows of per-tissue-class kernel arrays.

    Kernel-based and tabulated models carry depth-dependent data for a finite set of
    tissue classes, each identified by the parameters its model is tabulated for
    (reference LQ parameters for LEM-style data, domain/nucleus geometry for MKM). A
    lookup decides how a voxel of a given tissue class uses that data — by exact class
    match today, possibly by nearest class or interpolation in the future.

    The tissue classes are machine / table metadata and are kept as host arrays; they are
    fixed for the lifetime of the lookup.

    Parameters
    ----------
    tissue_classes : list[TissueProperties]
        The tissue classes of the patient, i.e. the ones the per-voxel tissue class index
        addresses. Their own parameters are read from here, see
        :meth:`gather_tissue_parameters`.
    tissue_classes_ix : array-like, shape (n_classes,)
        Index by which each of those classes is referenced in the per-voxel
        ``tissue_class_ix`` of the dose grid.
    kernel_rows : array-like of int, shape (n_classes,), optional
        Row of the per-tissue-class kernel arrays holding the base data of each of those
        classes. Patient tissues are ordered by the structure set, the kernel rows by the
        base data, so the two are related by the match a lookup is built from and must not
        be assumed identical. Defaults to the class order itself, for kernels that are
        tabulated in exactly the order of `tissue_classes`.

    Raises
    ------
    ValueError
        No tissue class is declared, or the declarations have different lengths.
    """

    def __init__(
        self,
        tissue_classes: list[TissueProperties],
        tissue_classes_ix: Any,
        kernel_rows: Any = None,
    ):
        self.tissue_classes = list(tissue_classes)
        self.tissue_classes_ix = np.reshape(np.asarray(tissue_classes_ix, dtype=int), (-1,))
        if not self.tissue_classes:
            raise ValueError("A tissue lookup needs at least one reference tissue class.")
        if len(self.tissue_classes) != self.tissue_classes_ix.size:
            raise ValueError(
                f"Tissue classes declare {len(self.tissue_classes)} reference classes but "
                f"{self.tissue_classes_ix.size} class indices."
            )

        if kernel_rows is None:
            kernel_rows = np.arange(len(self.tissue_classes))
        self.kernel_rows = np.reshape(np.asarray(kernel_rows, dtype=int), (-1,))
        if self.kernel_rows.size != len(self.tissue_classes):
            raise ValueError(
                f"Tissue classes declare {len(self.tissue_classes)} reference classes but "
                f"{self.kernel_rows.size} kernel rows."
            )

    @abstractmethod
    def validate(self, voxel_params: dict[str, Any]) -> None:
        """Raise ValueError if some voxel of the dose grid cannot be looked up."""

    @abstractmethod
    def gather(self, context: Mapping[str, Any], fields: list[str]) -> dict[str, Any]:
        """
        Select per-voxel kernel values for one bixel.

        Parameters
        ----------
        context : mapping
            Full biological evaluation context. It carries the per-voxel tissue parameters
            the lookup identifies its classes by (``tissue_class_ix`` for the exact lookup)
            and, under the names given by ``fields``, the interpolated kernel arrays of
            shape ``(n_kernel_rows, n_voxels)`` or ``(n_voxels,)``. The rows are the tissue
            classes of the base data, which :attr:`kernel_rows` maps the tissue classes of
            the patient onto.
        fields : list[str]
            Names of the kernel arrays to gather.

        Returns
        -------
        dict[str, Array]
            One ``(n_voxels,)`` array per field.
        """

    @abstractmethod
    def gather_tissue_parameters(
        self, context: Mapping[str, Any], fields: list[str]
    ) -> dict[str, Any]:
        """
        Expand parameters of the tissue classes themselves onto the voxels.

        Models whose kernels tabulate intermediate quantities need the parameters of a
        voxel's tissue class alongside them (MKM derives alpha/beta from its zs kernels
        together with the class' ``alpha_0`` / ``beta_0``). Mapping voxels onto classes is
        the lookup's job, so the expansion lives here rather than in the evaluator.

        Parameters
        ----------
        context : mapping
            Full biological evaluation context, as for :meth:`gather`.
        fields : list[str]
            Attribute names to read off each :class:`TissueProperties` class.

        Returns
        -------
        dict[str, Array]
            One ``(n_voxels,)`` array per field.
        """


class ExactClassLookup(TissueParameterLookup):
    """
    Lookup requiring every voxel to match one tissue class exactly.

    Voxels with tissue class index ``0`` (outside any structure) map to class 0.

    The class indices, kernel rows and tissue class parameters are converted into the array
    namespace, device and dtype of the per-voxel arrays once per combination instead of on
    every bixel, so a bixel only indexes arrays already on the device. Once
    :meth:`validate` has confirmed the whole dose grid, :meth:`gather` skips the per-bixel
    match check, since the bixel indices are a subset of the validated grid and the check
    would force a device synchronization per bixel.
    """

    def __init__(
        self,
        tissue_classes: list[TissueProperties],
        tissue_classes_ix: Any,
        kernel_rows: Any = None,
    ):
        super().__init__(tissue_classes, tissue_classes_ix, kernel_rows)
        self._reference_cache: dict[Any, Any] = {}
        self._parameter_cache: dict[Any, Any] = {}
        self._grid_validated = False

    def _reference(self, like: Any) -> tuple[Any, Any, Any]:
        """Namespace, class indices and kernel rows in ``like``'s namespace and dtype."""
        xp = array_api_compat.array_namespace(like)
        key = (xp, device_cache_key(like), like.dtype)
        reference = self._reference_cache.get(key)
        if reference is None:
            device = array_api_compat.device(like)
            reference = tuple(
                xp.asarray(values, dtype=like.dtype, device=device)
                for values in (self.tissue_classes_ix, self.kernel_rows)
            )
            self._reference_cache[key] = reference
        return (xp, *reference)

    def class_index(self, v_tissue_class_ix: Any, validate: bool = True) -> Any:
        """
        Index of the matching tissue class per voxel.

        Parameters
        ----------
        v_tissue_class_ix : Array, shape (n_voxels,) or (n_voxels, n_ct_scen)
            Tissue class index of every voxel; ``0`` marks a voxel outside any structure.
        validate : bool
            Check that every voxel matches a class. Only the per-bixel gather of an already
            validated dose grid may skip this.

        Raises
        ------
        ValueError
            If a voxel references a tissue class that is not among the matched classes.
        """
        xp, reference, _kernel_rows = self._reference(v_tissue_class_ix)

        # (..., n_classes) boolean match against every class
        matches = v_tissue_class_ix[..., None] == reference
        if validate:
            outside = v_tissue_class_ix == 0
            unmatched = ~xp.any(matches, axis=-1) & ~outside
            if xp.any(unmatched):
                first = tuple(int(b[0]) for b in xp.nonzero(unmatched))
                raise ValueError(
                    "No matching tissue class for tissue class index "
                    f"{int(v_tissue_class_ix[first])}. "
                    f"Available indices {self.tissue_classes_ix.tolist()} "
                    f"for classes: {self.tissue_classes}"
                )
        return xp.argmax(xp.astype(matches, xp.int64), axis=-1)

    def validate(self, voxel_params: dict[str, Any]) -> None:
        if voxel_params.get("tissue_class_ix") is None:
            raise ValueError(
                "A tissue class lookup needs the per-voxel 'tissue_class_ix', which the dose "
                "calculation did not provide."
            )
        self._grid_validated = False
        self.class_index(voxel_params["tissue_class_ix"])
        self._grid_validated = True

    def _voxel_class_index(self, context: Mapping[str, Any]) -> Any:
        """Tissue class of every voxel of the current bixel, as a row of `tissue_classes`."""
        return self.class_index(context["tissue_class_ix"], validate=not self._grid_validated)

    def _voxel_kernel_row(self, context: Mapping[str, Any], class_ix: Any) -> Any:
        """Kernel row holding the base data of every voxel's tissue class."""
        _xp, _classes, kernel_rows = self._reference(context["tissue_class_ix"])
        xp = array_api_compat.array_namespace(class_ix)
        return xp.take(kernel_rows, class_ix, axis=0)

    def gather(self, context: Mapping[str, Any], fields: list[str]) -> dict[str, Any]:
        class_ix = self._voxel_class_index(context)
        xp = array_api_compat.array_namespace(class_ix)
        voxel_ix = xp.arange(class_ix.shape[0], device=array_api_compat.device(class_ix))
        # A voxel's tissue class and the kernel row holding its base data are ordered
        # independently, see `kernel_rows`; a kernel not tabulated per class is used as-is.
        row_ix = self._voxel_kernel_row(context, class_ix)
        return {
            name: context[name][voxel_ix]
            if context[name].ndim == 1
            else context[name][row_ix, voxel_ix]
            for name in fields
        }

    def gather_tissue_parameters(
        self, context: Mapping[str, Any], fields: list[str]
    ) -> dict[str, Any]:
        if not fields:
            return {}

        class_ix = self._voxel_class_index(context)
        xp = array_api_compat.array_namespace(class_ix)
        dtype = context["alpha_x"].dtype
        return {
            name: xp.take(self._class_parameter(name, class_ix, dtype), class_ix, axis=0)
            for name in fields
        }

    def _class_parameter(self, name: str, like: Any, dtype: Any) -> Any:
        """Parameter ``name`` of every tissue class, on ``like``'s namespace and device."""
        xp = array_api_compat.array_namespace(like)
        key = (name, xp, device_cache_key(like), dtype)
        values = self._parameter_cache.get(key)
        if values is None:
            values = xp.asarray(
                [float(getattr(tissue, name)) for tissue in self.tissue_classes],
                dtype=dtype,
                device=array_api_compat.device(like),
            )
            self._parameter_cache[key] = values
        return values


TISSUE_LOOKUPS: dict[str, type[TissueParameterLookup]] = {"exact": ExactClassLookup}


def make_tissue_lookup(
    method: str,
    tissue_classes_machine: Any,
    tissue_classes_cst: Any,
    tissue_type: type[TissueProperties] = TissueProperties,
) -> TissueParameterLookup:
    """
    Instantiate a lookup by name (``"exact"``).

    Only the machine tissue classes of `tissue_type` are considered, together with plain
    :class:`TissueProperties` ones, which do not name a model. A machine can carry classes
    for several models (the generic oxygen machine has LEM and MKM ones) and the base data
    of one model must not be selected through the tissue class of another.

    Each CST tissue is then matched against those candidates on the fields identifying its
    own model (see :meth:`TissueProperties.matches`), so a LEM tissue is never matched to
    MKM base data and vice versa. Parameters a machine class leaves undeclared (NaN) are
    not compared, nor is the provenance label (MKM vs mMKM, the LEM version); of several
    matching candidates one of the tissue's own provenance is preferred, see
    :meth:`TissueProperties.find_match`.

    The candidates keep the order of the base data, which is the order its per-tissue-class
    kernels are tabulated in: a machine carrying classes of several models tabulates each
    model's kernels over that model's classes alone (the generic oxygen machine has one MKM
    and two LEM classes, and alpha/beta kernels with one row per LEM class). The lookup
    therefore addresses a kernel row by the position of the matched candidate, never by the
    position of the CST tissue, which the structure set orders independently.
    """
    try:
        cls = TISSUE_LOOKUPS[method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown tissue lookup '{method}'. Available: {sorted(TISSUE_LOOKUPS)}"
        ) from exc

    candidates = [
        machine_tp
        for machine_tp in tissue_classes_machine
        if isinstance(machine_tp, tissue_type) or type(machine_tp) is TissueProperties
    ]
    if not candidates:
        raise ValueError(
            f"The machine base data provides no {tissue_type.__name__} tissue classes, only "
            f"{[type(machine_tp).__name__ for machine_tp in tissue_classes_machine]}"
        )

    matched_pairs = TissueProperties.lookup(tissue_classes_cst, candidates, ignore_undeclared=True)
    unmatched = [cst_tp for cst_tp, machine_tp in matched_pairs if machine_tp is None]
    if unmatched:
        raise ValueError(
            f"No matching tissue class in the machine base data for CST tissue(s): "
            f"{unmatched}. Available classes: {candidates}"
        )

    # `lookup` reports one pair per CST tissue, in CST order, so the per-voxel tissue class
    # index (which addresses the CST tissues) indexes the pairs directly.
    tissue_classes_matched_cst = [cst_tp for cst_tp, _machine_tp in matched_pairs]
    kernel_rows = [
        next(row for row, candidate in enumerate(candidates) if candidate is machine_tp)
        for _cst_tp, machine_tp in matched_pairs
    ]
    return cls(tissue_classes_matched_cst, range(len(matched_pairs)), kernel_rows)
