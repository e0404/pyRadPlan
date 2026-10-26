"""Tabulated RBE models based on pre-computed lookup tables."""

import logging
import os
import warnings
from abc import abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Optional, Union

import numpy as np
from pymatreader import read_mat

from pyRadPlan.core import (
    add_base_data_file,
    base_data_generation,
    find_base_data,
    get_base_data_kind,
    list_base_data,
)
from pyRadPlan.biology.models._evaluator import BioModelEvaluator, TabulatedSpectrumEvaluator
from pyRadPlan.biology.models._kernel_sets import KernelSetInfo, TissueClassInfo
from pyRadPlan.biology.models._resolution import KernelSetSelectionError
from pyRadPlan.biology.tissues import TissueProperties, get_provenance
from .kernel_based_lq_model import resolved_lookup
from .lq_models import LQModel

logger = logging.getLogger(__name__)


#: Base data kinds of the tables, see :data:`pyRadPlan.core.BASE_DATA_KINDS`.
RBE_TABLE_KIND = "rbe_tables"
STOPPING_POWER_TABLE_KIND = "stopping_power_tables"


def _base_data_path(kind: str, name: str) -> Any:
    """Location of the base data file ``name`` of ``kind``; raise if there is none."""
    found = find_base_data(kind, name)
    if found is None:
        label = get_base_data_kind(kind).label
        raise FileNotFoundError(f"Could not find the {label} {name!r}.")
    return found.path


def _file_stamp(path: Any) -> Optional[tuple[int, int, int]]:
    try:
        stat = os.stat(str(path))
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size, base_data_generation()


def _table_meta(table_path: Any, root_key: str) -> dict[str, Any]:
    """Metadata of a quantity table (see :func:`_read_table_meta`), re-read once it changes."""
    return _read_table_meta(table_path, root_key, _file_stamp(table_path))


@lru_cache(maxsize=None)
def _read_table_meta(table_path: Any, root_key: str, _stamp: Any) -> dict[str, Any]:
    """
    Metadata of a quantity table: provenance, reference and tissue classes.

    ``meta.model`` (optionally with ``meta.version``) names the provenance and
    ``meta.modelParameters`` holds the tissue parameters the table was computed with, under
    the camelCase names :meth:`TissueProperties.from_dict` accepts as they are stored. The
    parameters are scalars for a table covering a single tissue class and equally long
    arrays for one tabulated per class; a scalar alongside arrays applies to every class.
    The metadata describes the tabulated values only; how they are combined (e.g. averaged
    over fluence spectra) is up to the model using the table, so a ``meta.formalism`` entry
    is ignored.

    Cached per file and modification stamp, since a tissue editor reads every available
    table each time it offers the classes.
    """
    meta = read_mat(table_path)[root_key]["meta"]
    model = str(meta.get("model", "Generic"))
    version = meta.get("version")
    provenance = get_provenance(model, version)
    columns = {
        name: np.reshape(np.asarray(value), -1)
        for name, value in meta.get("modelParameters", {}).items()
    }
    n_classes = max((column.size for column in columns.values()), default=1)
    extra = {"version": str(version)} if version is not None else {}
    tissues = tuple(
        TissueProperties.from_base_data(
            {name: column[i if column.size > 1 else 0].item() for name, column in columns.items()},
            model=model,
            **extra,
        )
        for i in range(n_classes)
    )
    return {
        "provenance": provenance.name,
        "reference": str(meta.get("reference", "") or ""),
        "tissue_classes": tissues,
    }


def _table_tissue_classes(table_path: Any, root_key: str) -> tuple[TissueProperties, ...]:
    """Tissue classes a quantity table is tabulated for, read from its metadata."""
    return _table_meta(table_path, root_key)["tissue_classes"]


class TabulatedRBEModel(LQModel):
    """
    Abstract base class for RBE models driven by pre-computed lookup tables.

    Two tables are required: a **quantity table** (e.g. alpha/beta vs. energy per fragment
    species, loaded by the concrete subclass via :meth:`load_quantity_table`) and a
    **stopping power table** (dE/dx vs. energy per fragment species, loaded by
    :meth:`load_sp_table`). The model itself only holds the tables and the pure
    dose-averaging math; the per-machine pre-computation over fluence spectra happens in
    :class:`~pyRadPlan.biology.models.TabulatedSpectrumEvaluator`.

    Tables are computed with NumPy: they are machine data, not per-bixel arrays.

    Parameters
    ----------
    sp_table_name : str
        File name of the stopping power table, shipped with pyRadPlan or added with
        :func:`add_stopping_power_table`.
    fragments_to_include : array-like of (A, Z) rows or None
        Fragment species to use. ``None`` includes every charged fragment present in the
        kernel spectra that both tables know.
    tissue_lookup : str
        How voxel ``(alpha_x, beta_x)`` pairs select a table tissue class (``"exact"``).

    Attributes
    ----------
    quantities_in_table : tuple[str, ...]
        Names of the quantity columns in the loaded table (e.g. ``("alpha", "beta")``).
    quantities_in_kernel : tuple[str, ...]
        Names of the dose-averaged arrays produced per kernel (e.g. ``("alpha", "sqrt_beta")``).
    quantity_transforms : dict[str, str | None]
        Element-wise transform applied to a table quantity before averaging
        (e.g. ``{"beta": "sqrt"}``).
    table_root_key : str
        Entry of the quantity table ``.mat`` holding its ``meta`` and ``data``.
    """

    possible_radiation_modes = ("protons", "helium", "carbon", "oxygen")
    kernel_set_kind = "rbe_table"
    quantities_in_table: tuple[str, ...] = ()
    quantities_in_kernel: tuple[str, ...] = ()
    quantity_transforms: dict[str, Optional[str]] = {}
    table_root_key: str = "RBEtable"

    def __init__(
        self,
        sp_table_name: str = "SPtable.mat",
        fragments_to_include: Any = None,
        tissue_lookup: str = "exact",
    ):
        self.sp_table_name = sp_table_name
        self.tissue_lookup = tissue_lookup
        self.fragments_to_include = (
            None if fragments_to_include is None else np.asarray(fragments_to_include, float)
        )
        self._sp_table = self.load_sp_table()
        self._q_table = self.load_quantity_table()

    def evaluator(self, machine: Any, voxel_params: dict[str, Any]) -> BioModelEvaluator:
        """Create the evaluator looking up the quantity table rows of each voxel's tissue."""
        lookup, _table = resolved_lookup(self, machine, voxel_params)
        return TabulatedSpectrumEvaluator(self, machine, lookup, voxel_params)

    def select_kernel_set(
        self, machine: Any, tissues: Optional[list[TissueProperties]] = None
    ) -> Optional[KernelSetInfo]:
        """
        Return the loaded quantity table as a kernel set of kind ``rbe_table``.

        The table is chosen by ``quantity_table_name`` (or ``data_model``) when the model is
        built, so the machine only has to provide the fluence spectra it is averaged over.

        Raises
        ------
        KernelSetSelectionError
            The table is of another provenance than the configured ``data_model``.
        """
        table = self.table_kernel_set
        data_model = getattr(self, "data_model", None)
        if data_model and get_provenance(data_model).name != table.provenance:
            raise KernelSetSelectionError(
                f"Biological model '{self.model}' is configured for data model "
                f"'{get_provenance(data_model).label}', but its quantity table "
                f"'{table.id}' is tabulated for {table.provenance_label}."
            )
        return table

    @property
    def table_kernel_set(self) -> KernelSetInfo:
        """The loaded quantity table as a kernel set (one tissue class per table row)."""
        return self.quantity_table_kernel_set(str(getattr(self, "quantity_table_name", "")))

    @classmethod
    def quantity_table_kernel_set(cls, quantity_table_name: str) -> KernelSetInfo:
        """
        Describe a quantity table as a kernel set of kind ``rbe_table``.

        Its provenance and reference come from the table's metadata.
        """
        meta = _table_meta(
            _base_data_path(RBE_TABLE_KIND, quantity_table_name), cls.table_root_key
        )
        return KernelSetInfo(
            id=quantity_table_name,
            kind="rbe_table",
            fields=tuple(cls.quantities_in_table),
            provenance=meta["provenance"],
            reference=meta["reference"],
            declared=True,
            tissue_classes=tuple(
                TissueClassInfo(tissue=t, kernel_set=quantity_table_name, column=i)
                for i, t in enumerate(meta["tissue_classes"])
            ),
        )

    @classmethod
    def quantity_table_kernel_sets(cls) -> list[KernelSetInfo]:
        """Every readable available quantity table, as kernel sets."""
        sets = []
        for table_name in cls.available_quantity_tables():
            try:
                sets.append(cls.quantity_table_kernel_set(table_name))
            except Exception:  # noqa: BLE001 - one broken table must not hide the others
                logger.warning(
                    "Could not read the metadata of quantity table '%s'",
                    table_name,
                    exc_info=True,
                )
        return sets

    # ------------------------------------------------------------------ tables
    @property
    def table_alpha_x(self) -> np.ndarray:
        """Reference alpha_x per tissue class of the quantity table, shape (n_classes,)."""
        return np.atleast_1d(np.asarray(self._q_table["alpha_x"], dtype=float))

    @property
    def table_beta_x(self) -> np.ndarray:
        """Reference beta_x per tissue class of the quantity table, shape (n_classes,)."""
        return np.atleast_1d(np.asarray(self._q_table["beta_x"], dtype=float))

    @property
    def table_tissue_classes(self) -> list[TissueProperties]:
        """
        Tissue classes the loaded quantity table is tabulated for.

        Unlike :attr:`table_alpha_x` / :attr:`table_beta_x`, these carry every tissue
        parameter the table was computed with (the LEM version, the cell geometry, ...),
        which is what a tissue of the patient has to reproduce to be matched to the table.
        """
        return list(self._q_table.get("tissue_classes", ()))

    @classmethod
    def available_quantity_tables(cls) -> list[str]:
        """
        File names of the available quantity tables, in alphabetical order.

        These are the RBE tables shipped with pyRadPlan and those the user added (see
        :func:`add_rbe_table`), the tables the model can be loaded with; a concrete model
        names the one it uses in its own table parameter (``quantity_table_name``).
        """
        return sorted((entry.name for entry in list_base_data(RBE_TABLE_KIND)), key=str.casefold)

    @classmethod
    def quantity_table_tissue_classes(cls, quantity_table_name: str) -> list[TissueProperties]:
        """Tissue classes a quantity table is tabulated for, read from its metadata."""
        return list(
            _table_tissue_classes(
                _base_data_path(RBE_TABLE_KIND, quantity_table_name), cls.table_root_key
            )
        )

    @classmethod
    def available_table_tissue_classes(cls) -> list[TissueProperties]:
        """
        Tissue classes of every available quantity table.

        These are the tissues the model can be run for at all, each by loading the table
        tabulating it (see :meth:`available_quantity_tables`). A class several tables
        cover is listed once; a table whose metadata cannot be read is skipped rather
        than keeping the others from being offered.
        """
        classes: list[TissueProperties] = []
        for table_name in cls.available_quantity_tables():
            try:
                tissues = cls.quantity_table_tissue_classes(table_name)
            except Exception:  # noqa: BLE001 - one broken table must not hide the others
                logger.warning(
                    "Could not read the tissue classes of quantity table '%s'",
                    table_name,
                    exc_info=True,
                )
                continue
            classes.extend(tissue for tissue in tissues if tissue not in classes)
        return classes

    def load_sp_table(self) -> dict:
        """Load the stopping power table (``energies`` in MeV/u, ``dE_dx`` in keV/um)."""
        data = read_mat(_base_data_path(STOPPING_POWER_TABLE_KIND, self.sp_table_name))
        sp_table = {}
        sp_table["fragments_AZ"] = np.column_stack(
            (data["SPtable"]["data"]["A"], data["SPtable"]["data"]["Z"])
        )
        sp_table["dE_dx"] = np.array(data["SPtable"]["data"]["dEdx"])
        sp_table["energies"] = np.array(data["SPtable"]["data"]["energies"])
        sp_table["units"] = {"energies": "MeV/nucleon", "dE_dx": "keV/um"}
        return sp_table

    @abstractmethod
    def load_quantity_table(self) -> dict:
        """
        Load the quantity table (``.mat``) into a dict.

        Contains ``fragments_AZ``, ``energies``, the quantity columns, the reference
        ``alpha_x`` / ``beta_x`` per tissue class and the ``tissue_classes`` the table is
        tabulated for (see :attr:`table_tissue_classes`).
        """

    # ---------------------------------------------------------------- fragments
    def select_fragments(self, fluence_spectrum: Any) -> dict[str, Any]:
        """
        Select the fragment species to include and map them onto both tables.

        Parameters
        ----------
        fluence_spectrum : ChargedBeamFragmentSpectrum
            Spectrum of one kernel; all kernels of a machine share the same species.

        Returns
        -------
        dict
            ``fragments_AZ`` (n, 2), ``kernel_ix``, ``sp_table_ix``, ``q_table_ix`` — index
            lists of the selected fragments into the kernel spectrum and both tables.
        """
        # Local import: pyRadPlan.machines imports pyRadPlan.plan, which imports this package.
        from pyRadPlan.machines.particles._beam_fragment_spectrum import (  # noqa: PLC0415
            match_fragment_species,
        )

        charged = [(i, f) for i, f in enumerate(fluence_spectrum.fragments) if f.Z > 0]
        available = np.asarray([[f.A, f.Z] for _, f in charged], dtype=float).reshape(-1, 2)

        if self.fragments_to_include is None:
            requested = available
        else:
            requested = np.asarray(self.fragments_to_include, dtype=float).reshape(-1, 2)

        selected = {"fragments_AZ": [], "kernel_ix": [], "sp_table_ix": [], "q_table_ix": []}
        for fragment in requested:
            # A == NaN aggregates all isotopes of a charge and never compares equal, so the
            # spectrum entries are matched by species rather than by array equality. Both
            # tables are tabulated per charge, hence the Z-only lookup.
            in_kernel = match_fragment_species(available, fragment)
            idx_sp = np.flatnonzero(self._sp_table["fragments_AZ"][:, 1] == fragment[1])
            idx_q = np.flatnonzero(self._q_table["fragments_AZ"][:, 1] == fragment[1])
            if in_kernel.size == 0 or idx_sp.size == 0 or idx_q.size == 0:
                warnings.warn(
                    f"Fragment (A, Z)={fragment.tolist()} is not present in the kernel spectra, "
                    f"the SP table or the quantity table; skipping it."
                )
                continue
            selected["fragments_AZ"].append(fragment)
            selected["kernel_ix"].append(charged[in_kernel[0]][0])
            selected["sp_table_ix"].append(int(idx_sp[0]))
            selected["q_table_ix"].append(int(idx_q[0]))

        if not selected["kernel_ix"]:
            raise ValueError("No fragment of the kernel spectra is covered by the RBE tables.")
        selected["fragments_AZ"] = np.asarray(selected["fragments_AZ"], dtype=float)
        if not np.array_equal(
            self._sp_table["fragments_AZ"][selected["sp_table_ix"], 1],
            self._q_table["fragments_AZ"][selected["q_table_ix"], 1],
        ):
            raise ValueError(
                "Selected fragments are not consistent between the SP table and the quantity table."
            )
        return selected

    # ------------------------------------------------------------ dose average
    def _table_quantity(self, name: str) -> np.ndarray:
        """Quantity table column as ``(n_classes, n_fragments, n_energies)``."""
        values = np.asarray(self._q_table[name], dtype=float)
        if values.ndim == 2:  # single tissue class
            values = values[None, :, :]
        return values

    def interpolate_in_energies(
        self, energies: np.ndarray, fragments: dict[str, Any]
    ) -> tuple[dict, dict]:
        """
        Interpolate both tables onto per-fragment energy grids.

        Parameters
        ----------
        energies : ndarray, shape (n_fragments, n_energies)
            Target energies per selected fragment (MeV/u).

        Returns
        -------
        (q_table, sp_table)
            ``q_table[qty]`` of shape (n_classes, n_fragments, n_energies);
            ``sp_table["dE_dx"]`` of shape (n_fragments, n_energies).
        """
        sp_interp = np.zeros(energies.shape)
        for i, ix in enumerate(fragments["sp_table_ix"]):
            sp_interp[i] = np.interp(
                energies[i], self._sp_table["energies"][ix], self._sp_table["dE_dx"][ix]
            )
        sp_table = {"dE_dx": sp_interp, "energies": energies}

        q_table = {"energies": energies}
        for qty in self.quantities_in_table:
            table_values = self._table_quantity(qty)
            qty_interp = np.zeros((table_values.shape[0],) + energies.shape)
            for c in range(table_values.shape[0]):
                for i, ix in enumerate(fragments["q_table_ix"]):
                    qty_interp[c, i] = np.interp(
                        energies[i], self._q_table["energies"][ix], table_values[c, ix]
                    )
            q_table[qty] = qty_interp
        return q_table, sp_table

    def get_quantity(self, q_table: dict, qty: str) -> np.ndarray:
        """Read a quantity from an interpolated table, applying its configured transform."""
        transform = self.quantity_transforms.get(qty)
        raw = q_table[qty]
        return getattr(np, transform)(raw) if transform else raw

    def dose_average(self, kernel: Any, fragments: dict[str, Any]) -> dict[str, np.ndarray]:
        """
        Dose-averaged kernel quantities for one pencil-beam kernel.

        Parameters
        ----------
        kernel : ParticlePencilBeamKernel
            Kernel with a fragment fluence spectrum.
        fragments : dict
            Output of :meth:`select_fragments`.

        Returns
        -------
        dict[str, ndarray]
            One ``(n_classes, n_depths)`` array per entry of ``quantities_in_kernel``.
        """
        spectrum = kernel.fluence_spectrum.fragments
        energies = np.stack([spectrum[i].energy for i in fragments["kernel_ix"]])
        q_table, sp_table = self.interpolate_in_energies(energies, fragments)

        # (n_fragments, n_energies, 1) against fluence (n_fragments, n_energies, n_depths)
        sp = sp_table["dE_dx"][:, :, None]
        fluence = np.stack([spectrum[i].fluence_spectrum for i in fragments["kernel_ix"]])
        n_classes = self.table_alpha_x.shape[0]

        denominator = np.sum(sp * fluence, axis=(0, 1))  # (n_depths,)
        valid = denominator > 0
        safe_denominator = np.where(valid, denominator, 1.0)

        result = {}
        for qty_kernel, qty_table in zip(self.quantities_in_kernel, self.quantities_in_table):
            values = self.get_quantity(q_table, qty_table)  # (n_cls, n_frag, n_e)
            if values.shape[0] not in (1, n_classes):
                raise ValueError(
                    f"Quantity table '{qty_table}' has {values.shape[0]} tissue classes, "
                    f"expected 1 or {n_classes}."
                )
            numerator = np.sum(values[:, :, :, None] * sp * fluence, axis=(1, 2))
            averaged = np.where(valid, numerator / safe_denominator, 0.0)
            result[qty_kernel] = np.broadcast_to(averaged, (n_classes, len(kernel.depths))).copy()
        return result


class TabulatedAlphaBetaModel(TabulatedRBEModel):
    """
    Tabulated LQ model using dose-averaged alpha and sqrt(beta) kernels.

    Looks up alpha and beta per fragment and energy from an RBE table, dose-averages them
    over the pencil-beam fluence spectrum (beta as sqrt(beta), squared back per voxel).

    Parameters
    ----------
    quantity_table_name : str
        File name of the RBE table, one of
        :meth:`~TabulatedRBEModel.available_quantity_tables`.
    data_model : str, optional
        Provenance the table must be tabulated for (e.g. ``"LEM_I"``). Without an explicit
        ``quantity_table_name``, the only table of that provenance is chosen if the default
        table is of another one.
    """

    model = "alpha_beta_tables"
    name = "α/β tables + fluence spectra"
    description = (
        "LQ model with per-ion α/√β tables averaged over the fragment fluence spectra "
        "(Zaider–Rossi)."
    )
    model_aliases = ("dose_average_alpha_beta",)
    matrad_name = "dose_average_alpha_beta"
    DEFAULT_QUANTITY_TABLE: ClassVar[str] = "RBEtable_LEMI_Scholz06_AX01_BX005.mat"
    quantities_in_table = ("alpha", "beta")
    quantities_in_kernel = ("alpha", "sqrt_beta")
    required_quantities = ("fluence",)
    quantity_transforms = {"alpha": None, "beta": "sqrt"}

    def __init__(
        self,
        quantity_table_name: str = "RBEtable_LEMI_Scholz06_AX01_BX005.mat",
        sp_table_name: str = "SPtable.mat",
        fragments_to_include: Any = None,
        tissue_lookup: str = "exact",
        data_model: Optional[str] = None,
    ):
        self.data_model = data_model
        if data_model and "quantity_table_name" not in self._parameters:
            quantity_table_name = self._table_for_data_model(data_model, quantity_table_name)
        self.quantity_table_name = quantity_table_name
        super().__init__(
            sp_table_name=sp_table_name,
            fragments_to_include=fragments_to_include,
            tissue_lookup=tissue_lookup,
        )

    @classmethod
    def _table_for_data_model(cls, data_model: str, default: str) -> str:
        """Choose the default table if it fits ``data_model``, else the only one that does."""
        provenance = get_provenance(data_model).name
        fitting = [s.id for s in cls.quantity_table_kernel_sets() if s.provenance == provenance]
        if default in fitting or len(fitting) != 1:
            return default
        return fitting[0]

    def load_quantity_table(self) -> dict:
        """Load the alpha/beta quantity table."""
        data = read_mat(_base_data_path(RBE_TABLE_KIND, self.quantity_table_name))
        quantity_table = {}
        quantity_table["tissue_classes"] = self.quantity_table_tissue_classes(
            self.quantity_table_name
        )
        quantity_table["alpha_x"] = data["RBEtable"]["meta"]["modelParameters"]["alphaX"]
        quantity_table["beta_x"] = data["RBEtable"]["meta"]["modelParameters"]["betaX"]
        quantity_table["fragments_AZ"] = np.column_stack(
            (data["RBEtable"]["data"]["A"], data["RBEtable"]["data"]["Z"])
        )
        quantity_table["energies"] = np.array(data["RBEtable"]["data"]["energies"])
        quantity_table["alpha"] = np.array(data["RBEtable"]["data"]["alpha"])
        quantity_table["beta"] = np.array(data["RBEtable"]["data"]["beta"])
        quantity_table["units"] = {"energies": "MeV/nucleon", "alpha": "1/Gy", "beta": "1/Gy^2"}
        return quantity_table

    def alpha_beta_from_kernel_rows(self, rows: dict[str, Any]) -> tuple[Any, Any]:
        """Return the ``(alpha, beta)`` kernel quantities from the looked-up kernel rows."""
        return rows["alpha"], rows["sqrt_beta"] ** 2


def _check_table_data(filename: Union[os.PathLike, str], root_key: str, fields) -> None:
    """Raise ``ValueError`` unless the file holds ``<root_key>.data`` with ``fields``."""
    try:
        content = read_mat(str(filename))
    except Exception as exc:
        raise ValueError(f"Could not read {filename}: {exc}") from exc
    entry = content.get(root_key) if isinstance(content, dict) else None
    data = entry.get("data") if isinstance(entry, dict) else None
    if not isinstance(data, dict):
        raise ValueError(f"{filename} holds no '{root_key}.data' entry.")
    missing = [field for field in fields if field not in data]
    if missing:
        raise ValueError(f"'{root_key}.data' of {filename} lacks {', '.join(missing)}.")


def add_rbe_table(
    filename: Union[os.PathLike, str],
    name: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Add an RBE table to the tables available by name.

    The file is checked to be an α/β table :class:`TabulatedAlphaBetaModel` can load, with
    readable metadata, and copied into the user's RBE table folder
    (:func:`pyRadPlan.core.user_base_data_dir` of ``"rbe_tables"``). The table is then
    selectable as ``quantity_table_name`` and offered by the GUI's biological models.

    Parameters
    ----------
    filename : os.PathLike or str
        RBE table to add.
    name : str, optional
        File name to add it under. Defaults to the name of ``filename``.
    overwrite : bool, optional, default=False
        Replace a table of the same name previously added by the user.

    Returns
    -------
    Path
        The added file.

    Raises
    ------
    ValueError
        If the file is no readable α/β table.
    FileExistsError
        If a table of that name exists already (see
        :func:`pyRadPlan.core.add_base_data_file`).
    """
    model = TabulatedAlphaBetaModel
    _check_table_data(
        filename, model.table_root_key, ("A", "Z", "energies", *model.quantities_in_table)
    )
    try:
        _read_table_meta(str(filename), model.table_root_key, None)
    except Exception as exc:
        raise ValueError(f"Could not read the metadata of {filename}: {exc}") from exc
    return add_base_data_file(RBE_TABLE_KIND, filename, name=name, overwrite=overwrite)


def add_stopping_power_table(
    filename: Union[os.PathLike, str],
    name: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Add a stopping power table to the tables available by name.

    The file is checked to hold an ``SPtable`` with ``A``, ``Z``, ``energies`` and
    ``dEdx`` and copied into the user's stopping power table folder; it is then
    selectable as ``sp_table_name`` of the tabulated RBE models.

    Parameters
    ----------
    filename : os.PathLike or str
        Stopping power table to add.
    name : str, optional
        File name to add it under. Defaults to the name of ``filename``.
    overwrite : bool, optional, default=False
        Replace a table of the same name previously added by the user.

    Returns
    -------
    Path
        The added file.
    """
    _check_table_data(filename, "SPtable", ("A", "Z", "energies", "dEdx"))
    return add_base_data_file(STOPPING_POWER_TABLE_KIND, filename, name=name, overwrite=overwrite)
