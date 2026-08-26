"""Tabulated RBE models based on pre-computed lookup tables."""

import sys
import warnings
from abc import abstractmethod
from typing import Any, Optional

import numpy as np
from pymatreader import read_mat

from pyRadPlan.bio_models._evaluator import BioModelEvaluator, TabulatedSpectrumEvaluator
from pyRadPlan.bio_models._tissue_lookup import make_tissue_lookup
from .lq_models import LQModel

if sys.version_info < (3, 10):
    import importlib_resources as resources  # Backport for older versions
else:
    from importlib import resources


class TabulatedRBEModel(LQModel):
    """
    Abstract base class for RBE models driven by pre-computed lookup tables.

    Two tables are required: a **quantity table** (e.g. alpha/beta vs. energy per fragment
    species, loaded by the concrete subclass via :meth:`load_quantity_table`) and a
    **stopping power table** (dE/dx vs. energy per fragment species, loaded by
    :meth:`load_sp_table`). The model itself only holds the tables and the pure
    dose-averaging math; the per-machine pre-computation over fluence spectra happens in
    :class:`~pyRadPlan.bio_models.TabulatedSpectrumEvaluator`.

    Tables are computed with NumPy: they are machine data, not per-bixel arrays.

    Parameters
    ----------
    sp_table_name : str
        Filename of the stopping power table inside ``pyRadPlan.data.SPtables``.
    fragments_to_include : array-like of (A, Z) rows or None
        Fragment species to use. ``None`` includes every charged fragment present in the
        kernel spectra that both tables know.
    tissue_lookup : str
        How voxel ``(alpha_x, beta_x)`` pairs select a table tissue class (``"exact"``).

    Attributes
    ----------
    quantities_in_table : list[str]
        Names of the quantity columns in the loaded table (e.g. ``["alpha", "beta"]``).
    quantities_in_kernel : list[str]
        Names of the dose-averaged arrays produced per kernel (e.g. ``["alpha", "sqrt_beta"]``).
    quantity_transforms : dict[str, str | None]
        Element-wise transform applied to a table quantity before averaging
        (e.g. ``{"beta": "sqrt"}``).
    """

    possible_radiation_modes = ["protons", "helium", "carbon", "oxygen"]
    quantities_in_table: list[str] = []
    quantities_in_kernel: list[str] = []
    quantity_transforms: dict[str, Optional[str]] = {}

    _folder_name_quantity_tables = resources.files("pyRadPlan.data.RBEtables")
    _folder_name_sp_tables = resources.files("pyRadPlan.data.SPtables")

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
        lookup = make_tissue_lookup(self.tissue_lookup, self.table_alpha_x, self.table_beta_x)
        return TabulatedSpectrumEvaluator(self, machine, lookup, voxel_params)

    # ------------------------------------------------------------------ tables
    @property
    def table_alpha_x(self) -> np.ndarray:
        """Reference alpha_x per tissue class of the quantity table, shape (n_classes,)."""
        return np.atleast_1d(np.asarray(self._q_table["alpha_x"], dtype=float))

    @property
    def table_beta_x(self) -> np.ndarray:
        """Reference beta_x per tissue class of the quantity table, shape (n_classes,)."""
        return np.atleast_1d(np.asarray(self._q_table["beta_x"], dtype=float))

    def load_sp_table(self) -> dict:
        """Load the stopping power table (``energies`` in MeV/u, ``dE_dx`` in keV/um)."""
        data = read_mat(self._folder_name_sp_tables / self.sp_table_name)
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
        """Load the quantity table (``.mat``) into a dict with ``fragments_AZ``, ``energies``,
        the quantity columns and the reference ``alpha_x`` / ``beta_x`` per tissue class.
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
        charged = [(i, f) for i, f in enumerate(fluence_spectrum.fragments) if f.Z > 0]
        available = np.asarray([[f.A, f.Z] for _, f in charged], dtype=float).reshape(-1, 2)

        if self.fragments_to_include is None:
            requested = available
        else:
            requested = np.asarray(self.fragments_to_include, dtype=float).reshape(-1, 2)

        selected = {"fragments_AZ": [], "kernel_ix": [], "sp_table_ix": [], "q_table_ix": []}
        for fragment in requested:
            in_kernel = np.flatnonzero((available == fragment).all(axis=1))
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
        Filename of the RBE table inside ``pyRadPlan.data.RBEtables``.
    """

    model = "dose_average_alpha_beta"
    quantities_in_table = ["alpha", "beta"]
    quantities_in_kernel = ["alpha", "sqrt_beta"]
    required_quantities = ["fluence"]
    quantity_transforms = {"alpha": None, "beta": "sqrt"}

    def __init__(
        self,
        quantity_table_name: str = "RBEtable_LEMI_Scholz06_AX01_BX005.mat",
        sp_table_name: str = "SPtable.mat",
        fragments_to_include: Any = None,
        tissue_lookup: str = "exact",
    ):
        self.quantity_table_name = quantity_table_name
        super().__init__(
            sp_table_name=sp_table_name,
            fragments_to_include=fragments_to_include,
            tissue_lookup=tissue_lookup,
        )

    def load_quantity_table(self) -> dict:
        data = read_mat(self._folder_name_quantity_tables / self.quantity_table_name)
        quantity_table = {}
        quantity_table["alpha_x"] = data["RBEtable"]["meta"]["modelParameters"]["alphaX"]
        quantity_table["beta_x"] = data["RBEtable"]["meta"]["modelParameters"]["betaX"]
        quantity_table["r_nucleus"] = data["RBEtable"]["meta"]["modelParameters"]["rNucleus"]
        quantity_table["fragments_AZ"] = np.column_stack(
            (data["RBEtable"]["data"]["A"], data["RBEtable"]["data"]["Z"])
        )
        quantity_table["energies"] = np.array(data["RBEtable"]["data"]["energies"])
        quantity_table["alpha"] = np.array(data["RBEtable"]["data"]["alpha"])
        quantity_table["beta"] = np.array(data["RBEtable"]["data"]["beta"])
        quantity_table["units"] = {"energies": "MeV/nucleon", "alpha": "1/Gy", "beta": "1/Gy^2"}
        return quantity_table

    def alpha_beta_from_kernel_rows(self, rows: dict[str, Any]) -> tuple[Any, Any]:
        return rows["alpha"], rows["sqrt_beta"] ** 2
