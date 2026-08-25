"""Tabulated RBE models based on pre-computed lookup tables."""

import warnings

import array_api_compat
import numpy as np
from pymatreader import read_mat
from typing import Any
from abc import abstractmethod
from ...core.xp_utils.typing import Array
from ...core.xp_utils.compat import interp1d as array_interp

import sys

if sys.version_info < (3, 10):
    import importlib_resources as resources  # Backport for older versions
else:
    from importlib import resources  # Standard from Python 3.9+

from .lq_models import LQModel


class TabulatedRBEModel(LQModel):
    """
    Abstract base class for RBE models driven by pre-computed lookup tables.

    This family of models interpolates alpha and beta (or related quantities) from tabulated
    data stored in mat files bundled with pyRadPlan. Two tables are
    always required:

    - A **quantity table** (e.g. alpha/beta vs. energy per fragment species),
      loaded by the concrete subclass via `load_quantity_table`.
    - A **stopping power table** (dE/dx vs. energy per fragment species),
      loaded by `load_sp_table` from ``pyRadPlan.data.SPtables``.

    Fragment species present in the quantity table must be a subset of those
    in the stopping power table; a :exc:`ValueError` is raised at construction
    if this constraint is violated.

    Class Attributes
    ----------------
    possible_radiation_modes : list[str]
        ``["protons", "helium", "carbon"]``
    fragments_to_include : list or None
        Fragment (A, Z) pairs to use. ``None`` or ``"all"`` includes every
        fragment in the quantity table.
    fragments_q_table_ix : list[int] or None
        Row indices into the quantity table for the selected fragments.
        Populated by load_fragments.
    fragments_sp_table_ix : list[int] or None
        Row indices into the stopping power table corresponding to each
        selected fragment. Populated by load_fragments`.
    fragments_kernel_ix : list[int] or None
        Indices into the kernel fluence spectrum fragments. Populated by
        set_kernel_fragments` at dose-calculation time.
    quantities_in_table : list[str] or None
        Names of quantity columns present in the loaded quantity table
        (e.g. ``["alpha", "beta"]``). Set by the concrete subclass.
    required_quantities : list[str] or None
        Quantity names that must be supplied by the machine file / dose engine.
        Overrides the parent class attribute; set by the concrete subclass.
    quantity_transforms : dict[str, str] or None
        Optional element-wise array transforms to apply when reading a
        quantity from the table (e.g. ``{"beta": "sqrt"}`` applies
        ``xp.sqrt`` before returning). ``None`` means no transform.

    Parameters
    ----------
    sp_table_name : str
        Filename of the stopping power table inside ``pyRadPlan.data.SPtables``.
        Defaults to ``"SPtable.mat"``. Can be overridden by subclasses.

    """

    required_quantities = [
        "physical_dose",
    ]  # Requires physical dose and LET information
    possible_radiation_modes = ["protons", "helium", "carbon", "oxygen"]
    fragments_to_include = None  # To be set based on the specific model and tables used
    fragments_q_table_ix = None  # To be set based on the specific model and tables used
    fragments_sp_table_ix = None  # To be set based on the specific model and tables used
    fragments_kernel_ix = None  # To be set based on the specific model and tables used

    quantities_in_table = None
    required_quantities = None  # input in machine file
    quantity_transforms = None

    _folder_name_quantity_tables = resources.files(
        "pyRadPlan.data.RBEtables"
    )  # Folder where RBE tables are stored
    _folder_name_sp_tables = resources.files(
        "pyRadPlan.data.SPtables"
    )  # Folder where stopping power tables are stored

    _sp_table = None
    _q_table = None

    def __init__(self):
        super().__init__()
        # should be overwritten with pln params, in genereall some parameters are coming that need to be overwritten
        self.sp_table_name = "SPtable.mat"
        self._sp_table = self.load_sp_table()
        self._q_table = self.load_quantity_table()  # could maybe also be a z* table or so

    def load_fragments(self, kernel):
        """
        Validate fragment consistency and build index mappings between tables.

        """
        if self.fragments_to_include is None or self.fragments_to_include == "all":
            self.fragments_to_include = np.array(
                [
                    [
                        k.A for k in kernel.fluence_spectrum.fragments if k.Z > 0
                    ],  # electrons are -1
                    [k.Z for k in kernel.fluence_spectrum.fragments if k.Z > 0],
                ]
            ).T
            self.fragments_kernel_ix = list(range(self.fragments_to_include.shape[0]))
        else:  # given as A,Z values in matrix
            all_fragments = np.array(
                [
                    [k.A for k in kernel.fluence_spectrum.fragments if k.Z > 0],
                    [k.Z for k in kernel.fluence_spectrum.fragments if k.Z > 0],
                ]
            ).T
            self.fragments_kernel_ix = np.where(
                (all_fragments[:, None] == self.fragments_to_include).all(axis=2).any(axis=1)
            )[0]
        self.fragments_sp_table_ix = []
        self.fragments_q_table_ix = []
        keep = []
        for i, fragment in enumerate(self.fragments_to_include):
            idx_sp = np.where(self._sp_table["fragments_AZ"][:, 1] == fragment[1])[0]
            idx_q = np.where(self._q_table["fragments_AZ"][:, 1] == fragment[1])[0]
            if idx_sp.shape[0] == 0 or idx_q.shape[0] == 0:
                warnings.warn(
                    f"Fragment {fragment} is not present in the SP or quantity table "
                    f"(SP fragments: {self._sp_table['fragments_AZ'].tolist()}); skipping it."
                )
                continue
            keep.append(i)
            self.fragments_sp_table_ix.append(int(idx_sp[0]))
            self.fragments_q_table_ix.append(int(idx_q[0]))
        self.fragments_to_include = self.fragments_to_include[keep]
        self.fragments_kernel_ix = [self.fragments_kernel_ix[i] for i in keep]
        # Check that all selected fragments are present in both tables
        if not np.array_equal(
            self._sp_table["fragments_AZ"][self.fragments_sp_table_ix, :],
            self._q_table["fragments_AZ"][self.fragments_q_table_ix, :],
        ):
            raise ValueError(
                "Selected fragments are not consistent between the SP table and the quantity table."
            )

    def load_sp_table(self) -> dict:
        """
        Load and parse the stopping power table from disk.

        Reads the ``.mat`` file at ``pyRadPlan.data.SPtables/<sp_table_name>``
        and returns a normalised dict with consistent key names and explicit
        unit annotations.

        Returns
        -------
            dict: A dictionary containing the stopping power table data.
        """
        sp_table_path = self._folder_name_sp_tables / self.sp_table_name
        data = read_mat(sp_table_path)
        sp_table = {}
        sp_table["fragments_AZ"] = np.column_stack(
            (data["SPtable"]["data"]["A"], data["SPtable"]["data"]["Z"])
        )
        sp_table["dE_dx"] = np.array(data["SPtable"]["data"]["dEdx"])
        sp_table["energies"] = np.array(data["SPtable"]["data"]["energies"])
        sp_table["units"] = {}
        sp_table["units"]["energies"] = "MeV/nucleon"
        sp_table["units"]["dE_dx"] = "keV/um"
        return sp_table

    def get_tissue_information(self, _, v_alpha_x: Any, v_beta_x: Any) -> Any:
        """
        Build a per-voxel tissue-index array by matching reference alpha/beta pairs.

        TODO: can this be generalized with the kernel based model? It is currently duplicated in both models, but it is not specific to either of them.
        ALso here more checks for the nucelous or other tissue parameters can be added in the future if needed.
        """
        return self.match_tissue_classes(
            v_alpha_x, v_beta_x, self.table_alpha_x, self.table_beta_x
        )

    @property
    def table_alpha_x(self) -> np.ndarray:
        """Reference alpha_x per tissue class of the loaded quantity table, shape (n_classes,)."""
        return np.atleast_1d(np.asarray(self._q_table["alpha_x"], dtype=float))

    @property
    def table_beta_x(self) -> np.ndarray:
        """Reference beta_x per tissue class of the loaded quantity table, shape (n_classes,)."""
        return np.atleast_1d(np.asarray(self._q_table["beta_x"], dtype=float))

    def get_quantity(self, q_table: dict, qty: str, xp) -> Array:
        """
        Read a single quantity row from the quantity table.

        Hereby any configured transform (e.g. sqrt) is applied before returning the value.
        """
        raw = q_table[qty]
        transform = self.quantity_transforms.get(qty)
        return getattr(xp, transform)(raw) if transform else raw

    def interpolate_in_energies(self, kernel: dict) -> tuple[dict, dict]:
        """
        Interpolate the quantity and stopping power tables onto the kernel's
        energy grid.
        """
        xp = array_api_compat.array_namespace(kernel.fluence_spectrum.fragments[0].energy)
        energies = xp.stack(
            [kernel.fluence_spectrum.fragments[f_ix].energy for f_ix in self.fragments_kernel_ix]
        )  # energies for which the kernel table has values
        # interppolate dEdx
        sp_table = {}
        sp_interp = xp.full(energies.shape, 0.0)
        new_energies = xp.full(energies.shape, 0.0)
        for i, ix in enumerate(self.fragments_sp_table_ix):
            sp_interp[i, :] = array_interp(
                energies[i, :],
                self._sp_table["energies"][ix, :],
                self._sp_table["dE_dx"][ix, :],
            )
            new_energies[i, :] = energies[i, :]
        sp_table["dE_dx"] = sp_interp
        sp_table["energies"] = new_energies
        q_table = {}
        # interpolate quantity table values
        for qty in self.quantities_in_table:
            qty_interp = xp.full(energies.shape, 0.0)
            new_energies = xp.full(energies.shape, 0.0)
            for i, ix in enumerate(self.fragments_q_table_ix):
                qty_interp[i, :] = array_interp(
                    energies[i, :],
                    self._q_table["energies"][ix, :],
                    self._q_table[qty][ix, :],
                )
                new_energies[i, :] = energies[i, :]
            q_table[qty] = qty_interp
        q_table["energies"] = new_energies
        return q_table, sp_table

    def compute_kernel_quantities(self, kernel: dict, v_tissue_index: Any) -> dict:
        """
        Compute dose-averaged biological quantities for a pencil-beam kernel.
        """
        [q_table, sp_table] = self.interpolate_in_energies(kernel)
        xp = array_api_compat.array_namespace(kernel.depths)

        n_depths = len(kernel.depths)
        n_tissue = self.table_alpha_x.shape[0]
        # Stack energy arrays
        kernel_spectra_energies = xp.stack(
            [kernel.fluence_spectrum.fragments[f_ix].energy for f_ix in self.fragments_kernel_ix]
        )  # (n_fragments, n_energies)

        q_energies = q_table["energies"][:]
        sp_energies = sp_table["energies"][:]

        if not xp.all(q_energies == sp_energies):
            raise ValueError(
                f"Energies in Quantity table and SP table do not match. "
                f"Q energies: {q_energies}, SP energies: {sp_energies}."
            )
        if not xp.all(kernel_spectra_energies == q_energies):
            raise ValueError(
                f"Energies in kernel spectra and Quantity table do not match. "
                f"Kernel spectra energies: {kernel_spectra_energies}, Quantity energies: {q_energies}."
            )

        sp = sp_table["dE_dx"][:]
        quantitys = {}
        for iqty, qty in enumerate(self.quantities_in_kernel):
            quantitys[qty] = self.get_quantity(q_table, self.quantities_in_table[iqty], xp)

        # (n_fragments, n_energies, 1) for broadcasting against fluence (n_energies, n_depths)
        sp = sp[:, :, xp.newaxis]
        for qty in self.quantities_in_kernel:
            quantitys[qty] = quantitys[qty][:, :, xp.newaxis]

        # fluence_spectra: (n_fragments, n_energies, n_depths)
        fluence_spectra = xp.stack(
            [
                kernel.fluence_spectrum.fragments[f].fluence_spectrum
                for f in self.fragments_kernel_ix
            ]
        )
        # Weighted sums over energy axis → (n_fragments, n_depths)
        dose_per_fragment = xp.sum(sp * fluence_spectra, axis=1)
        quantity_num_per_fragment = {}
        for qty in self.quantities_in_kernel:
            quantity_num_per_fragment[qty] = xp.sum(quantitys[qty] * sp * fluence_spectra, axis=1)

        # Accumulate over fragments -> (n_depths,). The bundled tables carry a single
        # (alpha_x, beta_x) class, so the result is broadcast over the tissue axis;
        # tables with per-class quantities need to index the table by class here.
        denominator = xp.sum(dose_per_fragment, axis=0)[:, None]
        valid = denominator > 0
        for qty in self.quantities_in_kernel:
            numerator = xp.sum(quantity_num_per_fragment[qty], axis=0)[:, None]
            kernel.quantities[qty] = xp.broadcast_to(
                xp.where(valid, numerator / xp.where(valid, denominator, 1.0), 0.0),
                (n_depths, n_tissue),
            )  # (n_depths, n_tissue_classes)

        return kernel

    def calc_biological_quantities_for_bixel(self, bixel: dict, kernels: dict) -> dict:
        """Calculate the biological quantities for a bixel."""
        bixel = super().calc_biological_quantities_for_bixel(bixel, kernels)
        return bixel

    @abstractmethod
    def load_quantity_table(self) -> dict:
        """
        Load the quantity table from the specified file. The table is expected to be in a .mat format and should contain the necessary data for quantity calculations.

        Returns
        -------
            dict: A dictionary containing the quantity table data.
        """
        pass


class TabulatedAlphaBetaModel(TabulatedRBEModel):
    """
    Tabulated LQ model using dose-averaged alpha and sqrt(beta) kernels.

    Looks up pre-computed alpha and sqrt(beta) values from a RBE
    table and accumulates dose-averaged quantities over the pencil-beam
    fluence spectrum.

    Class Attributes
    ----------------
    model : str
        ``"dose_average_alpha_beta``"
    quantities_in_table : list[str]
        ``["alpha", "beta"]`` — columns read from the ``.mat`` quantity table.
    quantities_in_kernel : list[str]
        ``["alpha", "sqrt_beta"]`` — names used in ``kernel.quantities``.
    required_quantities : list[str]
        ``["fluence"]`` — the machine file must supply fluence spectra.
    quantity_transforms : dict
        ``{"beta": "sqrt"}`` — beta is stored as sqrt(beta) in the kernel to
        allow dose-weighted averaging; squared back to beta at bixel level.

    Parameters
    ----------
    quantity_table_name : str
        Filename of the RBE table inside ``pyRadPlan.data.RBEtables``.
        Defaults to ``"RBEtable_LEMI_Scholz06_AX01_BX005.mat"``.
    """

    model = "dose_average_alpha_beta"
    quantities_in_table = ["alpha", "beta"]
    quantities_in_kernel = ["alpha", "sqrt_beta"]
    required_quantities = ["fluence"]  # input in machine file
    quantity_transforms = {
        "alpha": None,
        "beta": "sqrt",
    }

    def __init__(self):
        self.quantity_table_name = "RBEtable_LEMI_Scholz06_AX01_BX005.mat"
        super().__init__()

    def load_quantity_table(self) -> dict:
        """
        Load the quantity table from the specified file. The table is expected to be in a .mat format and should contain the necessary data for quantity calculations.

        Returns
        -------
            dict: A dictionary containing the quantity table data.
        """
        _table_path = self._folder_name_quantity_tables / self.quantity_table_name
        data = read_mat(_table_path)
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
        quantity_table["units"] = {}
        quantity_table["units"]["energies"] = "MeV/nucleon"
        quantity_table["units"]["alpha"] = "1/Gy"
        quantity_table["units"]["beta"] = "1/Gy^2"
        return quantity_table

    def calc_biological_quantities_for_bixel(self, bixel: dict, kernels: dict) -> dict:
        """Calculate the biological quantities for a bixel."""
        bixel = super().calc_biological_quantities_for_bixel(bixel, kernels)
        # here we do calculation from the quantity to the alpha and beta value from the LQ model, here simple **2 for beta
        xp = array_api_compat.array_namespace(bixel["rad_depths"])
        # kernels["alpha"/"sqrt_beta"] have shape (n_voxels, n_tissue_classes)
        tissue_ix = xp.astype(xp.asarray(bixel["v_tissue_index"]), xp.int64)
        voxel_ix = xp.arange(tissue_ix.shape[0])
        bixel["alpha"] = kernels["alpha"][voxel_ix, tissue_ix]
        bixel["beta"] = kernels["sqrt_beta"][voxel_ix, tissue_ix] ** 2
        return bixel
