from typing import Annotated, Any, Literal, Optional, ClassVar, Union, cast, List
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from pydantic import (
    Field,
    field_validator,
    model_validator,
    ValidatorFunctionWrapHandler,
    ValidationError,
)

from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.machines.base import ExternalBeamMachine
from ._svd_kernel import PhotonSVDKernel


class PhotonSpectrumAnalyticMV(PyRadPlanBaseModel):
    """Analytic MV photon spectrum after Ali & Rogers (2012).

    Parameterizes the bremsstrahlung spectrum of a flattened linac beam by the
    fitted electron energy and four fit constants (Ali & Rogers, Phys. Med.
    Biol. 57 (2012) 31-50, table 5). pyRadMC evaluates this form directly.

    Attributes
    ----------
    fitted_electron_energy : float
        The fitted electron endpoint energy in MeV (differs from the nominal
        accelerating potential, e.g. 6.83 MeV for the Siemens 6 MV beam).
    c1, c2, c3, c4 : float
        The four fit constants of the analytic form.
    energy_min : float
        Lower spectrum cutoff in MeV when discretizing to a histogram.
    num_bins : int
        Number of histogram bins when discretizing.
    """

    kind: Literal["analytic_mv"] = "analytic_mv"
    fitted_electron_energy: Annotated[float, Field(gt=0.0)]
    c1: float
    c2: float
    c3: float
    c4: float
    energy_min: Annotated[float, Field(gt=0.0)] = 0.15
    num_bins: Annotated[int, Field(gt=0)] = 100

    @property
    def max_energy(self) -> float:
        """Maximum photon energy in MeV."""
        return self.fitted_electron_energy


class PhotonSpectrumHistogram(PyRadPlanBaseModel):
    """Histogram photon spectrum: bin edges (MeV) and per-bin weights.

    Attributes
    ----------
    edges : np.ndarray
        Bin edges in MeV, strictly increasing, first edge positive (n + 1).
    weights : np.ndarray
        Per-bin content (n non-negative values, at least one positive).
    convention : str
        'number' (photons per bin) or 'energy_fluence' (converted to photon
        number at the bin midpoint).
    """

    kind: Literal["histogram"] = "histogram"
    edges: NDArray
    weights: NDArray
    convention: Literal["number", "energy_fluence"] = "number"

    @field_validator("edges", "weights", mode="before")
    @classmethod
    def _as_float_array(cls, v: Any) -> np.ndarray:
        return np.asarray(v, dtype=np.float64)

    @model_validator(mode="after")
    def _check_histogram(self):
        if self.edges.ndim != 1 or self.edges.size < 2:
            raise ValueError("edges must be a 1D array with at least two entries")
        if self.edges[0] <= 0.0 or not np.all(np.diff(self.edges) > 0.0):
            raise ValueError("edges must be strictly increasing with a positive first edge")
        if self.weights.shape != (self.edges.size - 1,):
            raise ValueError("need exactly one weight per bin")
        if np.any(self.weights < 0.0) or not np.any(self.weights > 0.0):
            raise ValueError("weights must be non-negative with at least one positive entry")
        return self

    @property
    def max_energy(self) -> float:
        """Maximum photon energy in MeV."""
        return float(self.edges[-1])


class PhotonSpectrumMonoenergetic(PyRadPlanBaseModel):
    """Monoenergetic photon 'spectrum' (a single energy in MeV)."""

    kind: Literal["monoenergetic"] = "monoenergetic"
    energy: Annotated[float, Field(gt=0.0)]

    @property
    def max_energy(self) -> float:
        """Maximum photon energy in MeV."""
        return self.energy


PhotonSpectrumModel = Annotated[
    Union[PhotonSpectrumAnalyticMV, PhotonSpectrumHistogram, PhotonSpectrumMonoenergetic],
    Field(discriminator="kind"),
]


class PhotonJawPairModel(PyRadPlanBaseModel):
    """One collimator jaw pair for Monte Carlo beam models.

    Geometry constants of the device (the per-field opening comes from the
    steering information). Distances are in mm downstream of the photon source
    along the beam axis.

    Attributes
    ----------
    axis : str
        The BEV axis the pair limits: 'x' (crossplane) or 'y' (inplane).
    z_top, z_bottom : float
        Upper/lower face of the jaw slab in mm from the source.
    material : str
        Device material ('tungsten' or 'water' for a water-equivalent model).
    density : float
        Mass density in g/cm^3 (tungsten alloy ~18.0).
    focused : bool
        Whether the jaw edges are focused to the source (divergent machine).
    """

    axis: Literal["x", "y"]
    z_top: Annotated[float, Field(gt=0.0)]
    z_bottom: Annotated[float, Field(gt=0.0)]
    material: Literal["tungsten", "water"] = "tungsten"
    density: Annotated[float, Field(gt=0.0)] = 18.0
    focused: bool = False

    @model_validator(mode="after")
    def _check_slab(self):
        if self.z_bottom <= self.z_top:
            raise ValueError("z_bottom must exceed z_top")
        return self


class PhotonMLCModel(PyRadPlanBaseModel):
    """A multi-leaf collimator for Monte Carlo beam models.

    Geometry constants of the device; per-field leaf positions come from the
    steering information. Distances in mm downstream of the source; lateral
    coordinates in mm *at the isocenter plane* (projected to the device plane
    when the transport geometry is built).

    Attributes
    ----------
    travel_axis : str
        The BEV axis the leaves travel along: 'x' (crossplane) or 'y' (inplane).
    z_top, z_bottom : float
        Upper/lower face of the leaf slab in mm from the source.
    leaf_edges : np.ndarray
        Leaf-side boundaries along the orthogonal axis, mm at the isocenter
        plane, strictly increasing (n_pairs + 1 values).
    leaf_end_design : str
        Physical leaf-end design: 'rounded' (cylindrical tips of
        ``tip_radius``) or 'double_focused' (straight ends focused to the
        source, e.g. Siemens PRIMUS). Transport engines that cannot model
        focused ends exactly may approximate them (and should warn).
    tip_radius : float
        Rounded leaf-tip radius in mm; for a double-focused MLC this is the
        radius used by engines that approximate the focused end.
    material : str
        Device material ('tungsten' or 'water').
    density : float
        Mass density in g/cm^3.
    """

    travel_axis: Literal["x", "y"] = "x"
    z_top: Annotated[float, Field(gt=0.0)]
    z_bottom: Annotated[float, Field(gt=0.0)]
    leaf_edges: NDArray
    leaf_end_design: Literal["rounded", "double_focused"] = "rounded"
    tip_radius: Annotated[float, Field(gt=0.0)] = 150.0
    material: Literal["tungsten", "water"] = "tungsten"
    density: Annotated[float, Field(gt=0.0)] = 18.0

    @field_validator("leaf_edges", mode="before")
    @classmethod
    def _as_float_array(cls, v: Any) -> np.ndarray:
        return np.asarray(v, dtype=np.float64)

    @model_validator(mode="after")
    def _check_mlc(self):
        if self.z_bottom <= self.z_top:
            raise ValueError("z_bottom must exceed z_top")
        if self.leaf_edges.ndim != 1 or self.leaf_edges.size < 2:
            raise ValueError("leaf_edges must be a 1D array with at least two entries")
        if not np.all(np.diff(self.leaf_edges) > 0.0):
            raise ValueError("leaf_edges must be strictly increasing")
        return self

    @property
    def num_leaf_pairs(self) -> int:
        """Number of leaf pairs."""
        return int(self.leaf_edges.size - 1)


def _default_primus_leaf_edges() -> np.ndarray:
    """Siemens PRIMUS leaf boundaries at the isocenter plane (mm).

    27 central leaves projecting to 10 mm at the isocenter, flanked by one
    65 mm leaf on each side (Sontag & Steinberg, Med. Phys. 26 (1999) 734).
    """
    return np.concatenate(([-200.0], np.arange(-135.0, 136.0, 10.0), [200.0]))


# Device slab thickness (mm) and jaw-to-MLC gap of the default PRIMUS-like
# geometry; the MLC mid-plane is anchored at the machine's source-to-collimator
# distance (scd), which is the collimation plane used for geometric penumbra.
_DEFAULT_DEVICE_THICKNESS = 76.0
_DEFAULT_JAW_MLC_GAP = 8.0


def _default_primus_devices(scd: float) -> dict:
    """Build the default PRIMUS-like jaw/MLC geometry anchored at ``scd`` (mm)."""
    mlc_top = scd - _DEFAULT_DEVICE_THICKNESS / 2.0
    mlc_bottom = scd + _DEFAULT_DEVICE_THICKNESS / 2.0
    jaw_bottom = mlc_top - _DEFAULT_JAW_MLC_GAP
    jaw_top = jaw_bottom - _DEFAULT_DEVICE_THICKNESS
    return {
        "jaws": [PhotonJawPairModel(axis="y", z_top=jaw_top, z_bottom=jaw_bottom, focused=True)],
        "mlc": PhotonMLCModel(
            travel_axis="x",
            z_top=mlc_top,
            z_bottom=mlc_bottom,
            leaf_edges=_default_primus_leaf_edges(),
            leaf_end_design="double_focused",
        ),
    }


class PhotonLINACBeamModelMC(PyRadPlanBaseModel):
    """Monte Carlo beam model description of a photon LINAC.

    Holds the primary photon spectrum per nominal energy, the focal spot size,
    and the beam-limiting device geometry (jaws/MLC) needed to build a
    Monte Carlo treatment-head model.

    The defaults describe the Siemens PRIMUS the 'Generic' machine is based on
    (Sontag & Steinberg, Med. Phys. 26 (1999) 734): a 6 MV Ali & Rogers
    analytic spectrum, one focused Y jaw pair, and an X-travelling
    double-focused MLC that replaces the lower jaw. The default device planes
    anchor the MLC mid-plane at the machine's source-to-collimator distance
    (``scd``, the collimation plane used for geometric penumbra); slab
    thickness and materials are approximate placeholders pending a measured
    machine description.

    Attributes
    ----------
    spectra : dict[float, PhotonSpectrumModel]
        Primary photon spectrum per nominal beam energy (MV).
    focal_spot_sigma : float
        Gaussian focal spot sigma in mm (0 = ideal point source).
    jaws : list[PhotonJawPairModel]
        Collimator jaw pairs, may be empty.
    mlc : PhotonMLCModel, optional
        The multi-leaf collimator, if the machine has one.
    """

    spectra: dict[float, PhotonSpectrumModel] = Field(
        default_factory=lambda: {
            6.0: PhotonSpectrumAnalyticMV(
                fitted_electron_energy=6.83, c1=1.184, c2=4.840, c3=-1.161, c4=0.01416
            )
        }
    )
    focal_spot_sigma: Annotated[float, Field(ge=0.0)] = 0.0
    jaws: list[PhotonJawPairModel] = Field(
        default_factory=lambda: _default_primus_devices(500.0)["jaws"]
    )
    mlc: Optional[PhotonMLCModel] = Field(
        default_factory=lambda: _default_primus_devices(500.0)["mlc"]
    )

    def get_spectrum(self, energy: float) -> PhotonSpectrumModel:
        """Get the spectrum for a nominal energy (closest stored key)."""
        if not self.spectra:
            raise ValueError("No spectra stored in the MC beam model.")
        closest = min(self.spectra.keys(), key=lambda e: abs(e - energy))
        return self.spectra[closest]


class PhotonLINAC(ExternalBeamMachine):
    """
    Machine Model for Photon LINAC-like Machines.

    Defines minimum meta-data a photon LINAC machine must hold
    Provides multiple data storage formats, currently supported
    - SVD Kernel Data (Bortfeld 1993, 10.1118/1.597070).

    Attributes
    ----------
    scd : float
        The source-to-collimator distance of the machine
    pb_kernels : dict[float, PhotonSVDKernel]
        Pencil-beam kernels for the machine
    of : np.ndarray
        Output factor data
    tpr : np.ndarray
        Tissue-to-phantom ratio data

    Notes
    -----
    Version-1 machine data is converted on validation: the pencil-beam kernels are rescaled
    by 4 to the version-2 convention and the machine then reports ``version == 2``, so that
    validating it or its dump again does not rescale the kernels a second time.
    """

    # Defining some naming properties (must keep type annotations when overriding pydantic fields)
    name: str = Field(default="Generic", alias="machine")
    key: str = "photon_linac_generic"
    radiation_mode: str = Field(default="photons", pattern="^(photons)$", validate_default=True)
    _possible_radiation_modes: ClassVar[List[str]] = ["photons"]

    # Machine geometry
    scd: float = Field(ge=0.0, description="Source to collimator distance", alias="SCD")

    # Optionally we can have pencil-beam kernels
    pb_kernels: Optional[dict[float, PhotonSVDKernel]] = None

    # Commissioning data
    of: Optional[NDArray] = Field(default=None, description="Output Factor")
    tpr: Optional[NDArray] = Field(default=None, description="Tissue-to-phantom-Ratio")

    # Monte Carlo beam model (spectrum + beam-limiting device geometry).
    # Defaults describe the Siemens PRIMUS the 'Generic' machine is based on,
    # with the MLC mid-plane anchored at the machine's scd; a machine file may
    # override this block once the format carries it.
    mc_beam_model: PhotonLINACBeamModelMC = Field(
        default_factory=lambda data: PhotonLINACBeamModelMC(
            **_default_primus_devices(data["scd"])
        ),
        description="Monte Carlo beam model (spectrum and beam-limiting devices)",
    )

    @classmethod
    def _parse_tabulated_energy_data_from_mat(
        cls, tabulated_energy_data: dict, returned_data: dict
    ):
        """Parse the tabulated energy data from a matRad machine file."""

        super(PhotonLINAC, cls)._parse_tabulated_energy_data_from_mat(
            tabulated_energy_data, returned_data
        )

        # parse kernel
        # returned_data["scd"] = tabulated_energy_data.get("scd", 0.0)
        returned_data["pb_kernels"] = tabulated_energy_data

    @field_validator("pb_kernels", mode="wrap")
    @classmethod
    def validate_pb_kernels(cls, v: Any, handler: ValidatorFunctionWrapHandler) -> PhotonSVDKernel:
        try:
            pb_kernels = handler(v)
        except ValidationError as err:
            # If we have a dict, we try to convert it to a PhotonSVDKernel object
            if isinstance(v, dict):
                # This check tries to check if we have a matRad kernel data set
                if "kernel" in v and "kernel_data" not in v:
                    try:
                        kernel = cast(dict, v["kernel"])
                        v["kernel_ssds"] = np.asarray(kernel.pop("SSD"), dtype=np.float64)
                        kernel_list = [
                            np.asarray(kernel.pop(f"kernel{i}"), dtype=np.float64)
                            for i in range(1, len(kernel) + 1)
                        ]
                        v["kernel_data"] = np.ascontiguousarray(
                            np.moveaxis(np.asarray(kernel_list), 0, 1)
                        )

                        # kernel dictionary by energy
                        v = {float(v["energy"]): v}
                    except Exception as exc:
                        raise ValueError("Could not parse matRad kernel data") from exc

                    pb_kernels = handler(v)
                else:
                    # Otherwise we don't know what to do
                    raise err
            else:
                raise err

        return cast(dict[float, PhotonSVDKernel], pb_kernels)

    @model_validator(mode="after")
    def _convert_legacy_kernels(self) -> Self:
        """
        Rescale version-1 pencil beam kernels to the version-2 convention, once.

        Version-1 kernels imply a hardcoded 2D convolution resolution of (0.5 mm)^2 and are
        multiplied by 4. The machine is then marked as version 2, so that validating it or its
        dump again does not rescale the kernels a second time. The kernels are replaced rather
        than modified in place, as they may be shared with the validated input.
        """
        if self.version < 2 and self.pb_kernels:
            # Written to __dict__: assignment validation would rerun this validator.
            self.__dict__["pb_kernels"] = {
                energy: kernel.model_copy(update={"kernel_data": 4.0 * kernel.kernel_data})
                for energy, kernel in self.pb_kernels.items()
            }
            self.__dict__["version"] = 2
            self.__pydantic_fields_set__.add("version")
        return self

    def get_kernel_by_index(self, ix_energy: int) -> PhotonSVDKernel:
        """Get the pencil beam kernel for a specific energy index.

        Parameters
        ----------
        ix_energy : int
            Index of the energy level in the stored data

        Returns
        -------
        IonPencilBeamKernel
            Pencil beam kernel data
        """
        return self.get_kernel_by_energy(self.energies[ix_energy])

    def get_kernel_by_energy(self, energy: float) -> PhotonSVDKernel:
        """Get the pencil beam kernel for a specific energy value.

        Parameters
        ----------
        energy : float
            Energy value to search for

        Returns
        -------
        IonPencilBeamKernel
            Pencil beam kernel data
        """
        if self.pb_kernels is None:
            raise ValueError("No pencil beam kernels available for this machine.")
        return self.pb_kernels[energy]
