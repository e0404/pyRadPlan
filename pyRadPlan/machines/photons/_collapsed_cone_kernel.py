import os
import sys
from typing import Any, Optional, Union

import numpy as np
from numpydantic import NDArray, Shape
from pydantic import Field, computed_field, field_validator, model_validator
from pymatreader import read_mat
from typing_extensions import Self

from pyRadPlan.core import PyRadPlanBaseModel

if sys.version_info < (3, 10):
    import importlib_resources as resources  # Backport for older versions
else:
    from importlib import resources

DEFAULT_KERNEL_FILE = resources.files("pyRadPlan.data.kernels").joinpath(
    "photons_collapsedCone_6MV.mat"
)


class PhotonCollapsedConeKernel(PyRadPlanBaseModel):
    """
    Collapsed cone energy deposition kernel, fitted per polar bin.

    The energy fraction deposited per unit radial distance r (mm of water) in a polar bin is
    ``A a exp(-a r) + sum_k B_k b_k exp(-b_k r)`` (primary and scatter component). The field
    aliases follow matRad's ``ccKernel.fit`` struct.

    Attributes
    ----------
    nominal_energy : float, optional
        Nominal energy of the kernel [MV].
    polar_angles : np.ndarray, shape (n_bins,)
        Solid angle bisectors of the polar bins [deg].
    polar_edges : np.ndarray, shape (n_bins + 1,)
        Polar bin edges [deg].
    primary_weights, primary_attenuations : np.ndarray, shape (n_bins,)
        Weights ``A`` and attenuation coefficients ``a`` [1/mm] of the primary component.
    scatter_weights, scatter_attenuations : np.ndarray, shape (n_terms, n_bins)
        Weights ``B`` and attenuation coefficients ``b`` [1/mm] of the scatter terms.
    min_radius : float
        Smallest radius [mm] included in the fit.
    scatter_max_radius : float
        Largest radius [mm] included in the fit of the scatter component.
    residual : dict
        Fit residuals per bin (see :func:`fit_collapsed_cone_kernel`).
    cones : dict, optional
        Conical kernel data (``angleEdges``, ``radiusEdges``, ``primary``, ``scatter``),
        needed to refit the kernel into other polar bins.
    """

    nominal_energy: Optional[float] = Field(
        default=None, ge=0.0, description="Nominal energy of the kernel [MV]"
    )
    polar_angles: NDArray[Shape["1-*"], np.float64] = Field(
        description="Solid angle bisectors of the polar bins [deg]"
    )
    polar_edges: NDArray[Shape["2-*"], np.float64] = Field(description="Polar bin edges [deg]")
    primary_weights: NDArray[Shape["1-*"], np.float64] = Field(
        alias="A", description="Primary weights"
    )
    primary_attenuations: NDArray[Shape["1-*"], np.float64] = Field(
        alias="a", description="Primary attenuation coefficients [1/mm]"
    )
    scatter_weights: NDArray[Shape["1-*, 1-*"], np.float64] = Field(
        alias="B", description="Scatter weights per term"
    )
    scatter_attenuations: NDArray[Shape["1-*, 1-*"], np.float64] = Field(
        alias="b", description="Scatter attenuation coefficients per term [1/mm]"
    )
    min_radius: float = Field(default=0.0, ge=0.0, description="Smallest fitted radius [mm]")
    scatter_max_radius: float = Field(
        default=np.inf, gt=0.0, description="Largest fitted radius of the scatter [mm]"
    )
    residual: dict[str, Any] = Field(default_factory=dict, description="Fit residuals per bin")
    cones: Optional[dict[str, Any]] = Field(default=None, description="Conical kernel data")

    @computed_field(return_type=int)
    @property
    def num_polar_angles(self):
        return self.polar_angles.size

    @computed_field(return_type=int)
    @property
    def num_scatter_exponentials(self):
        return self.scatter_weights.shape[0]

    @property
    def weights(self) -> np.ndarray:
        """Weights of all components, shape ``(n_components, n_bins)``."""
        return np.vstack([self.primary_weights, self.scatter_weights])

    @property
    def attenuations(self) -> np.ndarray:
        """Attenuation coefficients of all components, shape ``(n_components, n_bins)``."""
        return np.vstack([self.primary_attenuations, self.scatter_attenuations])

    @field_validator("polar_angles", "polar_edges", "primary_weights", "primary_attenuations",
                     mode="before")  # fmt: skip
    @classmethod
    def _as_vector(cls, v: Any) -> np.ndarray:
        return np.atleast_1d(np.asarray(v, dtype=np.float64)).ravel()

    @field_validator("scatter_weights", "scatter_attenuations", mode="before")
    @classmethod
    def _as_rows(cls, v: Any) -> np.ndarray:
        return np.atleast_2d(np.asarray(v, dtype=np.float64))

    @model_validator(mode="after")
    def _check_shapes(self) -> Self:
        n_bins = self.polar_angles.size
        if self.polar_edges.size != n_bins + 1:
            raise ValueError("Polar edges must have one entry more than the polar angles.")
        for name in ("primary_weights", "primary_attenuations"):
            if getattr(self, name).size != n_bins:
                raise ValueError(f"{name} must have one entry per polar bin.")
        if self.scatter_weights.shape != self.scatter_attenuations.shape:
            raise ValueError("Scatter weights and attenuation coefficients do not match.")
        if self.scatter_weights.shape[1] != n_bins:
            raise ValueError("Scatter weights must have one column per polar bin.")
        return self

    def check_energy(self, energy: Optional[float]) -> tuple[bool, str]:
        """
        Compare the nominal energy of the kernel with a beam energy.

        Returns
        -------
        tuple[bool, str]
            ``False`` and a message on a mismatch; ``True`` and a warning message if one of the
            energies is unknown; ``True`` and an empty message otherwise.
        """
        if self.nominal_energy is None or energy is None:
            return True, (
                "Nominal energy of the collapsed cone kernel or the machine unknown, "
                "energies not checked."
            )
        if abs(self.nominal_energy - float(energy)) > 1e-6:
            return False, (
                f"Collapsed cone kernel for {self.nominal_energy:g} MV does not match the beam "
                f"energy of {float(energy):g} MV. Generate a kernel with matRad's "
                "tools/collapsedConeKernels and set kernel_file."
            )
        return True, ""


def load_collapsed_cone_kernel(
    filename: Union[str, os.PathLike, None] = None,
) -> PhotonCollapsedConeKernel:
    """
    Read a matRad collapsed cone kernel file (struct ``ccKernel``).

    Parameters
    ----------
    filename : str or PathLike, optional
        Kernel file. Default: the 6 MV water kernel shipped with pyRadPlan.

    Returns
    -------
    PhotonCollapsedConeKernel
        The stored fit, with the conical kernel data if the file contains them.
    """
    path = DEFAULT_KERNEL_FILE if filename is None else filename
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Collapsed cone kernel file {path} not found!")
    data = read_mat(str(path))
    if "ccKernel" not in data:
        raise ValueError(f"Kernel file {path} does not contain a collapsed cone kernel!")
    kernel = data["ccKernel"]
    fit = kernel["fit"]
    cones = None
    if "cones" in kernel:
        cones = {
            key: np.asarray(kernel["cones"][key], dtype=np.float64)
            for key in ("angleEdges", "radiusEdges", "primary", "scatter")
        }
    nominal_energy = kernel.get("meta", {}).get("nominalEnergy")
    return PhotonCollapsedConeKernel(
        nominal_energy=None if nominal_energy is None else float(nominal_energy),
        polar_angles=fit["polarAngles"],
        polar_edges=fit["polarEdges"],
        A=fit["A"],
        a=fit["a"],
        B=fit["B"],
        b=fit["b"],
        min_radius=float(fit.get("minRadius", 0.0)),
        scatter_max_radius=float(fit.get("scatterMaxRadius", np.inf)),
        residual=dict(fit.get("residual", {})),
        cones=cones,
    )
