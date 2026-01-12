from typing import Any, Optional, Annotated, Union

import numpy as np
from pydantic import (
    Field,
    StringConstraints,
    field_validator,
    AliasChoices,
)
from numpydantic import NDArray, Shape
from pyRadPlan.util.helpers import dl2ld
from pyRadPlan.machines.base import ExternalBeamMachine
from ._beam_focus import ChargedBeamFocus
from .kernel._base import ParticlePencilBeamKernel
from ._beam_energy import ChargedEnergySpectrum, ChargedEnergySpectrumGaussian


class ParticleAccelerator(ExternalBeamMachine):
    """
    Base Machine Model for Charged Particle Accelerators.

    Defines minimum meta-data an ion machine must hold
    Provides multiple data storage formats, currently supported
    - Pencil-Beam Kernels
    - Monte Carlo Beam Models.

    Attributes
    ----------
    sad : float
        The source-to-axis (-isocenter) distance of the machine
    """

    radiation_mode: Annotated[str, StringConstraints(pattern="^(protons|helium|carbon|vhee)$")] = (
        Field(default="protons", validate_default=True)
    )

    sad: float = Field(ge=0.0, description="Source-to-axis distance", alias="SAD")
    bams_to_iso_dist: float = Field(
        ge=0.0,
        description="Beam-monioring-system/nozzle to iso center distance",
        alias="BAMStoIsoDist",
    )
    lut_spot_size: NDArray[Shape["2,*"], np.float64] = Field(
        description="2D array of look up table latteral spotspacing to foxus index",
        alias=AliasChoices(
            "LUTspotSize", "LUT_bxWidthminFWHM"
        ),  # We have two aliases here to capture version changes in matRad base data files
    )
    # kernel_data: Annotated[np.recarray, Field(description="Pencil beam kernel data")]

    fit_air_offset: float = Field(
        ge=0.0,
        default=0.0,
        description="Air distance included in fitting of the kernel",
    )

    # Minimum required data is energies, Bragg-peak positions and beam foci

    foci: dict[float, list[ChargedBeamFocus]]
    spectra: Optional[dict[float, ChargedEnergySpectrum]] = None

    pb_kernels: dict = None
    # Optionally we can have pencil-beam kernels

    @classmethod
    def _parse_tabulated_energy_data_from_mat(
        cls, tabulated_energy_data: dict, returned_data: dict
    ):
        """Parse the tabulated energy data from a matRad machine file."""

        super(ParticleAccelerator, cls)._parse_tabulated_energy_data_from_mat(
            tabulated_energy_data, returned_data
        )

        # extract beam foci
        foci = {}
        # First, check if we have the special situation of only a single focus

        foci_list = tabulated_energy_data["initFocus"]
        if not isinstance(foci_list, list):
            foci_list = [foci_list]

        for i, foci_entry in enumerate(foci_list):
            # check if all initFocus fields are lists
            if not all(isinstance(foci_entry[key], list) for key in foci_entry):
                focus_list = [foci_entry]
            else:
                focus_list = dl2ld(foci_entry)
            foci[returned_data["energies"][i]] = [
                ChargedBeamFocus(**focus) for focus in focus_list
            ]

        returned_data["foci"] = foci

        # Energy Spectra
        spectra_list = tabulated_energy_data.get("energySpectrum", None)
        if spectra_list is not None:
            returned_data["spectra"] = {
                returned_data["energies"][i]: ChargedEnergySpectrumGaussian(**spectrum)
                for i, spectrum in enumerate(spectra_list)
            }

    @field_validator("lut_spot_size", mode="before")
    @classmethod
    def validate_possible_cast_array(cls, v: Any) -> Any:
        """Validate if the input can be cast to a float64 array."""
        try:
            v = np.array(v, dtype=np.float64)
        except ValueError as exc:
            raise exc

        return v

    # TODO: Beam model descriptions for Monte Carlo

    @property
    def has_pb_kernels(self):
        """Test for existance of valid pencil-beam kernels."""
        return self.pb_kernels is not None

    @property
    def pb_kernels_have_let(self) -> bool:
        """Check if the kernel data contains LET values."""
        return all(kernel.let is not None for kernel in self.pb_kernels.values())

    def get_kernel_by_index(self, ix_energy: int) -> ParticlePencilBeamKernel:
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

    def get_foci_by_index(self, ix_energy: int) -> list[ChargedBeamFocus]:
        """
        Get the focus data of the kernel for a specific energy index.

        Parameters
        ----------
        ix_energy : int
            Index of the energy level in the stored data

        Returns
        -------
        IonBeamFocus
            Dictionary of available beam foci
        """
        return self.foci[self.energies[ix_energy]]

    def get_kernel_by_energy(self, energy: float) -> ParticlePencilBeamKernel:
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
            # return None
        return self.pb_kernels[energy]

    def update_kernel_at_index(self, ix_energy: int, kernel: ParticlePencilBeamKernel):
        """Update the pencil beam kernel for a specific energy index.

        Parameters
        ----------
        ix_energy : int
            Index of the energy level in the stored data
        kernel : IonPencilBeamKernel
            Pencil beam kernel data
        """

        self.update_kernel_for_energy(self.energies[ix_energy], kernel)

    def update_kernel_for_energy(
        self, energy: Union[float, np.float64], kernel: ParticlePencilBeamKernel
    ):
        """
        Update a pencil-beam kernel for a specific energy.

        Parameters
        ----------
        energy : float
            Energy value to update the kernel for
        kernel : IonPencilBeamKernel
            The Kernel to replace the existing one

        Notes
        -----
        If your intention is to build a new machine / add kernels to a machine without exisiting
        kernels, you should create the machine from scratch or set the whole model field at once
        which is a dictionary of all kernels with the energy as key
        """
        if self.pb_kernels is None:
            raise ValueError("No pencil beam kernels available for this machine.")
        energy = np.float64(energy)

        self.pb_kernels[energy] = kernel

    def get_focus(self, ix_energy: int) -> dict:
        """
        Get the focus data of the kernel for a specific energy index.

        Parameters
        ----------
        ix_energy : int
            Index of the energy level in the stored data

        Returns
        -------
        dict
            Dictionary of focus data
        """

        return self.foci[self.energies[ix_energy]]

    @property
    def has_single_gaussian_kernel(self) -> bool:
        """
        Check if it has single gaussian pencil-beamkernels.

        Returns
        -------
        bool
            True if all kernels have single gaussian lateral scattering model
        """
        if self.pb_kernels is None:
            return False

        return all(kernel.sigma is not None for kernel in self.pb_kernels.values())

    @property
    def has_double_gaussian_kernel(self) -> bool:
        """
        Check if it has double gaussian pencil-beamkernels.

        Returns
        -------
        bool
            True if all kernels have double gaussian lateral scattering model
        """
        if self.pb_kernels is None:
            return False

        return all(
            kernel.sigma_1 is not None and kernel.sigma_2 is not None and kernel.weight is not None
            for kernel in self.pb_kernels.values()
        )

    @property
    def has_multi_gaussian_kernel(self) -> bool:
        """
        Check if it has multi gaussian pencil-beamkernels.

        Returns
        -------
        bool
            True if all kernels have multi gaussian lateral scattering model
        """
        if self.pb_kernels is None:
            return False

        return all(
            kernel.sigma_multi is not None and kernel.weight_multi is not None
            for kernel in self.pb_kernels.values()
        )

    @property
    def has_focused_gaussian_kernel(self) -> bool:
        """
        Check if it has focused gaussian pencil-beamkernels.

        Returns
        -------
        bool
            True if all kernels have focused gaussian lateral scattering model
        """
        if self.pb_kernels is None:
            return False

        return all(
            kernel.sigma_x is not None and kernel.sigma_y is not None
            for kernel in self.pb_kernels.values()
        )

    @property
    def has_let_kernel(self) -> bool:
        """
        Check if it has LET values in the pencil-beam kernels.

        Returns
        -------
        bool
            True if all kernels have LET values
        """
        if self.pb_kernels is None:
            return False

        return all(kernel.let is not None for kernel in self.pb_kernels.values())

    @property
    def has_alpha_beta_kernels(self) -> bool:
        """
        Check if it has alpha-beta kernels.

        Returns
        -------
        bool
            True if all kernels have alpha-beta values
        """

        if self.pb_kernels is None:
            return False

        return all(
            kernel.alpha is not None and kernel.beta is not None
            for kernel in self.pb_kernels.values()
        )
