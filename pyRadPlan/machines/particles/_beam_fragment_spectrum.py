from typing import Any, ClassVar, Optional
import numpy as np
from pydantic import (
    Field,
)
from pyRadPlan.core import PyRadPlanBaseModel
from numpydantic import NDArray, Shape


class FragmentFluence(PyRadPlanBaseModel):
    """
    Fluence data for a single fragment species.

    Attributes
    ----------
    fluence_spectrum : (n_energies, n_depths)
        Fluence at each energy bin × depth.
    energy : (n_energies,)
        Energy bin centres/edges corresponding to fluence_spectrum rows.
    fluenceZ : (n_depths,)
        Fluence integrated over energy at each depth (collapsed spectrum).
    """

    Z: int = Field(..., description="Atomic number. -1 for electronentry.")
    A: float = Field(..., description="Mass number. NaN  if all with that Z are aggregated.")

    fluence_spectrum: NDArray[Shape["1-*, 1-*"], np.float64] = Field(
        ..., description="Fluence spectrum, shape (n_energies, n_depths)."
    )
    energy: NDArray[Shape["1-*"], np.float64] = Field(
        ..., description="Energy bins, length n_energies."
    )
    fluenceZ: NDArray[Shape["1-*"], np.float64] = Field(
        ..., description="Fluence integrated over energy at each depth, length n_depths."
    )


class ChargedBeamFragmentSpectrum(PyRadPlanBaseModel):
    """
    Fragment spectrum data for a charged particle beam.
    """

    type: ClassVar[str] = "fluence"

    fragments: list[FragmentFluence] = Field(
        ..., description="Per-species fluence data, one entry per (Z, A) pair."
    )

    def get(self, Z: float, A: float) -> Optional[FragmentFluence]:
        """Look up a specific fragment by (Z, A). Returns None if not found."""
        for entry in self.fragments:
            if entry.Z == Z and entry.A == A:
                return entry
        return None

    @property
    def Z_values(self) -> list[float]:
        return [e.Z for e in self.fragments]

    @property
    def A_values(self) -> list[float]:
        return [e.A for e in self.fragments]

    @classmethod
    def from_dict(cls, data: dict) -> "ChargedBeamFragmentSpectrum":
        """
        Construct from a imported machine data matfile loaded as a dict.

        Expected dict keys (matching MATLAB field names):
            Z               : array-like, shape (n_entries,)
            A               : array-like, shape (n_entries,)  — NaN for aggregate
            fluenceSpectrum : array-like, shape (n_entries,)  — each cell is (n_depths, n_energies)
            energyBin       : array-like, shape (n_entries,)  — each cell is (1, n_energies)
            fluenceDepth    : array-like, shape (n_entries,)  — each cell is (1, n_depths)

        """
        Z_arr = np.asarray(data["spectra"]["Z"]).ravel()
        A_arr = np.asarray(data["spectra"]["A"], dtype=float).ravel()
        spectra = data["spectra"]["fluenceSpectrum"]  # iterable of 2-D arrays
        e_bins = data["spectra"]["energyBin"]  # iterable of 1×n arrays
        f_Z = data["spectra"]["fluenceDepth"]  # iterable of 1×n arrays

        n = len(Z_arr)
        if not (len(A_arr) == len(spectra) == len(e_bins) == len(f_Z) == n):
            raise ValueError(
                f"All MATLAB struct fields must have the same number of entries, "
                f"got Z={len(Z_arr)}, A={len(A_arr)}, fluenceSpectrum={len(spectra)}, "
                f"energyBin={len(e_bins)}, fluenceDepth={len(f_Z)}."
            )

        fragments: list[FragmentFluence] = []

        for i in range(n):
            Z = int(Z_arr[i])
            A = float(A_arr[i])

            spectrum = np.asarray(spectra[i], dtype=np.float64)  # (n_depths, n_energies)
            energy = np.asarray(e_bins[i], dtype=np.float64).ravel()
            f_arr = np.asarray(f_Z[i], dtype=np.float64).ravel()

            entry = FragmentFluence(
                Z=Z,
                A=A,
                fluence_spectrum=spectrum,
                energy=energy,
                fluenceZ=f_arr,
            )
            fragments.append(entry)

        return cls(fragments=fragments)

    def to_namespace(self, xp: Any, device: Any = None) -> "ChargedBeamFragmentSpectrum":
        def _convert(arr):
            if device is not None:
                return xp.asarray(arr, device=device)
            return xp.asarray(arr)

        converted_fragments = [
            FragmentFluence(
                Z=frag.Z,
                A=frag.A,
                fluence_spectrum=_convert(frag.fluence_spectrum),
                energy=_convert(frag.energy),
                fluenceZ=_convert(frag.fluenceZ),
            )
            for frag in self.fragments
        ]
        return ChargedBeamFragmentSpectrum(fragments=converted_fragments)
