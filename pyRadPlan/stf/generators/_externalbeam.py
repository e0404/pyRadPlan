import math
import warnings
from typing import Union
import numpy as np
from numpy.typing import ArrayLike

from pyRadPlan.geometry import lps
from pyRadPlan.plan import Plan
from ._base import StfGeneratorBase


def _deprecated_alias(old: str, new: str) -> property:
    """Property forwarding the deprecated attribute *old* to *new*."""
    message = f"{old} is deprecated, use {new} instead."

    def fget(self):
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return getattr(self, new)

    def fset(self, value):
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        setattr(self, new, value)

    return property(fget, fset, doc=f"Deprecated alias for {new}.")


def resolve_iso_centers(iso_center, cst, num_of_beams: int | None = None) -> np.ndarray:
    """
    Resolve the isocenter of every beam as the stf generators use it.

    Parameters
    ----------
    iso_center : ArrayLike or None
        One isocenter (x, y, z) for all beams, or one per beam. ``None`` or an
        empty array selects the target center of mass of ``cst``.
    cst : StructureSet
        Structure set providing the target center of mass.
    num_of_beams : int, optional
        Number of beams. A single isocenter is repeated for all of them, per-beam
        isocenters have to match their number. If omitted, the isocenters are
        returned as given.

    Returns
    -------
    np.ndarray
        The isocenters in mm, shape ``(num_of_beams, 3)``, or ``(k, 3)`` without
        ``num_of_beams``.
    """
    if iso_center is None or np.size(iso_center) == 0:
        iso_center = cst.target_center_of_mass()
    iso_center = np.asarray(iso_center, dtype=np.float64)
    if iso_center.size % 3 != 0:
        raise ValueError(
            "Iso center needs to have three coordinates (1x3 array) "
            "or a set of 3D coordinates for each beam (nx3 array)."
        )
    iso_center = iso_center.reshape((-1, 3))
    if num_of_beams is None:
        return iso_center
    if iso_center.shape[0] == 1:
        return np.repeat(iso_center, num_of_beams, axis=0)
    if iso_center.shape[0] != num_of_beams:
        raise ValueError(
            f"Got {iso_center.shape[0]} iso centers for {num_of_beams} beams; "
            "give one iso center or one per beam."
        )
    return iso_center


class StfGeneratorExternalBeam(StfGeneratorBase):
    """
    Base class for geometry enerated for external beam therapy.

    Provides the basic functionality and infrastructure for external beam
    therapy, such as the definition of gantry / couch angles, an isocenter,
    and basic ray geometry transformations.

    Parameters
    ----------
    pln : Plan, optional
        A Plan object. If given, configuration will be loaded from the plan.

    Attributes
    ----------
    gantry_angles : list[float]
        A list of gantry angles.
    couch_angles : list[float]
        A list of couch angles.
    iso_center : Union[ArrayLike, None]
        The isocenter coordinates.
    """

    gantry_angles: list[float]
    couch_angles: list[float]
    iso_center: Union[ArrayLike, None]

    @property
    def num_of_beams(self) -> int:
        """Number of beams."""
        if len(self.gantry_angles) != len(self.couch_angles):
            raise ValueError("Inconsistent number of gantry and couch angles.")

        return len(self.gantry_angles)

    def __init__(self, pln: Plan = None):
        self.gantry_angles = [0.0]
        self.couch_angles = [0.0]
        self.iso_center = None

        super().__init__(pln)


class StfGeneratorExternalBeamRayBixel(StfGeneratorExternalBeam):
    """
    Base class for ray-bixel-based geometry for IMRT.

    Attributes
    ----------
    bixel_width : float
        The bixel width (beamlet / spot spacing).
    """

    bixel_width: float

    def __init__(self, pln: Plan = None):
        self.bixel_width = 5.0

        super().__init__(pln)

    def _computed_target_margin(self) -> float:
        """Target margin to apply for beamlet placing."""
        return self.bixel_width

    def _finalize_beam(self, beam: dict) -> dict:
        """Post-process a beam after its rays were created (matRad's ``finalizeBeam``)."""
        return beam

    def _create_rays(self, beam) -> list[dict]:
        """Get the rays on a beam / stf element."""

        ray_pos = self._generate_ray_positions_in_isocenter_plane(beam)

        # Get the rotation matrix in LPS rotating the coordinates into the BEV System
        rotation_matrix = lps.get_beam_rotation_matrix(beam["gantry_angle"], beam["couch_angle"])

        rays = list[dict]()
        for j in range(ray_pos.shape[1]):
            ray = {
                "ray_pos_bev": ray_pos[:, j].flatten(),
                "ray_pos": rotation_matrix @ ray_pos[:, j].flatten(),
            }
            rays.append(ray)

        return rays

    def _generate_ray_positions_in_isocenter_plane(self, beam) -> np.ndarray:
        """Generate the ray positions for the STF in the isocenter plane."""

        rotation_matrix_t = lps.get_beam_rotation_matrix(
            beam["gantry_angle"], beam["couch_angle"]
        ).transpose()

        sad = beam["sad"]
        bw = beam["bixel_width"]

        iso_coords = self._target_voxel_coordinates - beam["iso_center"].reshape((3, 1))

        # Get target voxel coordinates in BEV in the isocenter plane
        bev_coords_isoplane = rotation_matrix_t @ iso_coords
        bev_coords_isoplane = sad * bev_coords_isoplane / (sad + bev_coords_isoplane[1, :])
        bev_coords_isoplane[1, :] = 0.0

        # Obtain unique ray positions
        ray_pos = np.unique(bw * np.round(bev_coords_isoplane / bw), axis=1)

        # Obtain unique ray positions
        # Pad ray position array if resolution of target voxel grid not sufficient
        max_ct_resolution = max(
            [self._ct.resolution["x"], self._ct.resolution["y"], self._ct.resolution["z"]]
        )
        if self.bixel_width < max_ct_resolution:
            orig_ray_pos = ray_pos.copy()
            for j in range(
                -math.floor(max_ct_resolution / self.bixel_width),
                math.floor(max_ct_resolution / self.bixel_width),
            ):
                for k in range(
                    -math.floor(max_ct_resolution / self.bixel_width),
                    math.floor(max_ct_resolution / self.bixel_width),
                ):
                    if abs(j) + abs(k) == 0:
                        continue
                    ray_pos = np.hstack(
                        (
                            ray_pos,
                            orig_ray_pos
                            + np.array((j * self.bixel_width, 0.0, k * self.bixel_width)).reshape(
                                (3, 1)
                            ),
                        )
                    )

        # TODO: for DAO, there is a filling of empty bixels in matRad

        ray_pos = np.unique(ray_pos, axis=1)

        return ray_pos

    def _generate_source_geometry(self):
        """Generate the source geometry for the STF."""

        # Validate bixel width
        if self.bixel_width <= 0.0:
            raise ValueError(
                "Bixel width (spot distance) needs to be a real number [mm] larger than zero."
            )

        self.iso_center = resolve_iso_centers(self.iso_center, self._cst, self.num_of_beams)

        stf = []

        for i in self.track(range(self.num_of_beams), name="Beam", unit="b"):
            beam = {}

            # Correct for iso center position. With this correction isocenter is (0, 0, 0) [mm]
            # coordsX = (self.coordsX_vox * self.ct.resolution["x"] - self.iso_center[i][0])
            # coordsY = (self.coordsY_vox * self.ct.resolution["y"] - self.iso_center[i][1])
            # coordsZ = (self.coordsZ_vox * self.ct.resolution["z"] - self.iso_center[i][2])

            # Save meta information for treatment plan
            beam["gantry_angle"] = self.gantry_angles[i]
            beam["couch_angle"] = self.couch_angles[i]
            beam["bixel_width"] = self.bixel_width
            beam["radiation_mode"] = self.radiation_mode
            beam["sad"] = self.machine.sad
            beam["iso_center"] = self.iso_center[i]
            beam["machine"] = self.machine.name
            beam["source_point_bev"] = np.array([0.0, -beam["sad"], 0.0], dtype=float)

            # Get the rotation matrix in LPS rotating the coordinates into the BEV system
            rotation_matrix = lps.get_beam_rotation_matrix(
                beam["gantry_angle"], beam["couch_angle"]
            )

            beam["source_point"] = rotation_matrix @ beam["source_point_bev"]

            beam["rays"] = self._create_rays(beam)
            beam = self._finalize_beam(beam)

            stf.append(beam)
        return stf


class StfGeneratorSingleBixel(StfGeneratorExternalBeam):
    """Base class for ray-bixel-based geometry for IMRT."""

    def __init__(self, pln: Plan = None):
        super().__init__(pln)

    def _computed_target_margin(self) -> float:
        """Target margin to apply for beamlet placing."""
        return 0.0

    def _create_rays(self, beam) -> list[dict]:
        """Get the rays on a beam / stf element."""

        ray_pos = np.array([0.0, 0.0, 0.0]).reshape((3, 1))

        # Get the rotation matrix in LPS rotating the coordinates into the BEV System
        rotation_matrix = lps.get_beam_rotation_matrix(beam["gantry_angle"], beam["couch_angle"])

        ray = {
            "ray_pos_bev": ray_pos[:, 0].flatten(),
            "ray_pos": rotation_matrix @ ray_pos[:, 0].flatten(),
        }

        return [ray]

    def _generate_source_geometry(self):
        """Generate the source geometry for the STF."""

        self.iso_center = resolve_iso_centers(self.iso_center, self._cst, self.num_of_beams)

        stf = []

        for i in self.track(range(self.num_of_beams), name="Beam", unit="b"):
            beam = {}

            # Save meta information for treatment plan
            beam["gantry_angle"] = self.gantry_angles[i]
            beam["couch_angle"] = self.couch_angles[i]
            beam["radiation_mode"] = self.radiation_mode
            beam["sad"] = self.machine.sad
            beam["iso_center"] = self.iso_center[i]
            beam["machine"] = self.machine.name
            beam["source_point_bev"] = np.array([0.0, -beam["sad"], 0.0], dtype=float)

            # Get the rotation matrix in LPS rotating the coordinates into the BEV system
            rotation_matrix = lps.get_beam_rotation_matrix(
                beam["gantry_angle"], beam["couch_angle"]
            )

            beam["source_point"] = rotation_matrix @ beam["source_point_bev"]

            beam["rays"] = self._create_rays(beam)

            stf.append(beam)
        return stf
