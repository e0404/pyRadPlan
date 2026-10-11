"""Beam-angle presets for external-beam steering information generators.

A preset bundles gantry/couch angles with the radiation modes it is typical
for.  Presets live in a registry (:func:`register_beam_angle_preset`) so that
further presets, e.g. from a user configuration, can be added at runtime;
:func:`get_beam_angle_presets` lists them with the presets typical for a
radiation mode first.

Presets set the ``gantry_angles`` / ``couch_angles`` of
:class:`~pyRadPlan.stf.generators.StfGeneratorExternalBeam` generators
(:meth:`BeamAnglePreset.applies_to`); geometries without discrete beam angles
(e.g. arcs) would get their own preset type with the same ``name``,
``radiation_modes``, ``prop_stf`` and ``applies_to`` interface.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .generators._externalbeam import StfGeneratorExternalBeam

_ION_MODES = ("protons", "helium", "carbon", "oxygen")
_PHOTON_LIKE_MODES = ("photons", "VHEE")


class BeamAnglePreset(BaseModel):
    """Named set of gantry and couch angles.

    Attributes
    ----------
    name:
        Unique display name, also the registry key.
    gantry_angles:
        Gantry angles in degrees, one per beam.
    couch_angles:
        Couch angles in degrees, one per beam; defaults to 0° for every beam.
    radiation_modes:
        Radiation modes the preset is typical for; they list it first.  Empty
        means no preference.
    description:
        Short explanation, e.g. for tooltips.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    gantry_angles: tuple[float, ...] = Field(min_length=1)
    couch_angles: tuple[float, ...] = ()
    radiation_modes: tuple[str, ...] = ()
    description: str = ""

    @model_validator(mode="before")
    @classmethod
    def _default_couch_angles(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("couch_angles"):
            data = {**data, "couch_angles": [0.0] * len(data.get("gantry_angles") or ())}
        return data

    @model_validator(mode="after")
    def _check_beam_count(self) -> BeamAnglePreset:
        if len(self.couch_angles) != len(self.gantry_angles):
            raise ValueError("A preset needs one couch angle per gantry angle.")
        return self

    @property
    def num_of_beams(self) -> int:
        return len(self.gantry_angles)

    def prop_stf(self) -> dict[str, list[float]]:
        """Return the ``pln.prop_stf`` entries this preset sets."""
        return {
            "gantry_angles": list(self.gantry_angles),
            "couch_angles": list(self.couch_angles),
        }

    def applies_to(self, generator: type | Any) -> bool:
        """Whether the preset configures the given generator (class or instance)."""
        cls = generator if isinstance(generator, type) else type(generator)
        return issubclass(cls, StfGeneratorExternalBeam)


_PRESETS: dict[str, BeamAnglePreset] = {}


def register_beam_angle_preset(
    preset: BeamAnglePreset | dict[str, Any], *, overwrite: bool = False
) -> BeamAnglePreset:
    """Add a preset to the registry and return it.

    Parameters
    ----------
    preset:
        A :class:`BeamAnglePreset` or a dict of its fields (e.g. from a
        configuration file).
    overwrite:
        Replace an existing preset of the same name instead of raising.
    """
    preset = BeamAnglePreset.model_validate(preset)
    if preset.name in _PRESETS and not overwrite:
        raise ValueError(f"A beam angle preset named {preset.name!r} is already registered.")
    _PRESETS[preset.name] = preset
    return preset


def get_beam_angle_presets(
    radiation_mode: Optional[str] = None, generator: type | Any | None = None
) -> list[BeamAnglePreset]:
    """List the registered presets, those typical for *radiation_mode* first.

    Within the typical, neutral (no preferred mode) and remaining presets the
    registration order is kept.  With *generator*, only presets that apply to
    it are returned.
    """
    presets = [p for p in _PRESETS.values() if generator is None or p.applies_to(generator)]
    if radiation_mode is None:
        return presets

    def rank(preset: BeamAnglePreset) -> int:
        if radiation_mode in preset.radiation_modes:
            return 0
        return 1 if not preset.radiation_modes else 2

    return sorted(presets, key=rank)


def _equidistant(num_beams: int) -> list[float]:
    return [round(i * 360.0 / num_beams, 4) for i in range(num_beams)]


for _preset in (
    BeamAnglePreset(
        name="Single field",
        gantry_angles=[0],
        radiation_modes=_ION_MODES,
        description="One anterior field",
    ),
    BeamAnglePreset(
        name="Lateral opposing",
        gantry_angles=[90, 270],
        radiation_modes=_ION_MODES,
        description="Two opposing lateral fields",
    ),
    BeamAnglePreset(
        name="AP/PA opposing",
        gantry_angles=[0, 180],
        radiation_modes=_PHOTON_LIKE_MODES,
        description="Anterior and posterior opposing fields",
    ),
    BeamAnglePreset(
        name="3 equidistant fields",
        gantry_angles=_equidistant(3),
        radiation_modes=_ION_MODES,
        description="Three fields 120° apart",
    ),
    BeamAnglePreset(
        name="4-field box",
        gantry_angles=_equidistant(4),
        radiation_modes=_PHOTON_LIKE_MODES,
        description="Four orthogonal opposing fields",
    ),
    *(
        BeamAnglePreset(
            name=f"{n}-field IMRT star",
            gantry_angles=_equidistant(n),
            radiation_modes=_PHOTON_LIKE_MODES,
            description=f"{n} equidistant coplanar fields, {360 / n:.4g}° apart",
        )
        for n in (5, 7, 9, 12)
    ),
):
    register_beam_angle_preset(_preset)
