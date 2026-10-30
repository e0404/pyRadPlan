"""Tests for guarantee-flag discipline on PencilBeamEngineAbstract subclasses.

Every *concrete* (fully-implemented, non-abstract) subclass of
PencilBeamEngineAbstract that lives inside the ``pyRadPlan`` package must
define (or inherit from a concrete parent) both class-level boolean flags:

* ``_guarantee_canonical``  – CSC columns are already in sorted, unique-row order
* ``_guarantee_nonzero``    – no structural-zero entries are ever written

This is checked at test time by discovering the complete subclass graph and
filtering to production classes only (module path starts with ``pyRadPlan``).
"""

import importlib
import pkgutil
import sys
from typing import Iterator

import pytest

from pyRadPlan.dose.engines import ParticleHongPencilBeamEngine
from pyRadPlan.plan import create_pln

import pyRadPlan.dose.engines
from pyRadPlan.stf import Beam, IonSpot, Ray, validate_stf


def _import_all_engine_submodules() -> None:
    """Walk every module under pyRadPlan.dose.engines and import it.

    This ensures that all engine classes are defined and therefore show up
    in the ``__subclasses__()`` graph before we collect them.
    """
    pkg = importlib.import_module("pyRadPlan.dose.engines")
    for _finder, name, _ispkg in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        if name not in sys.modules:
            try:
                importlib.import_module(name)
            except Exception:  # noqa: BLE001  – skip engines with missing optional deps
                pass


def _iter_subclasses(cls) -> Iterator[type]:
    """Yield all transitive subclasses of *cls*."""
    for sub in cls.__subclasses__():
        yield sub
        yield from _iter_subclasses(sub)


def _pyradplan_pb_engines():
    """Return all PencilBeamEngineAbstract subclasses in pyRadPlan."""
    from pyRadPlan.dose.engines._base_pencilbeam import PencilBeamEngineAbstract

    _import_all_engine_submodules()

    result = []
    for cls in _iter_subclasses(PencilBeamEngineAbstract):
        # Skip abstract classes (they still have unresolved abstract methods)
        if getattr(cls, "__abstractmethods__", frozenset()):
            continue
        # Skip anything not defined in the pyRadPlan package (e.g. test helpers)
        if not getattr(cls, "__module__", "").startswith("pyRadPlan"):
            continue
        result.append(cls)
    return result


def _stf_with_density_flags(
    use_given_eq_density_cube, ignore_outside_densities, radiation_mode="protons"
):
    """Minimal single-beam stf carrying the generator's density assumptions."""

    ray = Ray(
        ray_pos=[0.0, 0.0, 0.0],
        ray_pos_bev=[0.0, 0.0, 0.0],
        beamlets=[IonSpot(energy=100.0)],
    )
    beam = Beam(
        iso_center=[0.0, 0.0, 0.0],
        rays=[ray],
        radiation_mode=radiation_mode,
        use_given_eq_density_cube=use_given_eq_density_cube,
        ignore_outside_densities=ignore_outside_densities,
    )
    return validate_stf({"beams": [beam]})


@pytest.mark.parametrize(
    "engine_cls",
    _pyradplan_pb_engines(),
    ids=lambda c: c.__qualname__,
)
class TestGuaranteeFlagsPresent:
    """Every discovered engine must carry both guarantee flags."""

    def test_guarantee_canonical_defined(self, engine_cls):
        assert hasattr(engine_cls, "_dij_guarantee_canonical"), (
            f"{engine_cls.__qualname__} (module: {engine_cls.__module__}) "
            "is missing the class variable '_dij_guarantee_canonical'. "
            "See PencilBeamEngineAbstract for the expected semantics."
        )

    def test_guarantee_canonical_is_bool(self, engine_cls):
        val = engine_cls._dij_guarantee_canonical
        assert isinstance(val, bool), (
            f"{engine_cls.__qualname__}._dij_guarantee_canonical must be a plain bool, "
            f"got {type(val)!r}"
        )

    def test_guarantee_nonzero_defined(self, engine_cls):
        assert hasattr(engine_cls, "_dij_guarantee_nonzero"), (
            f"{engine_cls.__qualname__} (module: {engine_cls.__module__}) "
            "is missing the class variable '_dij_guarantee_nonzero'. "
            "See PencilBeamEngineAbstract for the expected semantics."
        )

    def test_guarantee_nonzero_is_bool(self, engine_cls):
        val = engine_cls._dij_guarantee_nonzero
        assert isinstance(val, bool), (
            f"{engine_cls.__qualname__}._dij_guarantee_nonzero must be a plain bool, got {type(val)!r}"
        )


class TestStfDensityConsistency:
    """The engine warns when the stf was built on differently prepared densities."""

    @pytest.fixture
    def engine(self):
        return ParticleHongPencilBeamEngine(
            create_pln({"radiation_mode": "protons", "machine": "Generic"})
        )

    def test_matching_flags_do_not_warn(self, engine, recwarn):
        engine.use_given_eq_density_cube = False
        engine.ignore_outside_densities = False
        engine._check_stf_density_consistency(_stf_with_density_flags(False, False))
        assert not [w for w in recwarn if "densities" in str(w.message)]

    def test_mismatched_ignore_outside_densities_warns(self, engine):
        engine.use_given_eq_density_cube = False
        engine.ignore_outside_densities = False
        with pytest.warns(UserWarning, match="generated with ignore_outside_densities=True"):
            engine._check_stf_density_consistency(_stf_with_density_flags(False, True))

    def test_mismatched_use_given_eq_density_cube_warns(self, engine):
        engine.use_given_eq_density_cube = False
        engine.ignore_outside_densities = True
        with pytest.warns(UserWarning, match="generated with use_given_eq_density_cube=True"):
            engine._check_stf_density_consistency(_stf_with_density_flags(True, True))

    def test_unrecorded_flags_warn_as_unknown(self, engine, recwarn):
        engine._check_stf_density_consistency(_stf_with_density_flags(None, False))
        messages = [str(w.message) for w in recwarn if "stf" in str(w.message)]
        assert len(messages) == 1
        assert "does not record use_given_eq_density_cube" in messages[0]
        assert "ignore_outside_densities" not in messages[0]

    def test_beam_records_nothing_by_default(self):
        beam = _stf_with_density_flags(None, None).beams[0]
        assert beam.use_given_eq_density_cube is None
        assert beam.ignore_outside_densities is None

    def test_photon_beams_are_not_checked(self, engine, recwarn):
        engine.ignore_outside_densities = True
        engine._check_stf_density_consistency(_stf_with_density_flags(None, None, "photons"))
        engine._check_stf_density_consistency(_stf_with_density_flags(False, False, "photons"))
        assert not [w for w in recwarn if "densities" in str(w.message)]
