"""Exporters writing the simulation inputs of a plan for the TOPAS and FRED Monte Carlo codes."""

import logging
import os
from typing import Any, ClassVar, Optional

import numpy as np
import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.cst import ExternalVOI, StructureSet
from pyRadPlan.dose.engines._base_montecarlo import MonteCarloEngineAbstract
from pyRadPlan.dose.engines import ParticleFredMCEngine, ParticleTOPASMCEngine
from pyRadPlan.plan import validate_pln

from ..base import BaseExporter

logger = logging.getLogger(__name__)


def _whole_volume_structure_set(ct: CT) -> StructureSet:
    """Structure set with one external contour around the whole (first phase of the) CT."""
    from pyRadPlan.ct import validate_ct  # noqa: PLC0415

    ct = validate_ct(ct)
    reference = ct.cube_hu[:, :, :, 0] if ct.num_of_ct_scen > 1 else ct.cube_hu
    mask = sitk.Image(reference.GetSize(), sitk.sitkUInt8) + 1
    mask.CopyInformation(reference)
    return StructureSet(vois=[ExternalVOI(name="Body", mask=mask)], ct_image=ct)


class McExporter(BaseExporter):
    """Base of the exporters writing Monte Carlo simulation inputs into a directory.

    The inputs are those the dose engine writes for an external calculation. The engine is
    configured from ``pln`` (``pln.prop_dose_calc`` entries the engine knows, e.g. the dose
    grid, the number of histories or runs, and the biological model), forced to write inputs
    only, and no Monte Carlo code needs to be installed. The objects passed to
    :meth:`save` are not modified.

    Parameters
    ----------
    path : str or os.PathLike
        Output directory.
    overwrite : bool, default False
        If the directory already contains input files of an earlier export,
        :meth:`save` raises :class:`FileExistsError`. With ``overwrite=True`` exactly
        these files are deleted before writing; other files are left alone.
    num_histories : float, optional
        Total number of simulated histories of the plan (the engine's
        ``num_histories_direct``), overriding ``pln.prop_dose_calc``.
    **settings
        Further engine settings overriding ``pln.prop_dose_calc``, e.g. ``num_runs`` for
        TOPAS.

    Notes
    -----
    Spot weights (``weights=`` or ``result["w"]``) are required: the Monte Carlo inputs
    distribute the histories over the spots according to them. Only ion plans are supported.
    The ``cst`` selects the voxels the dose is scored in; without it, the whole CT is.
    """

    container: ClassVar[bool] = False
    supports: ClassVar[frozenset[str]] = frozenset(
        {"ct", "cst", "pln", "stf", "weights", "result"}
    )
    engine_cls: ClassVar[type[MonteCarloEngineAbstract]]

    def __init__(
        self,
        path,
        overwrite: bool = False,
        num_histories: Optional[float] = None,
        **settings: Any,
    ):
        super().__init__(path)
        self.overwrite = bool(overwrite)
        self.settings = dict(settings)
        if num_histories is not None:
            self.settings["num_histories_direct"] = float(num_histories)

    def save(
        self,
        *,
        ct: Optional[CT] = None,
        cst: Optional[StructureSet] = None,
        dose: Optional[sitk.Image] = None,
        pln: Optional[Any] = None,
        stf: Optional[Any] = None,
        weights: Optional[Any] = None,
        result: Optional[dict] = None,
        **extra,
    ) -> None:
        if ct is None or pln is None or stf is None:
            raise ValueError(f"The {self.format} export needs ct, pln and stf.")
        if weights is None and result is not None:
            weights = result.get("w")
        if weights is None:
            raise ValueError(
                f"The {self.format} export needs spot weights (weights= or result['w']) to "
                "distribute the histories over the spots."
            )

        engine = self._configured_engine(pln)
        if cst is None:
            cst = _whole_volume_structure_set(ct)
        existing = engine.existing_input_files(self.path)
        if existing and not self.overwrite:
            raise FileExistsError(
                f"Directory {str(self.path)!r} already contains {len(existing)} "
                f"{self.engine_cls.name} input file(s); pass overwrite=True to replace them."
            )
        for path in existing:
            os.remove(path)

        engine.write_inputs(ct, cst, stf, np.asarray(weights), self.path)

    def _configured_engine(self, pln):
        engine = self.engine_cls()
        pln = validate_pln(pln).model_copy(deep=True)
        if pln.radiation_mode not in engine.possible_radiation_modes:
            raise ValueError(
                f"The {self.format} export supports ion plans ({', '.join(engine.possible_radiation_modes)}), "
                f"not radiation mode {pln.radiation_mode!r}."
            )

        known = set(engine.config_model().model_fields)
        known.update(
            info.alias for info in engine.config_model().model_fields.values() if info.alias
        )
        props = pln.prop_dose_calc if isinstance(pln.prop_dose_calc, dict) else {}
        pln.prop_dose_calc = {
            key: value for key, value in props.items() if key in known or hasattr(engine, key)
        }
        engine.assign_properties_from_pln(pln)
        if self.settings:
            engine.apply_config(self.settings, overwrite_source="exporter options")
        return engine


class TopasExporter(McExporter):
    """Exporter writing TOPAS simulation inputs into a directory.

    Writes ``pyRadPlan_cube.dat`` / ``pyRadPlan_cube.txt`` (CT cube and patient definition),
    ``beamSetup_pyRadPlan_plan_field<N>.txt`` (one per beam, with the spot list) and
    ``pyRadPlan_plan_field<N>_run<M>.txt`` (one per beam and run). The weights select the
    spot histories: ``num_histories_direct / num_runs`` histories are drawn over all spots
    in proportion to the weights, and every run file simulates them. See
    :class:`~pyRadPlan.io.mc.McExporter` for the options.
    """

    format: ClassVar[str] = "topas"
    name: ClassVar[str] = "TOPAS Exporter"
    extensions: ClassVar[tuple[str, ...]] = (".topas",)
    engine_cls = ParticleTOPASMCEngine


class FredExporter(McExporter):
    """Exporter writing FRED simulation inputs into a directory.

    Writes the input tree FRED reads (``fred -f fred.inp -i inp``) below ``inp``:
    ``fred.inp``, ``plan/plan.inp`` (the spots), ``plan/planDelivery.inp``,
    ``regions/regions.inp``, ``regions/hLut.inp`` and the CT ``regions/CTpatient.mhd`` /
    ``.raw``. The weights are the spot weights of the plan (scaled by the engine's
    ``scaling_factor``), and ``num_histories_direct`` histories are distributed evenly over
    the spots. See :class:`~pyRadPlan.io.mc.McExporter` for the options.
    """

    format: ClassVar[str] = "fred"
    name: ClassVar[str] = "FRED Exporter"
    extensions: ClassVar[tuple[str, ...]] = (".fred",)
    engine_cls = ParticleFredMCEngine
