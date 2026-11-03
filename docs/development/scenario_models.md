# Scenario models: design and implementation plan

Status: first pass implemented on branch `feature/secnario_models` (2026-10-22): steps 0 to 4
below are done; dose calculation, the Dij, the quantities and the plan use scenario sets. Robust
optimization strategies, models emitting composite scenarios and adaptive (PCE-driven) drivers
remain open; the hooks for them (`ScenarioBasis`, `ScenarioSet.from_points` / `extend`,
`Dij.extend_scenarios`, the `scenarios=` override of the dose calculation) are in place.

## 1. Goals

- Port matRad's scenario models (`nomScen`, `wcScen`, `impScen`, `rndScen`) with identical
  realizations and probabilities, keeping matRad interop (`pln.multScen` dicts, `dij` cell
  layout).
- Replace matRad's hard-coded three scenario axes (CT phase, setup shift, range) with named
  *uncertainty dimensions* that models combine freely, and that dose calculation and quantities
  query instead of assuming.
- Make scenario generation lazy and the scenario *set* a first-class, immutable value object that
  can be built from explicit points, extended, and handed to a dose engine, so that a later
  driver (polynomial chaos, adaptive sampling) can request scenarios the model did not enumerate
  up front.
- Reserve the interface for scenarios that are *derived* from computed ones (RayStation-style
  linear combinations) without implementing them.

Non-goals of the first pass: robust objectives/strategies, composite dij evaluation, 4D
accumulation, GUI editing beyond model selection and a parameter dialog.

## 2. Current state and defects to retire

`pyRadPlan/scenarios/_base.py` is a near 1:1 port of `matRad_ScenarioModel`: the pydantic
fields (`range_rel_sd`, `shift_sd`, `ct_scen_prob`, `wc_sigma`) trigger an eager
`update_scenarios()` in an after-validator that fills a dozen private arrays (`scen_for_prob`,
`scen_mask` of shape `(ct, shift, range)`, `linear_mask`, `scen_prob`, `scen_weight`). Only
`NominalScenario` exists. Consumers:

| Consumer | What it uses today |
|---|---|
| `dose/engines/_base.py` | `create_scenario_model(mult_scen, ct)`, `dij["num_of_scenarios"] = 1` |
| `dose/engines/_base_pencilbeam.py` | outer loop over `tot_num_shift_scen`, inner over `num_of_ct_scen × tot_num_range_scen`, `sub2scen_ix`, `scen_mask`, `scen_num`, `linear_mask`, `iso_shift`, `rel/abs_range_shift`; containers allocated with `scen_mask.shape` |
| `_fredmc.py`, `_topasmc.py` | allocate with `scen_mask.shape`, refuse `> 1` scenario |
| `stf/generators/_base.py` | `iso_shift` max and `max_abs/rel_range_shift` for the target margin |
| `dij/_dij.py` | object-array containers of any shape, `.flat[i]`; `alphax[:, scenario_index]` |
| `quantities/_base.py` | `scenarios=[0]` index list, `_q_cache` mirrors container shape |
| `optimization/problems/_nonlin_fluence.py` | sums objective values over `q.scenarios` |
| `plan/_plans.py`, GUI plan widget | `mult_scen` field, model combo box with disabled matRad names |

Defects found while reading, to be fixed by the rewrite (each gets a regression test):

1. `_base_pencilbeam.py:197` adds `iso_shift` to the shared `stf` beams in place, so shifts
   accumulate across shift scenarios.
2. `_base_pencilbeam.py:792` and `_base_pencilbeam_particle.py:991` read
   `linear_mask[0][scen_num]`, i.e. row 0 of the mask, not the CT column. Wrong CT phase for any
   scenario but the first.
3. `max_rel_range_shift` returns the max of the *absolute* range shift (copied from matRad).
4. `Dij.get_result_arrays_from_intensity` indexes `alphax[:, scenario_index]`; the column is a
   CT phase, not a scenario. `test_multi_scenario_quantities.py` currently enforces the
   conflation (`alphax` columns == number of scenarios).
5. The stf generators tested `"mult_scen" in pln`, which is never true for a pydantic model, so
   the plan's scenario model never reached beamlet placement; the target margin took the maximum
   of shift, range and beamlet margin where matRad adds them; and a forward dose calculation with
   several scenarios returned slot 0 silently. (Found independently on
   `bugfix/multi_shift_scenarios` and ported from there with its tests.)
6. `NominalScenario.extract_single_scenario` is a stub; `ct_scen_prob` with several phases
   builds `scen_mask` from a list index and silently ignores `num_of_available_ct_scen`.

## 3. Design

### 3.1 Three layers instead of one class

matRad folds three concerns into `matRad_ScenarioModel`. We separate them:

1. **Uncertainty space**: which dimensions are uncertain, with what marginal and joint
   distribution. (`UncertaintyDimension`, `JointDistribution`)
2. **Scenario set**: a concrete, finite list of realizations with probability density, weight and
   a storage index. Immutable, serializable, buildable without a model. (`Scenario`,
   `ScenarioSet`)
3. **Application**: how a compute step perturbs its inputs for a realization (isocenter shift,
   radiological-depth scaling, CT phase selection). Owned by the dose engine / stf generator,
   keyed by dimension *name*, not by axis position.

A `ScenarioModel` is then a pydantic configuration object that owns (1) and a sampling rule, and
produces (2) lazily. Everything downstream consumes (2).

### 3.2 Uncertainty dimensions

```python
class UncertaintyDimension(PyRadPlanBaseModel):
    name: ClassVar[str]                 # "setup", "range", "ct_phase", ...
    components: ClassVar[tuple[str, ...]]
    kind: ClassVar[Literal["continuous", "discrete"]]
    unit: ClassVar[str]

class SetupUncertainty(UncertaintyDimension):      # name "setup", components ("x","y","z"), mm
    sd: tuple[float, float, float] = (2.25, 2.25, 2.25)
    mean: tuple[float, float, float] = (0, 0, 0)

class RangeUncertainty(UncertaintyDimension):      # name "range", components ("abs","rel")
    abs_sd: float = 1.0        # mm
    rel_sd: float = 3.5        # percent, stored as fraction in realizations (matRad: /100)
    combined: bool = True      # matRad combineRange: abs and rel move together on a grid

class CTPhaseUncertainty(UncertaintyDimension):    # name "ct_phase", discrete
    phases: list[int] = [0]
    probabilities: list[float] = [1.0]
```

Later dimensions slot in without touching consumers that do not know them: `HUUncertainty`
(density scaling), `BioUncertainty` (alpha/beta, see 3.8), `MachineUncertainty` (energy/MU). A
consumer that cannot apply a dimension present in the set raises
(`ScenarioSet.require_supported(...)`), so unsupported uncertainties never silently become the
nominal case.

Each dimension declares what it perturbs, so consumers can reason about sharing without knowing
the dimension itself:

```python
class UncertaintyDimension(PyRadPlanBaseModel):
    ...
    affects: ClassVar[frozenset[str]]   # subset of {"geometry", "range", "voxel_params"}
```

`setup` and `ct_phase` affect `"geometry"`, `range` affects `"range"`, `ct_phase` additionally
affects `"voxel_params"` (one CT phase can carry different structures and tissue parameters),
and a future `bio` dimension affects only `"voxel_params"`.

### 3.3 Joint distribution

```python
class JointDistribution(Protocol):
    def pdf(self, points: np.ndarray) -> np.ndarray          # (n, d) -> (n,)
    def sample(self, n: int, rng) -> np.ndarray
    def std(self) -> np.ndarray                               # for grids / worst case
```

`IndependentGaussian` (diagonal covariance from the continuous dimensions' SDs, multiplied by the
discrete phase pmf) reproduces matRad's probability exactly. `MultivariateGaussian(cov)` and a
`Copula(marginals, copula)` adapter are the first extensions; they only need to satisfy the
protocol, the models do not change.

### 3.4 Scenario set

```python
class Scenario(PyRadPlanBaseModel):
    index: int                                  # position in the Dij container (flat)
    values: dict[str, np.ndarray]               # dimension name -> component values
    probability: float                          # joint pdf/pmf at the realization
    weight: float                               # normalized weight used for aggregation
    tags: frozenset[str] = frozenset()          # {"nominal"}, {"worst_case"}, ...
    basis: Optional[ScenarioBasis] = None       # 3.8: derived scenario, not computed

class ScenarioSet(PyRadPlanBaseModel):
    dimensions: tuple[UncertaintyDimension, ...]
    scenarios: tuple[Scenario, ...]
    layout: ScenarioLayout                      # 3.5

    # queries used by consumers
    def __len__, __iter__, __getitem__
    def values(self, dim: str) -> np.ndarray                     # (n, n_components)
    def extent(self, dim: str) -> np.ndarray                     # max |value| per component
    def group_by(self, *dims: str) -> list[tuple[dict, list[int]]]   # shared values -> indices
    def nominal_index(self) -> int
    def computed(self) -> "ScenarioSet"                          # drop derived scenarios
    def probabilities / weights -> np.ndarray
    def ct_phase(self, i: int) -> int                            # 0 if dimension absent
    def dimensions_affecting(self, what: str) -> tuple[str, ...] # e.g. "voxel_params"
    def voxel_param_set(self, i: int) -> int                     # column of alphax/betax/tissue_class_ix
    def voxel_param_sets(self) -> list[tuple[dict, list[int]]]   # == group_by(*dimensions_affecting("voxel_params"))

    # construction without a model
    @classmethod
    def from_points(cls, dimensions, points: dict[str, np.ndarray], distribution=None)
    def extend(self, other: "ScenarioSet") -> "ScenarioSet"      # new indices appended
    def subset(self, indices) -> "ScenarioSet"
```

`group_by` is the replacement for matRad's `sub2scenIx`/`scenMask` loops: the pencil-beam engine
groups by `dimensions_affecting("geometry")` because those change ray tracing, and iterates the
range perturbation inside the group. The engine no longer knows or cares whether the model was
gridded, combined or sampled.

The per-voxel parameter arrays on the Dij (`alphax`, `betax`, `tissue_class_ix`) carry one column
per *voxel parameter set*, i.e. per unique combination of the dimensions that affect
`"voxel_params"`, and `voxel_param_set(i)` maps a scenario to its column. Today that is the CT
phase alone, so the mapping is the identity on `ct_phase`; it is exposed as its own query so that
a biological dimension can be added without touching quantities or the Dij validators.

### 3.5 Storage layout and dimensional indexing

Canonical pyRadPlan storage is **flat**: `Dij` containers have shape `(n_scenarios,)` and
`Scenario.index` addresses them via `.flat[index]`, which is what quantities already do.
`Scenario.index` is the only stable handle; everything persisted or cached refers to it.

Legible, dimensional access is a *derived view* on top, re-tieable at will:

```python
class ScenarioLayout(PyRadPlanBaseModel):
    axes: tuple[tuple[str, ...], ...]   # per axis the tied component keys, e.g.
                                        # (("ct_phase",), ("setup.x","setup.y","setup.z"), ("range.abs","range.rel"))
    coords: tuple[np.ndarray, ...]      # distinct value tuples per axis, first-appearance order, nominal first
    shape: tuple[int, ...]              # len(coords[k]) per axis
    position: np.ndarray                # (n, n_axes) grid subscripts per scenario   (matRad linearMask)
    mask: np.ndarray                    # bool of `shape`, True where a scenario exists (matRad scenMask)

class ScenarioSet:
    layout: ScenarioLayout                              # cached, axes spec from the model
    def with_axes(self, *axes) -> "ScenarioSet"         # re-tie: ("setup",) == all its components; "setup.x" one
    def isel(self, **positions) -> "ScenarioSet"        # set.isel(setup=1, range=0)
    def sel(self, atol=1e-9, **values) -> "ScenarioSet" # set.sel(setup=(2.25, 0, 0), ct_phase=0)
    def linear_index(self, **positions) -> int          # matRad sub2scenIx
    def grid_view(self, container) -> np.ndarray        # object array of `shape`, None off-mask,
                                                        # same matrix objects, no copy
    def to_xarray(self)                                 # optional sugar if xarray is installed
```

Axis specifications name component keys (`"setup.x"`) or whole dimensions (`"setup"`, all
components tied). Models provide a default: gridded models use matRad's three axes, the random
model one `("ct_phase",)` axis and one sample axis tying setup and range. Any tying is valid for
any set: a tying that does not match how the model varied the dimensions only yields a sparse
mask (matRad's random model on the three-axis layout is the N by N diagonal), never an error.

What the view does not promise: grid positions change when a set is extended (coordinates are
appended), value selection is tolerance-based float matching, and a dense grid exists only for a
suitable tying. Engines therefore never iterate axes; they use `group_by` (3.4). matRad export
re-ties to the three matRad axes and writes `grid_view` plus `mask`/`position`; import builds the
flat set from the cell array and keeps the matRad tying as the default layout. If a set carries
a dimension matRad cannot express, export raises.

### 3.6 Scenario models

```python
class ScenarioModel(PyRadPlanBaseModel, ABC):
    name: ClassVar[str]; short_name: ClassVar[str]       # matRad names kept: nomScen, wcScen, ...
    setup: SetupUncertainty = SetupUncertainty()
    range: RangeUncertainty = RangeUncertainty()
    ct_phase: CTPhaseUncertainty = CTPhaseUncertainty()
    wc_sigma: float = 1.0
    distribution: Literal["independent_gaussian"] | JointDistribution = "independent_gaussian"

    def scenarios(self, ct: Optional[CT] = None) -> ScenarioSet      # lazy, cached, invalidated on assignment
    def dimensions(self) -> tuple[UncertaintyDimension, ...]
    @abstractmethod
    def _generate(self, dims, dist) -> ScenarioSet

class NominalScenarioModel      # one scenario per CT phase
class GriddedScenarioModel(ScenarioModel, ABC)   # combinations: none|shift|all, combine_range, grid points
class WorstCaseScenarioModel    # 3 grid points, matRad wcScen
class ImportanceScenarioModel   # n grid points, matRad impScen
class RandomScenarioModel       # n_samples, seed, include_nominal, matRad rndScen
```

Keep the matRad-flat aliases (`rangeRelSD`, `rangeAbsSD`, `shiftSD`, `ctScenProb`, `wcSigma`,
`combinations`, `combineRange`, `numOfSetupGridPoints`, `nSamples`) as `validation_alias` /
before-validators that populate the nested dimensions, so `validate_scenario_model(matrad_dict)`
and `Plan(mult_scen={"model": "wcScen", ...})` keep working. `to_matrad` flattens back.

Laziness: the cache lives in a private attribute; a `__setattr__` override (pydantic
`validate_assignment` is on) clears it. `scenarios()` with a `ct` checks `phases` against
`ct.num_of_ct_scen`. The CT is not stored on the model (the current constructor keeps a count
from it, which goes stale).

### 3.7 Application in dose calculation

Engine base (`_base.py`):

- `scenarios: Union[str, dict, ScenarioModel, ScenarioSet]` (alias `mult_scen`). `_calc_dose_init` resolves it to
  `self._scenarios: ScenarioSet` (`model.scenarios(ct)`, or the set as given), calls
  `self._scenarios.require_supported(self.supported_uncertainties)` with a class-level tuple
  (`("ct_phase", "setup", "range")` for pencil beams, `("ct_phase",)` for the MC engines), and
  records `dij["scenarios"] = self._scenarios`, `dij["num_of_scenarios"] = len(...)`.
- Container allocation uses `(len(scenarios),)`; derived scenarios are skipped (`computed()`).

Pencil beam (`_base_pencilbeam.py`):

```python
for geo_values, scen_ids in self._scenarios.computed().group_by("ct_phase", "setup"):
    scen_stf = stf.model_copy(deep=True); shift every beam.iso_center by geo_values["setup"]
    for beam ...: init beam (per CT phase rad depths as today)
        for ray ...:
            for s in scen_ids:
                scen = self._scenarios[s]
                scen_ray = self._apply_scenario_to_ray(ray, scen)   # ct phase select + range scaling
                bixels -> self._fill_dij(..., scen.index, ...)
```

`_apply_scenario_to_ray` replaces `_extract_single_scenario_ray(ray, scen_idx)`; the particle
subclass override picks `v_alpha_x` etc. by `scen.ct_phase`. `_fill_dij` and `_finalize_dij`
index `.flat[scen.index]`. stf generators replace the `iso_shift` / `max_*_range_shift` reads
with `set.extent("setup")` and `set.extent("range")`.

### 3.8 Forward-compatible hooks (interface only in this pass)

- **Adaptive sets.** `calc_dose_influence(ct, cst, stf, scenarios=ScenarioSet)` overrides the
  plan model; `Dij.extend_scenarios(other_dij)` concatenates containers of two Dijs with
  identical bixel bookkeeping and merges their sets. A PCE driver then loops: build points,
  `ScenarioSet.from_points`, compute, extend. The pass implements `from_points`, `extend` and
  `Dij.extend_scenarios` with tests, but no driver.
- **Derived scenarios.** `Scenario.basis: ScenarioBasis(coefficients: dict[int, float])` marks a
  scenario whose quantity is `Σ c_k q_k` of computed scenarios. Engines skip it; the quantity
  base class resolves it in `_compute_quantity_single_scenario` by linear combination of the
  computed ones. Only the field, validation (coefficients reference computed indices) and the
  quantity-level combination are implemented; no model generates such scenarios yet. This is the
  hook for RayStation-style worst-case shortcuts and ties into the planned Dij-level caching.
- **Biological scenarios.** A `BioUncertainty` dimension (`affects = {"voxel_params"}`) is kept
  separate from the CT phase: it changes no geometry and, on the indirect quantity path
  (`alpha_x * dose`), not even an influence matrix. Its realizations are either discrete
  alternative tissue parameter sets per structure or a continuous perturbation of alpha/beta
  (ratio or model parameters); `StructureSet.get_reference_lq_params` and
  `get_tissue_class_indices` gain a per-voxel-parameter-set argument that applies the perturbation
  and fills the extra columns. Scenarios differing only in `bio` share their physical dose slot
  through `Scenario.basis` with a single unit coefficient, so no matrix is duplicated. Engines with
  direct alpha/beta kernel paths (particle pencil beam with LEM-type base data) list `"bio"` in
  `supported_uncertainties` and select `v_alpha_x` by `voxel_param_set`; engines without it refuse
  the dimension. Only the `affects` declaration, the `voxel_param_set` mapping and the slot sharing
  are implemented in this pass.
- **Shared index storage.** `Dij.share_scenario_structure()` (applied by the engines when
  `share_scenario_structure=True`, off by default) rewrites the matrices of all computed scenarios and quantities
  over the union of their sparsity patterns: one int32 `indices`/`indptr` pair for the whole Dij
  and, per quantity, a `(matrices, nnz)` value block whose rows are the scenarios' data arrays
  (explicit zeros where a scenario misses a voxel). The union is one stable argsort of the
  concatenated `col * n_rows + row` keys (timsort merges the presorted runs), the positions come
  from the rank of each key scattered back through the permutation; no `searchsorted`, so the cost
  is linear, ~50 ns per stored entry on the CPU (8.5 s for 9 x 20M). `to_namespace` uploads the
  shared index arrays once. The block is the storage a later fused multi-scenario matvec / transposed product (one
  gather of the fluence per nonzero, one FMA per scenario; voxel-wise worst case and derived
  scenarios as elementwise operations on the block) would work on; that step is not implemented.
- **Robust optimization.** The optimization problem keeps summing over `q.scenarios`; the
  weights and tags on the set are what expected-value / worst-case strategies will consume. No
  change there in this pass beyond defaulting `scenarios=None` to all computed scenarios of the
  Dij.

### 3.9 Why this and not configurable axes

Making the axis assignment of each error configurable (the original idea) still forces every
consumer to reason about a grid. Most of the flexibility wanted (collapsing dimensions, joint
sampling, copulas) is a property of the *sampling and distribution*, and the only thing a dose
engine needs to know is which scenarios share expensive inputs. `group_by` gives exactly that
for any sampling scheme, including fully correlated random samples where no grid exists at all.
Axes survive only as an optional export layout. The dimension concept is kept because it is what
makes the set self-describing (an engine can refuse a dimension it cannot apply, a generator can
compute margins per dimension, a distribution can be swapped per dimension).

## 4. Public API summary

```python
from pyRadPlan.scenarios import (
    ScenarioModel, NominalScenarioModel, WorstCaseScenarioModel, ImportanceScenarioModel,
    RandomScenarioModel, ScenarioSet, Scenario, SetupUncertainty, RangeUncertainty,
    CTPhaseUncertainty, create_scenario_model, validate_scenario_model, available_scenario_models,
)
model = WorstCaseScenarioModel(setup=SetupUncertainty(sd=(3, 3, 3)), combinations="all")
scen_set = model.scenarios(ct)
dij = calc_dose_influence(ct, cst, stf, pln)            # pln.scenarios = model (alias mult_scen)
dij.scenarios                                            # the ScenarioSet, index == container slot
dose = PhysicalDose(dij)                                 # all computed scenarios by default
```

`NominalScenario` stays as a deprecated alias of `NominalScenarioModel` for one release.

## 5. Integration points (file by file)

| File | Change |
|---|---|
| `scenarios/_dimensions.py` (new) | dimension models |
| `scenarios/_distributions.py` (new) | `JointDistribution`, `IndependentGaussian`, `MultivariateGaussian` |
| `scenarios/_set.py` (new) | `Scenario`, `ScenarioBasis`, `ScenarioLayout`, `ScenarioSet` |
| `scenarios/_base.py` | rewrite `ScenarioModel` (lazy, nested dims, matRad aliases) |
| `scenarios/_nominal.py`, `_gridded.py`, `_worst_case.py`, `_importance.py`, `_random.py` | models |
| `scenarios/__init__.py` | registry, factory, matRad dict validation, 1-based `ctScenProb` conversion |
| `dij/_dij.py` | `scenarios: Optional[ScenarioSet]` field; container-size validator; `alphax`/`betax`/`tissue_class_ix` columns validated against the number of voxel parameter sets and selected via `voxel_param_set(i)`; shared slots for scenarios with a unit-coefficient basis; `extend_scenarios`; `share_scenario_structure` (union sparsity pattern, one index array for all scenarios and quantities); matRad export via layout |
| `dose/engines/_base.py` | resolve set, `supported_uncertainties`, `dij["scenarios"]`, optional `scenarios=` argument on the public calc functions |
| `dose/engines/_base_pencilbeam.py`, `_base_pencilbeam_particle.py` | loop restructure (3.7), fix defects 1 and 2, allocation and finalize over flat container |
| `dose/engines/_fredmc.py`, `_topasmc.py` | allocation shape, `supported_uncertainties = ("ct_phase",)` |
| `stf/generators/_base.py` | parameter `scenarios` (alias `mult_scen`), margins from `extent` |
| `quantities/_base.py` | default scenarios from `dij.scenarios`, derived-scenario combination, LQ paths select `alphax[:, set.voxel_param_set(i)]` |
| `optimization/problems/_optiprob.py`, `_nonlin_fluence.py` | no behavior change; `_mult_scen` becomes `_scenarios` |
| `plan/_plans.py` | field renamed to `scenarios` with `mult_scen` / `multScen` as validation aliases and `multScen` on matRad export; accepts model / dict / str / set |
| `gui/widgets/plan/_plan_widget.py` | read/write `pln.scenarios`, enable all models, `[…]` parameter dialog (`ScenarioModel` is a pydantic model; reuse `ConfigFormDialog` if it takes a model class, else a small adapter) |
| `io` matlab | `multScen` dict round trip, `dij` cell array ↔ flat container + layout |
| `docs/user_guide/concepts/dose_calculation.rst`, new `uncertainty.rst`, `CHANGELOG.md` | documentation |

## 6. Validation

- **matRad references.** Generate reference scenario tables with
  `matlab -batch` from `D:\MATLAB\matRad` for each model and option combination
  (`combinations` × `combineRange` × grid points × `ctScenProb` with 2 phases): `scenForProb`,
  `scenProb`, `scenWeight`, `linearMask`, `scenMask`. Store as small `.mat`/`.npz` fixtures under
  `test/data`. Tests assert realizations, probabilities and the export layout match.
- **Random model.** Same seed cannot match MATLAB's `randn`; test distribution moments, nominal
  inclusion, weights, and that `pdf` of the samples equals `IndependentGaussian.pdf`.
- **Dose calculation.** Nominal results on the existing photon/proton/helium fixtures must be
  bit-identical to before (no regression). A worst-case run on a small phantom checks: the
  nominal slot equals the nominal-only run; a pure setup shift equals a nominal run with the
  isocenter moved; a pure absolute range shift equals a nominal run with the rad depths offset.
  A two-phase CT test checks the correct phase per scenario (defect 2).
- **Quantities / optimization.** `test_multi_scenario_quantities.py` rewritten so `alphax`
  columns count voxel parameter sets (CT phases today), with a test where two scenarios share
  one physical dose slot but use different columns, which is the shape a biological scenario
  will take; a derived-scenario test checks the linear combination against an
  explicit sum; a `from_points` + `extend_scenarios` test checks the adaptive path.
- **GUI.** `pytest test/gui` plus a widget test that switching the model updates `pln.mult_scen`.
- Run `pytest` fully after engine changes (CLAUDE.md) and the ruff 0.15.22 diff check.

## 7. Work breakdown and ordering

Step 0 is sequential and defines the contract; steps 1 to 3 run in parallel as subagents in
isolation worktrees; step 4 integrates.

**Step 0 (one agent, first):** `_dimensions.py`, `_distributions.py`, `_set.py`, the rewritten
`ScenarioModel` base with `NominalScenarioModel`, factory and alias handling, unit tests for the
set API (`group_by`, `extent`, `from_points`, `extend`, `with_axes`/`isel`/`sel`/`grid_view`,
layout round trip against matRad's `scenMask`/`linearMask` convention). This is the interface
everything else codes against; it should be reviewed before fanning out.

**Step 1 — models (agent A):** gridded base, worst-case, importance, random; matRad reference
fixtures via `matlab -batch`; probability/weight tests; `to_matrad` of models.

**Step 2 — Dij and quantities (agent B):** `Dij.scenarios`, validators, `ct_phase` column
selection, `extend_scenarios`, matRad cell/mask ↔ flat conversion in the Dij validators and
serializers, quantity defaults and derived-scenario combination, rewritten multi-scenario
quantity tests.

**Step 3 — dose engines and stf (agent C):** engine base resolution and `supported_uncertainties`,
pencil-beam loop restructure with the defect fixes, MC engine guards, stf margins, the
phantom-based scenario tests in section 6. Depends on step 0 only; uses a temporary
`dij["scenarios"]` until step 2 lands, then rebases.

**Step 4 — integration (one agent):** plan model, GUI plan widget, matlab IO round trip test,
docs and changelog, full test suite, GPU smoke via `docs/execute_examples.py` for one proton
example (the GUI runs cupy).

Robust optimization (expected value, voxel/objective/composite worst case, `Variance`) landed in
the pass described in `robust_optimization.md`; `calc_dose_forward` returns per-scenario doses
(`<quantity>_scenario` lists next to the nominal result). Still open: robust constraints (the
term model is ready), Dij-level
backward accumulation, matRad reference values for the robust objectives, a model that emits
derived scenarios, a PCE driver using `from_points`/`extend_scenarios`, the `BioUncertainty`
dimension with its CST hook, dimensions beyond that (HU, machine), and the fused multi-scenario
kernels on the shared value block.

CT phases in the pencil-beam engines (2026-11-01): the SSD, and the initial particle spot width
and air-gap depth offset derived from it, are computed per CT phase (`ray["SSD"]` etc. are lists
over the CT's phases, selected in `_apply_scenario_to_ray`); the reference tissue parameter columns
are addressed through `_voxel_param_phases` (one column per voxel parameter set, not per phase).
Per-beam approximations that need one SSD (particle lateral cutoff LUT, photon SVD kernel
selection) use `_reference_ct_phase`, the lowest computed phase, as matRad uses the first.

Review round of 2026-11-01 (PR #26): `ScenarioSet.reference_indices()` is the nominal scenario
or, without one, the first computed scenario per CT phase; `Dij.nominal_scenario_index()` and
`NoRobustness` use it. `matrad_layout` addresses the CT axis by phase index (one row per phase,
used or not), and the cell import accepts more rows than the set uses. The engines resolve
scenario dictionaries through `validate_scenario_model` (1-based `ctScenProb` like the plan), and
`_nominal_only_dimensions` treats a non-zero mean (setup offset) as a perturbation. The Monte
Carlo engines simulate the CT phase of their single scenario (`_scenario_ct`). 4D workflow:
`PyRadPlanBaseModel.__eq__` compares images by geometry and voxel values, `create_cst` attaches
a structure set to a CT on the same 3D grid (`resample_on_new_ct` repeats 3D structures over the
phases of a 4D CT and resamples 4D structures phase by phase; PR #13 made the overlap priorities
and tissue columns 4D-aware), the engines take per-phase voxel lists (`_phase_voxel_lists`), the
optimization evaluates each scenario on its phase's contour (`RobustTerm.scenario_masks`,
operators with `phase_contours`; the others read the first scenario's phase), and the stf
generators work on `_geometry_ct`, the first phase. Still open: matRad export writes
`max phase + 1` CT rows (the Dij knows no CT) and all phases are ray-traced even when one is used.

## 8. Decisions (confirmed 2026-10-22)

1. Flat canonical container layout; dimensional access through the derived, re-tieable
   `ScenarioLayout` view (`with_axes`, `isel`, `sel`, `grid_view`). Only the flat index is stable.
2. The plan field, engine and generator parameters are named `scenarios`; `mult_scen` (and
   `multScen` from matRad) stay as validation aliases, and matRad export writes `multScen`.
3. `NominalScenario` and the matRad-flat field aliases are kept for one release, then removed.
