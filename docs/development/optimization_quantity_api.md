# Optimization quantity API: implementation plan

Status: proposed design for review and later implementation. No runtime API described as
proposed below has been implemented as part of this document.

This plan is based on inspection of the repository. Recheck the referenced implementation
before starting work, since the code may have changed in the meantime.

## 1. Scope and semantic contract

Introduce an objective-only pseudo-quantity, `dose_auto`, and explicitly configured dose
optimization strategies. Preserve `QuantityResolver` as the owner of computed-quantity
construction and dependency resolution.

Required invariants:

1. Concrete objective quantities always retain their meaning and identifier.
2. Only `quantity="dose_auto"` authorizes biological prescription conversion.
3. A strategy must be explicitly selected when an automatic objective is present.
4. Strategies specify both a concrete target quantity and reference-parameter conversion.
5. Original objectives, images, arrays, and caches in the user's `StructureSet` are not mutated.
6. Invalid or ambiguous preparation fails before solver execution; there is no physical-dose
   fallback for an unavailable biological quantity.
7. Fraction normalization has an explicit ordering and quantity-specific semantics.
8. Reporting defaults do not select optimization strategies.

**Literal quantity selection bypasses biological conversion, not fraction normalization.**
An explicit `effect` reference is already an effect value. Under `dose_convention="total"`,
it may be normalized to per-fraction effect; it must never be interpreted as dose and passed
through an LQ transformation.

Out of scope:

- Dynamic influence-matrix storage on `Dij`; see [Dynamic quantities on Dij](dynamic_dij_quantities.md).
- A redesign of biological models or their evaluators.
- Per-objective strategy overrides in the first release.
- Variable fraction doses, time-dependent biological effects, and general course optimization.
- Scalar-to-voxel-reference expansion and scenario-dependent reference images in the first release.
- Automatic objective-weight retuning or equivalence between dose-space and effect-space penalties.

## 2. Repository findings

| Location | Current behavior relevant to this change |
|---|---|
| [`_optiprob.py`](../../pyRadPlan/optimization/problems/_optiprob.py) | Defaults `convert_dose_objectives` to `True`; `_collect_objectives()` overwrites quantities using modality and available matrices; makes shallow objective copies; preprocesses images before scalar fraction normalization. |
| [`_objective.py`](../../pyRadPlan/optimization/objectives/_objective.py) | Defaults quantity to `physical_dose`; validates against the concrete registry; divides all scalar `reference` parameters by fractions; skips `image_reference` parameters during normalization. |
| [`_resolver.py`](../../pyRadPlan/quantities/_resolver.py) | Resolves dependencies, deduplicates instances, and reports unavailable computation paths. |
| [`_plans.py`](../../pyRadPlan/plan/_plans.py) | Stores optimization configuration in `prop_opt`; defaults `dose_convention` to `per_fraction`; provides `result_dose_factor`. |
| [`test_dose_convention.py`](../../test/optimization/test_dose_convention.py) | Covers scalar dose normalization and repeat-run source immutability, but not biological conversion, images, or mixed quantities. |
| [`_dij.py`](../../pyRadPlan/dij/_dij.py) | Result scaling distinguishes intensive, additive, and square-root-extensive quantities. |
| [`_cst.py`](../../pyRadPlan/cst/_cst.py) | Builds reference LQ fields with overlap handling and nearest-neighbor resampling; currently repeats tissue fields across CT scenarios. |
| [`_rbe_x_dose.py`](../../pyRadPlan/quantities/_rbe_x_dose.py) | Chooses constant-RBE or effect-inversion paths; reference-parameter lookup has a scenario-handling TODO. |
| [`_optimization_widget.py`](../../pyRadPlan/gui/widgets/optimization/_optimization_widget.py) | Lists concrete registry quantities; replacing objective type currently creates a fresh default objective; reference-image choices are broadly drawn from results. |
| [`Objective.to_matrad()`](../../pyRadPlan/optimization/objectives/_objective.py) | Writes class name, parameters, and penalty, losing quantity intent. |

Two adjacent initialization issues must be addressed where this work touches them:

- `assign_properties_from_pln()` currently pops `opti_prob` from the plan's property dictionary;
  preparation should read a copy.
- `_initialize()` increments the objective index inside a loop over resolved quantities.
  Verify and correct objective counting/mapping for mixed roots and transitive dependencies.

## 3. Public API and defaults

### Objective intent

Keep `Objective.quantity` as a string accepting concrete registered identifiers plus the reserved
objective-only identifier `dose_auto`. Do not register `dose_auto` in `QUANTITIES`, expose it in
result viewers, or pass it to `QuantityResolver`.

Defaults:

- Omitted native objective quantity remains literal `physical_dose`.
- Serialized `quantity="physical_dose"` always remains literal physical dose.
- Omitted strategy means no automatic strategy is selected.
- An automatic objective without a strategy raises a preparation error.
- `quantity=None` is not an alias for automatic behavior.

### Strategy selection

Store the selection at `Plan.prop_opt["dose_optimization_strategy"]`. Accept a registered name
or a configuration dictionary; canonical serialization uses a dictionary with a `strategy` key.

Expose the same setting on `PlanningProblem` for direct programmatic construction. A problem
constructed from a plan takes a validated snapshot, without introducing a separate implicit
strategy default. Record the effective configuration in the preparation report.

Do not add a top-level `Plan` field or per-objective strategy field initially. A single automatic
strategy can coexist with any number of literal quantity objectives.

### Initial strategies and prescription bases

| Strategy | Input prescription meaning | Concrete quantity | Biological transform |
|---|---|---|---|
| `physical_dose` | Physical dose in Gy | `physical_dose` | Identity |
| `rbe_x_dose` | RBE-weighted dose in Gy(RBE) | `rbe_x_dose` | Identity |
| `lq_effect` | Reference-radiation dose in Gy for the selected reference tissue | `effect` | Reference LQ transformation |

For `rbe_x_dose`, do not multiply the prescription by model RBE: the computed delivered quantity
already includes that weighting. A future strategy accepting physical dose and converting it to
an RBE-weighted prescription must explicitly declare that different input basis. Variable RBE
generally cannot be determined from a scalar physical-dose prescription alone.

Changing strategy changes the interpretation of automatic prescriptions. The GUI must display
the input basis as well as the resulting quantity.

### Proposed usage

```python
from pyRadPlan.optimization.objectives import SquaredDeviation

pln.prop_opt["dose_optimization_strategy"] = "physical_dose"

automatic = SquaredDeviation(quantity="dose_auto", d_ref=60.0)
literal = SquaredDeviation(quantity="physical_dose", d_ref=60.0)
```

```python
pln.prop_opt["dose_optimization_strategy"] = {"strategy": "rbe_x_dose"}

# The input prescription is 60 Gy(RBE).
automatic = SquaredDeviation(quantity="dose_auto", d_ref=60.0)
```

```python
pln.dose_convention = "total"
pln.num_of_fractions = 30
pln.prop_opt["dose_optimization_strategy"] = {
    "strategy": "lq_effect",
    "reference": {
        "source": "explicit",
        "alpha_x": 0.1,  # Gy^-1; illustrative values
        "beta_x": 0.05,  # Gy^-2
    },
}

automatic = SquaredDeviation(quantity="dose_auto", d_ref=60.0)
# Prepared reference: 0.1 * 2 + 0.05 * 2**2 = 0.4 effect per fraction.

literal = SquaredDeviation(quantity="effect", d_ref=12.0)
# Prepared reference: 12 / 30 = 0.4 effect per fraction.
# No strategy invocation or LQ conversion.
```

```python
pln.prop_opt["dose_optimization_strategy"] = {
    "strategy": "lq_effect",
    "reference": {"source": "voi_homogeneous"},
}
```

## 4. Strategy registry and interfaces

Create `pyRadPlan/optimization/strategies/`, separate from biological-model and quantity registries.
Use stable identifiers, reject duplicate registration, and validate each strategy configuration
with a small Pydantic model. No automatic plugin discovery is needed initially.

Proposed registry surface:

```python
register_dose_optimization_strategy(strategy_class)
get_dose_optimization_strategy(spec)
get_available_dose_optimization_strategies()
```

Illustrative interface sketch, not a complete implementation:

```python
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Protocol


@dataclass(frozen=True)
class ResolutionContext:
    plan: Plan
    dij: Dij
    voi: VOI                  # Effective optimization VOI
    voxel_indices: Array
    scenario_indices: tuple[int, ...]
    # Helpers expose aligned tissue fields and validate their provenance.
    # These objects are read-only inputs by contract, not deep-frozen models.


class ReferenceTransform(Protocol):
    input_semantics: str
    output_semantics: str
    mapping_kind: str         # identity, linear, nonlinear

    def apply(self, values: Any) -> Any:
        """Transform per-fraction scalar/array references without mutating inputs."""


@dataclass(frozen=True)
class ResolutionRecipe:
    quantity: str
    transform: ReferenceTransform
    provenance: Mapping[str, Any]


class DoseOptimizationStrategy(Protocol):
    identifier: ClassVar[str]

    def resolve(
        self,
        objective: Objective,
        context: ResolutionContext,
    ) -> ResolutionRecipe:
        """Validate compatibility and describe conversion without changing inputs."""
```

The preparation coordinator applies the recipe and returns a validated resolved objective and
an audit record. Central ownership of copying, ordering, normalization, and cache finalization
prevents strategy implementations from each handling these differently.

The generic protocol has no alpha/beta or RBE fields. Those belong to strategy-specific
configuration and transforms. All automatic inputs are dose prescriptions on the plan's declared
fraction basis; their radiation/reference meaning is declared by the selected strategy.

Strategies inspect the actual `Dij`, selected biological model, VOI, and scenario context, but do
not construct computed quantities. Final computability is checked through one shared
`QuantityResolver`; additional data/shape/scenario validation should retain that dependency path
in its errors rather than reimplementing the graph.

`BiologicalModel.default_report_quantity` must not select the strategy. Narrow its current
documentation mentioning display and planning to reporting only.

## 5. Objective reference conversion

Retain `ParameterMetadata.kind` values. Treat both `reference` and `image_reference` as
quantity-valued references while retaining their scalar/image distinction.

Introduce an objective conversion hook conceptually equivalent to:

```python
def transform_reference_parameters(self, transform: ReferenceTransform) -> Objective:
    """Return a validated copy, or reject an unsupported transformation."""
```

Shared helpers may traverse reference fields, but metadata alone must not authorize nonlinear
conversion. Require explicit support from an objective class or a documented adapter. Default
to rejecting unknown nonlinear transformations.

| Objective | Identity / fraction normalization | Initial `lq_effect` support |
|---|---|---|
| Squared deviation, underdosing, overdosing | Supported | Scalar homogeneous or explicit reference tissue |
| Squared mimicking | Supported, including image reference | Pointwise conversion after spatial alignment |
| Min/Max DVH | Supported | One common monotonic reference mapping |
| Mean dose | Supported | Reject initially |
| EUD | Supported | Reject initially |
| Dose uniformity | Supported | Reject automatic nonlinear conversion initially |
| Third-party objective | Declared reference semantics | Explicit opt-in/override required |

Transforming a mean-dose reference does not generally yield an equivalent mean-effect reference.
The same issue applies to EUD. Even objectives without reference parameters require an explicit
decision when changing quantity: minimizing effect variation is different from minimizing dose
variation.

Supported conversion defines a penalty in the target quantity's space. It does not preserve the
original penalty values, gradients, optimum, or relative weight between objectives. Preserve
priorities and all non-reference parameters; document that weights may require retuning.

## 6. Preparation flow and fraction ordering

Preparation is a deterministic, repeatable operation shared by optimization and GUI preview.
Publish prepared runtime state only after the entire preparation succeeds.

1. Snapshot plan configuration and parse/copy objective definitions. Give mutable reference
   values independent ownership and clear runtime image caches. A shallow `model_copy()` is
   insufficient for this contract.
2. Prepare geometry: apply configured overlap priorities, resample CT/structures to the dose
   grid, and establish effective VOI indices.
3. Build context with fraction convention/count, selected plan model, actual `Dij` model
   metadata, effective VOI, and supported scenario mapping.
4. Resolve intent: concrete objectives bypass strategies; automatic objectives require a
   configured strategy and obtain a resolution recipe.
5. Spatially align source reference images to the optimization grid. Align reference tissue
   fields using the existing tissue-field machinery.
6. Normalize source references to one fraction. Automatic inputs use dose normalization;
   literal inputs use the concrete quantity's declared fraction rule.
7. Apply biological conversion only for automatic objectives, through the objective hook;
   assign the concrete quantity and revalidate the resolved objective.
8. Finalize flattened, VOI-indexed reference-image caches from the transformed references.
   Avoid another resampling operation.
9. Resolve all concrete roots through one `QuantityResolver`, validate required scenario data,
   and construct objective-to-quantity mappings.
10. Publish the objective list, quantity graph, and preparation report; configure the solver.

Split the existing image preprocessing operation into spatial alignment and final cache creation.
Nonlinear conversion and interpolation generally do not commute:

```text
T(resample(D)) != resample(T(D))
```

For total dose `D` and `N` identical fractions:

```text
d = D / N
effect_per_fraction = alpha_x * d + beta_x * d**2
effect_total = N * effect_per_fraction
```

Do not use `(alpha_x * D + beta_x * D**2) / N`.

For literal references, use the inverse of existing result-scaling conventions:

| Concrete quantity | Total-course reference to per-fraction reference |
|---|---|
| `physical_dose`, `rbe_x_dose` | Divide by `N` |
| `effect`, `alpha_dose` | Divide by `N` |
| `sqrt_beta_dose` | Divide by `sqrt(N)` |
| `let` | Unchanged |
| `let_dose` | Divide by `N` |

Apply the same rules to scalar and image references. Leave `numeric`, `relative_volume`,
priorities, and other non-reference values unchanged. Never normalize the converted effect a
second time after source dose normalization.

Declare fraction semantics on computed quantity classes through small metadata or a method.
Unknown custom quantities must declare a rule when total-course normalization is needed;
never silently apply `/N`. This concerns quantity semantics, not dynamic `Dij` storage.

The supported model assumes identical fractions. Do not imply support for varying fraction doses
or time-dependent effects.

## 7. Reference tissue and scenarios

Require an explicit `reference.source` for `lq_effect`.

### `voi_homogeneous`

Use reference fields on the effective optimization support, after the appropriate overlap and
geometry handling. Require common finite alpha and beta across that support and all relevant
supported scenarios, within documented tolerances. Do not inspect only the scalar attributes
of the original VOI and assume homogeneity.

Check consistency between CST-derived fields and the biological tissue fields used by `Dij`.
Do not average heterogeneous coefficients, pick the first voxel or dominant tissue, or silently
select the nominal scenario.

### `explicit`

Use configured absolute `alpha_x` and `beta_x` as the reference-radiation prescription tissue.
The reference may intentionally differ from local tissue parameters used to compute delivered
effect; record that distinction in the preparation report. This does not authorize stale or
inconsistent tissue fields within the dose computation itself.

An alpha/beta ratio alone is insufficient to compute effect. Both absolute coefficients are
needed. Require nonnegative coefficients with at least one positive for the proposed monotonic
LQ transformation.

### Deferred voxel-wise references

A future explicit policy could define:

```text
effect_reference[i] = alpha_x[i] * dose + beta_x[i] * dose**2
```

Most existing scalar objectives declare `float` reference fields. Do not insert arrays into those
fields or silently replace an objective with `SquaredMimicking`. Voxel-wise support needs
deliberate objective schemas/hooks and tests.

Initially support existing mimicking images with homogeneous or explicit reference tissue, and
reject scalar-to-image expansion or scenario-dependent reference images.

### Current scenario limits

Audit the relationship between optimization scenarios and CT-scenario columns before enabling
multi-scenario conversion. Current CST fields repeat across CT scenarios and RBE reference lookup
contains a TODO. Reject unsupported mappings or scenario-varying conversion requirements rather
than using scenario zero. Do not claim robust scenario support from the strategy interface alone.

## 8. Validation and diagnostics

Introduce `ObjectiveResolutionError`, preferably a `ValueError` subclass, identifying VOI,
objective index/name, requested quantity, strategy, and the failing field/dependency.

Example messages:

```text
VOI 'PTV', objective 2 ('Squared Deviation'):
quantity='dose_auto' requires prop_opt.dose_optimization_strategy.
```

```text
VOI 'PTV', objective 2:
lq_effect with reference.source='voi_homogeneous' found heterogeneous
reference parameters. Select an explicit reference tissue.
```

Fail before solver execution for:

- Unknown strategy/quantity, invalid options, or an unresolved pseudo-quantity.
- Unsupported objective transformation or reference shape.
- Invalid fraction count or nonfinite/invalid prescription and tissue parameters.
- Invalid image geometry, missing reference coverage, or incompatible scenario mappings.
- Missing quantity dependencies or missing matrix data for a requested scenario.
- An incompatible selected biological model and strategy.
- Conflicting known model configuration between `Plan` and `Dij`.
- Inconsistent CST/Dij tissue fields when the biological computation depends on them.

Promote the current logged tissue inconsistency to an error on affected biological paths.
An unused biological strategy must not block valid literal physical-dose optimization merely
because it is incompatible; validate runtime compatibility when the strategy is actually used.

Legacy `Dij` objects with missing model provenance can still support literal quantities through
`QuantityResolver`. Automatic biological conversion requires enough model/reference provenance
from a documented import adapter or explicit configuration. Do not infer a strategy from matrix
presence alone.

No fallback to physical dose is permitted for failed effect or RBE-weighted requests.

The preparation report should include original and resolved quantities, strategy configuration,
input prescription basis, fraction conversion, reference tissue source, transformed scalar values
or image summaries, and model/dependency provenance. Keep arrays/callables out of the serialized
report unless explicitly supported by the artifact format.

## 9. Backward compatibility and migration

Preserving the old overwrite behavior is incompatible with literal quantity semantics.

| Existing input | Proposed behavior |
|---|---|
| Native objective omits quantity | Literal `physical_dose` |
| Serialized `quantity="physical_dose"` | Literal physical dose, including ion plans |
| Explicit effect, LET, or other concrete quantity | Retains that quantity |
| Ion plan relying on implicit overwrite | Requires explicit migration |
| `dose_auto` without strategy | Preparation error |

Do not use Pydantic `model_fields_set` to infer intent at optimization time. Ordinary serialization
can erase that distinction and must not change the meaning of an objective.

### Legacy flag

- Remove the implicit `convert_dose_objectives=True` behavior.
- During one deprecation period, accept explicitly supplied `False` with a warning; it grants
  no conversion authority.
- Reject explicitly supplied `True` with an actionable migration error. Silently ignoring it
  could unexpectedly change existing ion optimization.
- Reject contradictory combinations of legacy and new settings.
- Remove the flag after the announced deprecation period.

### Migration helper

Provide an explicitly invoked helper returning copied plans/structures and a change report.
It may preview what the old modality/matrix heuristic would select, but must require a chosen
strategy and selected objectives before rewriting them to `dose_auto`.

Where historical intent is known, callers may instead migrate to literal `rbe_x_dose`.
Do not automatically classify serialized physical-dose objectives as intentional physical dose
versus artifacts of an old default.

Release notes must state that new defaults preserve the stored physical-dose interpretation,
not the old implicit biological optimization outcome for ion plans.

## 10. GUI integration

Add an objective-selector helper distinct from the concrete quantity registry:

| Label | Stored identifier |
|---|---|
| Automatic dose - use plan strategy | `dose_auto` |
| Physical dose - Gy | `physical_dose` |
| Effect - dimensionless | `effect` |
| RBE-weighted dose - Gy(RBE) | `rbe_x_dose` |
| Other concrete quantities with units | Concrete registry identifier |

Add a strategy selector and configuration editor to the optimization UI, writing to `prop_opt`.
Display prescription units/basis and per-fraction/total-course convention beside reference inputs.
Keep newly created objectives literal physical dose unless the user chooses automatic intent.

Use the common preparation service for previews, without replacing source objectives:

```text
Automatic dose -> effect
60 Gy reference dose / 30 fractions -> 0.4 effect per fraction
Reference tissue: explicit alpha_x=0.1, beta_x=0.05
```

- Distinguish awaiting `Dij` from validated computation.
- Refresh on `dij`, plan, and CST changes.
- Preserve quantity selection when changing objective type.
- Refresh labels when quantity changes; do not secretly transform existing numbers on selection.
- Show strategy changes as changes to prescription interpretation.
- Filter image-reference choices by known quantity and fraction basis. Unknown provenance must
  not be silently treated as a compatible dose image.
- Keep automatic quantities out of result viewers.
- Update AI objective schemas/prompts and context to follow these semantics.
- Display preparation errors before starting the solver.

## 11. Serialization and matRad compatibility

### Native formats

Preserve original objective quantity identifiers, canonical strategy configuration, fraction
count, and dose convention. Resolved objectives and preprocessing caches are runtime state.
An optional report stored with results supplements original intent rather than replacing it.

### matRad import

Translate recognized plan-wide biological optimization settings at the patient-import boundary,
where plan and CST are both available. The standalone objective parser cannot infer that context.

The vendored [`matRad_fluenceOptimization.m`](../../matRad/matRad_fluenceOptimization.m) divides
dose parameters by fraction count and selects a plan-wide projection from `bioOptimization`.
Use verified source semantics to map dose objectives to explicit `dose_auto` plus a strategy,
including the appropriate reference policy. Do not infer equivalent behavior solely from names.

Preserve total-course prescription meaning and normalize exactly once. Unknown settings or
missing necessary context require a diagnostic or explicit migration, not a guessed strategy.

### Export modes

Distinguish these outcomes:

1. **matRad-executable subset:** allow only combinations proven representable with matRad's
   plan-wide projection and objective semantics. Convert per-fraction dose prescriptions to
   total-course values on export copies where required. Reject unrepresentable mixed quantities,
   literal effect prescriptions, reference policies, and custom strategies.
2. **pyRadPlan round-trip `.mat` archive:** preserve original objectives and unsupported features
   in versioned pyRadPlan metadata. Do not advertise this archive as an equivalent executable
   matRad optimization setup.

Preserve class-name mappings, parameter order, penalties, VOI tissue parameters, geometry,
index/axis conventions, and biological-model parameters. Keep unsupported objectives in native
metadata instead of silently losing them.

Compatibility checks belong at patient/exporter level. `Objective.to_matrad()` alone lacks plan
and fraction context. Do not export resolved per-fraction objectives as ordinary original dose
objectives: doing so risks duplicate fraction scaling or biological conversion.

## 12. Focused test matrix

| Area | Required cases |
|---|---|
| Explicit intent | Every concrete quantity survives strategy, modality, and matrix changes. |
| Defaults | Omitted quantity, serialized physical dose, omitted strategy, `None` rejection. |
| Automatic resolution | Physical identity, Gy(RBE) identity, LQ conversion, mixed automatic/literal roots. |
| Fraction ordering | `T(D/N)`, literal `E/N`, LET unchanged, square-root scaling, `N=1`, no double scaling. |
| Reference types | Scalar/image parity, resampling before nonlinear conversion, untouched non-reference fields. |
| Copying | Source objectives, arrays, images, metadata, and caches unchanged after success and failure. |
| Objective support | Supported pointwise conversion; mean/EUD/uniformity/custom rejection; unchanged priority. |
| Tissue | Homogeneous/heterogeneous support, explicit reference, absolute coefficients required, invalid values. |
| Scenarios | Supported mappings, missing scenario matrices, unsupported varying reference fields. |
| Computability | Missing dependencies, known model conflicts, missing provenance, no fallback, deduplication. |
| Repeat preparation | Same output on repeated initialization, no cumulative normalization or stale caches. |
| Solver integration | Correct mixed-root objective count/mapping and gradients with both Jacobian modes. |
| Migration | Legacy flag warnings/errors, explicit migration copies/report, existing ion-plan behavior. |
| Serialization | Native intent round trip, image references in supported formats, canonical strategy config. |
| matRad | Verified import mapping, one fraction conversion, executable subset rejection, metadata round trip. |
| GUI | Selector persistence, labels, previews, type changes, image filtering, validation before solve. |

Use small analytic fixtures for prescription transformations and existing biological fixtures for
quantity integration. Exercise available array backends for array-valued reference operations.
Do not assert that switching from dose to effect preserves the optimizer's solution or objective
weights.

## 13. Phased implementation checklist

Start regular implementation work from `develop`, following the repository branch conventions.
Keep the final release gate intact if phases are developed in separate PRs; do not expose a GUI
option whose preparation/serialization semantics are unfinished.

### Phase 1: literal semantics and preparation boundaries

- [ ] Remove concrete-quantity overwriting and introduce legacy-flag diagnostics.
- [ ] Accept objective-only `dose_auto`; reject unresolved automatic requests clearly.
- [ ] Preserve the omitted-quantity physical-dose default.
- [ ] Add independent objective/reference copies and clear runtime caches.
- [ ] Read plan properties without mutating the plan.
- [ ] Establish the common preparation result/report boundary.
- [ ] Add regression tests for explicit quantities, defaults, repeat runs, and failure immutability.
- [ ] Verify/fix mixed-quantity objective indexing and mapping.

Primary files:

- `pyRadPlan/optimization/problems/_optiprob.py`
- `pyRadPlan/optimization/problems/_nonlin_fluence.py`
- `pyRadPlan/optimization/objectives/_objective.py`
- `pyRadPlan/optimization/objectives/_factory.py`
- New `pyRadPlan/optimization/_preparation.py` (proposed location)
- `test/optimization/test_objectives.py`
- `test/optimization/test_fluence_optiprob.py`
- `test/optimization/test_dose_convention.py`

Acceptance: literal identifiers never change, automatic requests without a supported strategy
fail clearly, and source state remains unchanged.

### Phase 2: strategies and reference preparation

- [ ] Add the strategy protocol, registry, configuration validation, and preparation errors.
- [ ] Implement `physical_dose`, `rbe_x_dose`, and `lq_effect` prescription contracts.
- [ ] Add objective conversion hooks and explicit nonlinear support restrictions.
- [ ] Split image alignment from cache finalization.
- [ ] Declare quantity-specific fraction normalization and handle image references.
- [ ] Implement homogeneous/explicit LQ reference policies and provenance validation.
- [ ] Enforce existing scenario limits and reject unsupported conversion.
- [ ] Resolve concrete dependencies through one shared `QuantityResolver`.
- [ ] Add analytic conversion tests and fixture-backed mixed-quantity integration tests.
- [ ] Clarify reporting-only documentation on `default_report_quantity`.

Primary files:

- New `pyRadPlan/optimization/strategies/__init__.py`
- New `pyRadPlan/optimization/strategies/_base.py`
- New `pyRadPlan/optimization/strategies/_factory.py`
- New `pyRadPlan/optimization/strategies/_physical.py`
- New `pyRadPlan/optimization/strategies/_rbe.py`
- New `pyRadPlan/optimization/strategies/_lq_effect.py`
- `pyRadPlan/optimization/_preparation.py`
- `pyRadPlan/optimization/objectives/_objective.py` and affected objective implementations
- `pyRadPlan/quantities/_base.py` and concrete quantity classes
- `pyRadPlan/cst/_cst.py` where tissue-field validation helpers are needed
- `pyRadPlan/bio_models/_base.py` (documentation only unless integration requires more)
- New strategy/preparation tests under `test/optimization/`
- Existing `test/quantities/` and `test/bio_models/` integration fixtures

Acceptance: documented scalar/image conversions and fraction ordering hold, unavailable or
ambiguous requests fail before solve, and the quantity graph remains owned by `QuantityResolver`.

### Phase 3: GUI, persistence, and migration

- [ ] Add automatic/literal quantity labels and strategy configuration controls.
- [ ] Show input basis, fraction convention, resolved previews, and errors.
- [ ] Preserve quantity on objective-type edits and refresh on `Dij` changes.
- [ ] Validate reference-image quantity/fraction provenance.
- [ ] Update AI objective schemas/prompts/context.
- [ ] Canonicalize strategy serialization and retain original objective intent.
- [ ] Add an explicit migration helper with copied output and report.
- [ ] Implement verified matRad patient-level import mappings.
- [ ] Implement strict executable-export checks and versioned archive metadata.
- [ ] Add native/matRad round-trip and GUI tests.

Primary files:

- `pyRadPlan/gui/widgets/optimization/_optimization_widget.py`
- Relevant `pyRadPlan/gui/widgets/plan/` and workflow bindings
- `pyRadPlan/ai/agents/` objective generation/context code
- `pyRadPlan/plan/_plans.py`
- `pyRadPlan/optimization/objectives/_objective.py` and `_factory.py`
- `pyRadPlan/cst/_cst.py` serialization path
- `pyRadPlan/io/matlab/_importer.py` and `_exporter.py`
- Other native serialization paths as required by the supported formats
- `test/gui/widgets/optimization/test_widget_optimization.py`
- `test/gui/widgets/plan/test_widget_plan.py`
- `test/io/test_io_matlab.py` and `test/test_plan.py`

Acceptance: users can distinguish and preview intent, native round trips retain it, and export
never silently claims unsupported matRad equivalence.

### Phase 4: documentation and release validation

- [ ] Update `CHANGELOG.md` under `[Unreleased]` with implemented behavior and migration impact.
- [ ] Update `docs/api/optimization_objectives.rst` and expose strategy API documentation.
- [ ] Add user-guide examples for automatic, literal, mixed-quantity, and image-reference cases.
- [ ] Document input prescription bases, quantity-aware fraction semantics, and weight retuning.
- [ ] Document scenario limits and unsupported objective transformations.
- [ ] Run relevant optimization, quantity, biological-model, plan, and I/O tests.
- [ ] Run `pytest test/gui` for data-model and algorithm API changes.
- [ ] Run the full required suite and formatting/lint checks; do not reduce coverage.
- [ ] Build documentation if user-guide/API changes require it.
- [ ] If examples change, refresh committed executed notebooks with `python docs/execute_examples.py`.
- [ ] Reconcile this design note with the final public API and mark implementation status.

## 14. Alternatives considered

| Alternative | Assessment |
|---|---|
| `quantity=None` | Compact but conflates missing data with intentional automation and is less visible in saved objectives. |
| Separate `quantity_mode` field | Explicit but creates contradictory combinations and additional UI/serialization state. |
| Plan-wide conversion policy alone | Cannot distinguish literal concrete objectives from automatic objectives. |
| Top-level strategy on `Plan` | Adds another planning field when `prop_opt` already owns this configuration. |
| Per-objective strategies immediately | Flexible, but adds precedence and configuration complexity before a demonstrated need. |
| Biological model selects optimization default | Couples model/reporting choices to objective prescription semantics. |
| Transform all reference fields generically | Mechanically simple but invalid for unsupported nonlinear objective meanings. |
| Strategy returns and mutates an objective directly | Flexible but distributes copying, normalization, and cache ordering across implementations. |

Recommendation: retain the existing objective quantity field with visible `dose_auto` intent;
put strategy selection in `prop_opt`; use strategy recipes, objective-aware conversion hooks,
and a single preparation coordinator; keep computed quantities in `QuantityResolver`.

## 15. Review decisions before implementation

The following are proposed choices, not already accepted public API commitments:

- [ ] Confirm omitted objective quantity remains physical dose and automatic selection has no default.
- [ ] Confirm the three strategy identifiers and their prescription bases.
- [ ] Confirm that `lq_effect` requires an explicit reference-source configuration.
- [ ] Confirm initial nonlinear objective support and deferred voxel-wise/scenario features.
- [ ] Confirm literal quantity fraction semantics match existing result scaling.
- [ ] Confirm rejection of legacy `convert_dose_objectives=True` during migration.
- [ ] Confirm the separation of executable matRad export from metadata-preserving archives.
- [ ] Choose concrete helper names, tissue comparison tolerances, metadata schema version, and
      the release defining the legacy flag's removal.

These review items belong to the later API implementation discussion. Creating this plan does
not authorize or claim implementation of the proposed runtime changes.
