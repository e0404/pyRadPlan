---
orphan: true
---

# Robust optimization on scenario sets: design and implementation plan

Status: contract committed (operator base, registry, `none` / `expected_value` operators,
`Objective.robustness`, `ScenarioSet.nominal_indices`); operators, problem integration, GUI and
docs implemented in parallel on top of it. Section 6 records the decisions and the Dij-level
caching investigation.

Reference: matRad's `matRad_objectiveFunction.m` / `matRad_objectiveGradient.m` (robustness
switch per objective: `none`, `STOCH`, `PROB`, `VWWC`, `VWWC_INV`, `COWC`, `OWC`) and its smooth
maximum approximations `logSumExp` / `pNorm` (`useMaxApprox`, `p = 30`).

## 1. Goals

- Robust optimization approaches on the scenario sets of `docs/development/scenario_models.md`:
  worst case (composite, objective-wise, voxel-wise) and probabilistic (weighted average over
  scenarios), with matRad's smooth maximum variants (`logsumexp`, `pnorm`, exact `none`).
- Robustness is an option *on the objective* (matRad's `robustness` property): a string for now,
  resolvable to an operator through a registry so that a dictionary / model with more
  configuration can replace the string later without touching consumers.
- Robustness enters the optimization through a `RobustnessOperator` that acts either on one
  objective (objective-wise worst case, expected value) or on the collection of all objectives
  with that operator (composite worst case). The operator owns the chain between the per-scenario
  quantity values and the robust objective value, including the per-scenario cotangents, so
  the planning problem's gradient bookkeeping stays generic.
- Scenario-aware objectives (a variance minimizer) fit the same pipeline.
- Per-scenario results that the GUI can separate (component selector, DVH per scenario) and a
  robustness column in the objective editor.
- Compatibility with the roadmap: multi-objective formulation with scalarization / a-priori /
  a-posteriori strategies (`refactor/optimization`: `Function`, `strategies`, `evaluate_functions(x,
  idxs)` / `evaluate_function_jacobians(x, idxs)`), and constraints (per scenario, over all
  scenarios, or on the robust operator value itself).

Non-goals: matRad's `PROB` variance term with the Ω matrices (replaced by the matrix-free
variance objective, see 3.6), 4D voxel tracking between CT phases, robust constraints
(the term model supports them, nothing evaluates them yet), the GUI editing of operator
parameters beyond the robustness name.

## 2. Current state

- Quantities evaluate every scenario of the set (`RTQuantity.default_scenarios`) and the fluence
  problem *sums* each objective over all of them (`_objective_functions`), i.e. a non-robust
  objective on a 9-scenario `wcScen` dij is already an (unweighted) expected value. With the
  operators, `robustness="none"` evaluates the nominal scenario(s) only, as in matRad.
- The gradient path already supports one cotangent per scenario slot
  (`compute_chain_derivative` with an object array; `_grad_cache_intermediate[q]` of shape
  `(n_scenarios, rows, n_vox)`), so operators only need to produce per-scenario cotangents.
- `refactor/optimization` has not merged into develop. Its `Function` keeps `priority`,
  `quantity`, bounds and `is_objective`; strategies partition the function list and own priority
  weighting. Our term model (3.4) is designed so that a *term* becomes one entry of that function
  list.

## 3. Design

### 3.1 Vocabulary

For objective `j` on voxel set `V_j` and scenario `s` with quantity vector `q_s`:
`f_js = phi_j(q_s[V_j])`, `g_js = d phi_j / d q_s[V_j]`. The robust problem value is
`F = sum_terms T(...)` where each term is produced by one operator over its entries.

### 3.2 `RobustnessOperator` (contract, `pyRadPlan/optimization/robustness/_base.py`)

```python
class RobustnessOperator(ABC):
    name: ClassVar[str]                      # canonical, e.g. "objective_worst_case"
    aliases: ClassVar[tuple[str, ...]]       # matRad names and short forms, e.g. ("OWC",)
    matrad_name: ClassVar[Optional[str]]     # written by Objective.to_matrad, None if none
    scope: ClassVar[Literal["objective", "composite"]] = "objective"

    def __init__(self, scenarios: ScenarioSet, options: RobustnessOptions | None = None)
    @property
    def scenario_indices(self) -> list[int]   # flat set indices the operator needs (default:
                                              # all scenarios with weight > 0)
    @property
    def weights(self) -> np.ndarray           # set weights of scenario_indices
    def evaluate(self, entries, values) -> Array        # scalar, priorities included
    def gradient(self, entries, values) -> list[Array]  # d F / d q_s[V_j], shape (S, n_j) per entry
```

`entries` is a sequence of `RobustTermEntry(objective, voi_type)`; `values[j]` is the stacked
quantity of entry `j` over the operator's scenarios, shape `(S, n_j)`, in `scenario_indices`
order. Objective-scope operators always receive one entry; composite-scope operators receive
every objective carrying that robustness name (one instance per problem and name).

Operators are array-API agnostic: `values` arrive in the optimization backend and the operator
uses `array_api_compat.array_namespace(values)`.

`RobustnessOptions` (frozen dataclass): `max_approximation` (`"logsumexp"` default, `"pnorm"`,
`"none"`), `p_norm` (30, matRad). The planning problem exposes them as configuration
(`NonLinearFluencePlanningProblem.max_approximation`, `.p_norm`) and hands them to every operator.

### 3.3 Operators

| name | aliases | scope | value | cotangent `dF/dq_s` |
|---|---|---|---|---|
| `none` | | objective | `sum_{s in nominal} p f_s` | `p g_s` on nominal scenarios |
| `expected_value` | `STOCH`, `stochastic`, `probabilistic` | objective | `p sum_s w_s f_s` | `p w_s g_s` |
| `expected_quantity` | | objective | `p phi(sum_s w_s q_s)` | `p w_s g(q_bar)` |
| `voxel_worst_case` | `VWWC` | objective | `p phi(d~)`, `d~ = min_s q_s` (TARGET) / `max_s q_s` (OAR) | `p g(d~) * 1[argmin/argmax == s]` |
| `voxel_worst_case_inverted` | `VWWC_INV` | objective | as above with max (TARGET) / min (OAR) | likewise |
| `objective_worst_case` | `OWC` | objective | `smoothmax_s (p f_s)` | `(d smoothmax / d f_s) p g_s` |
| `composite_worst_case` | `COWC` | composite | `smoothmax_s (sum_j p_j f_js)` | `(d smoothmax / d F_s) p_j g_js` |
| `scenario_stack` | | objective | `p phi(Q, w)` for scenario-aware objectives | `p dphi/dQ` |

`nominal` scenarios are `ScenarioSet.nominal_indices()`: every scenario at the nominal value of
all *continuous* dimensions (one per CT phase in a 4D set, matRad's `useNominalCtScen`), falling
back to the tagged nominal. `none` sums them unweighted as matRad does.

`expected_value` uses the set's `weights` (normalized aggregation weights). matRad's `STOCH`
multiplies by the raw densities `scenProb`; values therefore differ from matRad by the
normalization constant, which the documentation states.

`smoothmax` lives in `_smooth_max.py`: matRad's stable `logsumexp` (`f_max + log1p(sum_{s !=
max} exp(f_s - f_max))`, gradient `exp(f - f_max) / (1 + sum)`), its normalized `pnorm`
(`f_max (1/n)^(1/p) ||f/f_max||_p`, gradient `(1/n)^(1/p) (f/f_max / ||.||_p)^(p-1)`), and the
exact maximum with a one-hot gradient. All three are checked against finite differences.

Voxel-wise operators need the VOI type, which is why entries carry `voi_type`; an unknown type
(`EXTERNAL`, `HELPER`) is treated like an OAR with a warning. They raise for sets whose CT phases
differ between scenarios (4D voxel tracking is not implemented, as in matRad).

### 3.4 Terms (problem side, `pyRadPlan/optimization/robustness/_terms.py`)

```python
@dataclass
class RobustTerm:
    operator: RobustnessOperator
    items: list[TermItem]          # TermItem(mask, objective, quantity, voi_type)
    scenario_indices: list[int]    # operator.scenario_indices
```

`build_robust_terms(objectives_per_voi, scenarios, options)` creates one term per objective-scope
objective and one per composite-scope operator name. The planning problem keeps `self._terms`
and reports one value / one jacobian row per *term* (`f_vals`, `_grad_cache`), not per
objective. A composite term's priorities are applied inside its operator, so the outer
scalarization sees priority 1 for it. This is the shape `refactor/optimization` expects: a term
is one function of the strategy's function list, a constraint term is a term whose operator
value is bounded, and "a constraint on all scenarios" is a term over `expected_value` /
worst-case operators of the constraint function.

Scenario selection per quantity: the resolver is created with the union of
`term.scenario_indices` over the terms on that quantity, so a `none` objective on a
`wcScen` dij computes only the nominal slot, while robust objectives compute all slots they
need. Quantities shared by terms with different needs take the union.

Evaluation (`_objective_functions`): for each term, stack `q_vectors[item.quantity].flat[s][ix]`
over `term.scenario_indices`, call `operator.evaluate`. Gradient (`_objective_jacobian`): call
`operator.gradient`, scatter row `s` of each returned block into
`_grad_cache_intermediate[quantity][pos(s), term_row, ix]` where `pos(s)` is the position of
`s` in the quantity's `scenarios`. The back-projection through each scenario's influence matrix
is unchanged.

Legacy dijs without a scenario set: a single-slot dij gets an implicit nominal set; a multi-slot
dij without a set raises when any objective is robust, asking for `Dij.attach_scenarios`.

### 3.5 `Objective.robustness`

`robustness: str = "none"`, validated through the registry (aliases normalized to the canonical
name, so matRad's `OWC` validates and serializes as `objective_worst_case`). `to_matrad` writes
the operator's `matrad_name` as `robustness` and warns when there is none; the matRad import
reads the `robustness` key. `Objective.available_robustness()` lists the names an objective
accepts (all registered operators; scenario-aware objectives only `scenario_stack`). The field
stays a string in this pass; `get_robustness_operator()` already accepts a `{"name": ...,
**params}` dictionary, so a model-valued field is a type change on the objective only.

### 3.6 Scenario-aware objectives

`Objective.scenario_aware: ClassVar[bool] = False`. A scenario-aware objective's
`compute_objective(values, weights)` / `compute_gradient(values, weights)` receive the stacked
`(S, n)` quantity and the scenario weights and return a scalar / an `(S, n)` array. The problem
routes such objectives through the `scenario_stack` operator regardless of their `robustness`
string (the field is forced to `scenario_stack`).

`Variance` (name `"Variance"`): `f = (1/n) sum_v Var_w[q_v] = (1/n) sum_s w_s ||q_s - q_bar||^2`
with `q_bar = sum_s w_s q_s`; gradient `2 w_s (q_s - q_bar) / n` (the chain through `q_bar`
vanishes). Matrix-free; no Ω matrices on the dij.

### 3.7 Results and GUI

- `Dij.compute_result_*` gain `scenarios: bool | Sequence[int] = False`. With several
  scenarios they add `"<quantity>_scenario"` lists (one image per computed or alias scenario of
  the set, index = flat set index) next to the existing `"<quantity>_beam"` lists.
  `fluence_optimization` is unchanged; the GUI workflow asks for scenarios when the dij's set has
  more than one.
- `_display_names.group_result_keys` learns the `_scenario` component kind: the
  `QuantitySelector` component combo lists `full`, `beam i`, `scenario i`, labelling scenarios
  from the workspace dij's set (`setup (+3, 0, 0) mm, range +1 mm` style, nominal marked).
- DVH widget: with scenario components present, an "all scenarios" option overlays one thin
  curve per scenario behind the nominal one (band-like view).
- Objective editor: a `Robustness` column (combo of `Objective.available_robustness()`), stored
  on the objective; the plan widget's scenario model row is the only other GUI entry point.

### 3.8 Future compatibility

- Strategies (`refactor/optimization`): `_terms` replaces `_function_list` entries one-to-one;
  `evaluate_functions(x, idxs)` indexes terms. Priorities inside composite terms are a
  documented exception to "strategies own priorities" (the composite maximum is not linear in
  them).
- Constraints: `RobustTerm` with an `is_objective=False` function; a constraint "on the operator
  value" is the term value with bounds; "on a single scenario" is a term with a `none`-like
  operator pinned to that scenario; "on all scenarios" expands to one term per scenario.
- Dependent scenarios (`ScenarioBasis`) are transparent: operators see the quantity values the
  quantities already combined.

## 4. Dij-level evaluation and caching (investigation)

Question: move evaluation and caches (matrix-vector products) into the `Dij` to simplify the
quantities and speed up scenario evaluation?

Findings (see 6.2 for the measurement):

- Forward products are not duplicated today: the `QuantityResolver` instantiates each quantity
  once and `FluenceDependentQuantity.compute` caches by fluence, so `physical_dose` used by
  `Dose` and by the indirect `RBExDose` path is one `D_s w` per scenario.
- Backward products *are* duplicated whenever several consumers send cotangents into the same
  matrix (physical-dose and RBE-weighted-dose objectives, or `effect` through the alpha and
  sqrt-beta paths): one `D_s^T g` per consumer instead of one per matrix. This multiplies with
  the number of scenarios and is independent of robustness.
- A Dij-level cache cannot remove that duplication: the chain derivative is linear in the
  cotangent, so cached products are never reusable. Only accumulation (sum cotangents per
  matrix, then one transpose product) removes it, which is the reverse-mode backward pass over
  the quantity graph recorded as a follow-up (`project_quantity_backward_pass` memory).
- Derived (expected-value) scenarios are vector combinations of cached per-scenario values; no
  extra products.

Decision: keep evaluation in the quantities for this pass. The robust operators only change
*which* scenarios a quantity evaluates (union over terms) and what the cotangents are. The
Dij-level step worth doing later is the accumulating backward pass, with a `Dij.apply(field,
w, scenarios)` / `Dij.apply_transpose(field, cotangents)` pair as the place where one product
per matrix is enforced; it is a separate branch because it changes every quantity's derivative
contract.

## 5. Work packages

| Package | Owner | Files |
|---|---|---|
| Contract | orchestrator | `robustness/_base.py`, `_factory.py`, `_none.py`, `_expected_value.py`, `Objective.robustness`, `ScenarioSet.nominal_indices` |
| A1 operators | agent | `_smooth_max.py`, `_expected_quantity.py`, `_voxel_worst_case.py`, `_objective_worst_case.py`, `_composite_worst_case.py`, `objectives/_variance.py`, matRad import/export, unit tests (finite differences) |
| A2 problem | agent | `robustness/_terms.py`, `_optiprob.py`, `_nonlin_fluence.py`, problem options, tests on a multi-scenario dij (finite differences, `none` = nominal only, `expected_value` = weighted single-scenario values, composite row count) |
| B results & GUI | agent | `Dij.compute_result_*`, workflow widget, `_display_names`, `QuantitySelector`, DVH overlay, objective editor column, GUI tests |
| C investigation | agent | measurement for section 4 |
| Docs | agent | `docs/user_guide/concepts/optimization.rst` robust section, `uncertainty.rst`, API pages |
| Changelog, memory | orchestrator | |

## 6. Decisions and record

### 6.1 Decisions

- Robustness is a string on the objective validated against the operator registry; canonical
  snake_case names with matRad aliases. matRad `PROB` imports as `expected_quantity` with a
  warning that its variance term is dropped (use the `Variance` objective).
- One value / jacobian row per term, priorities inside composite operators.
- `none` evaluates the nominal scenario(s) only. Behavior change for multi-scenario dijs, listed
  in the changelog.
- `expected_value` uses normalized set weights, not matRad's raw densities.
- Evaluation stays in the quantities; Dij-level backward accumulation is a follow-up.

### 6.2 Investigation record

Filled in by the orchestrator from package C's report.
