---
orphan: true
---

# Biological models: formalisms, base-data provenance and tissue compatibility

Status: stages 1–5 (Section 8) and the independent z\* fix are implemented; Section 11 lists
the resulting API and where the implementation deviates from this design. Availability also
depends on the dose engine (Sections 5.5, 6 and 11).

## 1. Problem

The biological models currently mix two independent things: the way pyRadPlan evaluates
the biology at runtime, and the radiobiological model that produced the base data it reads.

- `kernel_based_lq` reads precomputed α/β kernels. Its name says nothing about that, and it
  is bound to `LEMTissueProperties` (`tissue_properties_type`), although the kernels could
  just as well have been computed with mMKM for a cell line.
- `dose_average_alpha_beta` averages α/√β tables over fragment fluence spectra
  (Zaider–Rossi). Its name suggests the dose averaging that produced the α/β kernels, and the
  tables could come from any model compatible with that formalism, not only LEM.
- `MKM` evaluates α = α₀ + β₀·z\* from z\* kernels. Any model built on a z\* kernel could be
  evaluated there, so "MKM" names one data source rather than the formalism.
- The list of models depends on the radiation mode only, not on what the machine contains.
- An incompatible tissue is accepted by the editors, passes dose calculation with a warning
  (falling back to physical dose) and fails late, e.g. in optimization on `rbe_x_dose`.

## 2. Concepts

**Formalism.** What pyRadPlan computes at runtime. A formalism is code: one
`BiologicalModel` class. The set is small and grows only when a new evaluation scheme is
implemented.

**Provenance (data model).** The radiobiological model and its parameterization that
produced a set of kernels or tables, e.g. LEM I, LEM IV, MKM, mMKM. Provenance is data. It
is declared in the base-data metadata and never hard-coded in a formalism.

**Kernel set.** One family of per-energy kernel data in a machine, e.g. "the α/β kernels",
"the z\* kernels", or a second, differently defined LET. A kernel set has a kind, an optional
variant, a provenance and one column per tissue class where it is tissue dependent.

**Tissue class.** One parameterization of one provenance in the base data (one column of a
kernel set, or one RBE table). A structure's `VOI.tissue_properties` must resolve to exactly
one tissue class of the kernel set the formalism uses.

The display convention is **formalism · provenance**, e.g. "α/β kernels · LEM I".

## 3. Formalisms

| Identifier (new) | Aliases kept | Display name | Evaluates | Needs |
|---|---|---|---|---|
| `none` | | None (physical dose) | physical dose | – |
| `constant_rbe` | `constRBE` | Constant RBE | RBE × dose | – |
| `WED`, `MCN`, `CAR`, `HEL`, `LSM` | | Wedenberg (LET), McNamara (LET), Carabe (LET), Mairani (LET), Linear scaling (LET) | phenomenological RBE from LET and α_x/β_x | a LET kernel set |
| `alpha_beta_kernels` | `kernel_based_lq`, `LEM` | α/β kernels | LQ with α, √β read from per-tissue kernels | an α/β kernel set |
| `alpha_beta_tables` | `dose_average_alpha_beta` | α/β tables + fluence spectra | LQ with per-ion α/√β tables averaged over the fragment fluence spectra (Zaider–Rossi) | RBE tables and fluence spectra |
| `zstar_kernels` | `MKM`, `zs_based_lq` | z\* kernels | α = α₀ + β₀·z\*, β = β₀ | a z\* kernel set |

- Identifiers are what plans store. The aliases keep existing plans, workspaces and matRad
  files loading; matRad export keeps the current names (`LEM`, `constRBE`, …).
- The LET models stay separate classes: their formula is the model, and new LET- or
  z\*-based formulas are added as new classes in the same way.
- Each class gets `name` (display name) and a one-line `description`, used for tooltips
  instead of the docstring's first line.
- `tissue_properties_type` is removed from the data-driven formalisms. The tissue type
  follows the provenance of the tissue class (Section 5.2). Formalisms instead declare the
  tissue parameters they read at runtime (`required_tissue_parameters`; `zstar_kernels`:
  `alpha_0`, `beta_0`).

## 4. Base-data metadata specification (version 1)

The spec describes matRad-style `.mat` machine files and RBE table files. Field names are
camelCase as in matRad; the loader maps them to snake_case. Everything is optional; Section
4.4 defines what is inferred when it is missing.

### 4.1 Kernel sets: `machine.meta.kernelSets`

One entry per kernel set:

| Field | Meaning |
|---|---|
| `id` | Unique name within the machine, referenced by tissue classes (`ref_Kernel`). |
| `kind` | `alpha_beta`, `zs`, `let`, or a future kind. |
| `fields` | The kernel fields holding the data, e.g. `{alpha, beta}`, `zs`, `LET`. |
| `variant` | Kind-specific definition, e.g. for `let`: `LETd`, `LETt`, `LETd_restricted_<keV>`; for `zs`: `z1D`, `z1D_saturation_corrected`. |
| `model` | Provenance, e.g. `LEM_I`, `mMKM`; empty for kinds that are not tissue dependent (LET). |
| `reference` | Free text: publication, code and version used to produce the data. |
| `default` | `true` marks the set used when several sets of the same kind exist and nothing is configured. |

Several sets of the same kind are allowed, e.g. α/β kernels from LEM I and from mMKM, or a
dose-averaged and a track-averaged LET. Their data lives in separately named kernel fields;
the loader exposes them as `machine.kernel_sets` (Section 5.1).

**`.mat` layout.** `machine.meta.kernelSets` is a 1×n struct array, one element per set (a
cell array of structs is read as well). All values are char unless noted:

```matlab
machine.meta.kernelSets(1) = struct('id','lem1', 'kind','alpha_beta', ...
    'fields',{{'alpha','beta'}}, 'variant','', 'model','LEM', 'version','LEM_I', ...
    'reference','Scholz 1997, ...', 'default',true);
machine.meta.kernelSets(2) = struct('id','mmkm', 'kind','alpha_beta', ...
    'fields',{{'alpha_mMKM','beta_mMKM'}}, 'variant','', 'model','mMKM', 'version','', ...
    'reference','Inaniwa 2010, ...', 'default',false);
```

- `fields` is a cellstr or one comma-separated char (`'alpha_mMKM,beta_mMKM'`); it defaults
  to the kind's standard fields (`alpha`/`beta`, `zs`, `LET`). `id` defaults to the kind
  (`kind_<n>` if the kind repeats). `default` is a logical or 0/1; `version` is optional.
- **Several sets of one kind** store their per-energy arrays in fields of their own in
  `machine.data`, next to matRad's standard ones; the convention is `<field>_<suffix>`
  (`alpha_mMKM`, `beta_mMKM`, `LETt`). Tissue-dependent arrays have the shape of `alpha`
  (n_depths × n_columns). Only the set reading the standard `alpha`/`beta` identifies its
  columns by the per-energy `alphaX`/`betaX`; the classes of any other set give `column`
  or are mapped in declaration order.

### 4.2 Tissue classes: `machine.meta.tissueParameters`

Already present in the oxygen machine. One entry per tissue class:

| Field | Meaning |
|---|---|
| `cell_Line` | Display name of the class, e.g. `Generic_LEMax01` (the field matRad already uses). |
| `model` | Provenance, e.g. `LEM_I`, `mMKM` (a LEM version may also be given as `model: LEM`, `version: LEM_I`). |
| model parameters | e.g. `alphaX`, `betaX`, `Dt`, `rNucleus`, `rDomain`, `alpha0`, `beta0`. |
| `ref_Kernel` | `id` of the kernel set this class belongs to (today `alpha_beta` / `zs`). |
| `column` | One-based column of that kernel set (NaN or missing: matched by `alphaX`/`betaX` as today, or declaration order for sets without them). |

Without `ref_Kernel`, a class belongs to the set of its kind (MKM parameters: z\*, all others:
α/β), chosen by provenance or the `default` flag if the kind has several sets.

### 4.3 RBE tables: `RBEtable.meta`

Already carries `model` and `modelParameters`. The spec adds the full tissue parameters of
the provenance (e.g. the LEM version, `Dt`, `rNucleus`) and `reference`. The metadata
describes the tabulated values only: how they are combined (averaging over fluence spectra
for `alpha_beta_tables`) is the formalism of the model using the table, not a property of the
table, so tables do not declare a formalism (an existing `formalism` entry is ignored).

### 4.4 Legacy data

- No `kernelSets`: one set per kind present in the kernels (`alpha_beta` if `alpha`/`beta`
  exist, `zs`, `let`), with ids `alpha_beta`, `zs`, `let` and `model` "not declared" — unless
  all tissue classes of a set declare the same provenance, which the set then takes over.
  The bundled machines declare their sets (Section 7).
- No `tissueParameters`: one tissue class per α/β kernel column, from `alphaX`/`betaX`,
  matched on α_x/β_x only (today's carbon behaviour).
- Labels for undeclared provenance read e.g. "α/β kernels · model not declared".
- Tissue-class parameters the base data does not document are stored as NaN ("not
  declared") and are not compared when a structure's tissue is matched to the class (the
  carbon LEM IV classes declare α_x/β_x only, not `Dt` / `rNucleus`). Equality of tissue
  properties stays strict.

## 5. Library architecture

### 5.1 Machine

- `machine.kernel_sets: dict[str, KernelSetInfo]` (id, kind, variant, provenance, reference,
  default flag, tissue classes).
- `machine.tissue_properties` stays the list of all tissue classes; each class knows its
  kernel set and column.
- `machine.provided_quantities()` reports kinds and, where several exist, variants.

### 5.2 Provenance registry

Maps a provenance name to the `TissueProperties` subclass holding its parameters:
`LEM_I`…`LEM_IV` → `LEMTissueProperties(version=…)`, `MKM`/`mMKM` → `MKMTissueProperties`,
unknown or undeclared → `TissueProperties` (α_x/β_x only). New provenances register here;
`TissueProperties.from_dict` dispatches through it.

The provenance of a *structure's* tissue (its `model` label and the LEM `version`,
`TissueProperties.provenance_fields`) is a soft label, not a match criterion. Matching a
structure's tissue to a tissue class compares the identifying parameters only (LEM: α_x,
β_x, `d_t`, `r_nucleus`; MKM/mMKM: `r_nucleus`, `r_domain`), parameters the class leaves
undeclared excluded. Tissues of different parameter classes match through the parameters
they share (a reference-only tissue and a LEM class by α_x/β_x); a LEM tissue and an MKM
class share none and never match. A tissue matching a class of another provenance is
evaluated with the base data, i.e. with the class's provenance, and gets a warning; users
are not forced to use matching labels. Of several matching classes the one of the tissue's
own provenance is preferred, then the declaration order. `TissueProperties.__eq__` stays
strict about the provenance.

### 5.3 Configuring a formalism

Data-driven formalisms take two optional parameters, serialized with the plan:

- `data_model`: provenance to use, e.g. `{"model": "alpha_beta_kernels", "data_model": "LEM_I"}`.
- `kernel_set`: explicit kernel-set id, for machines with several sets of one provenance.

Quantity-based formalisms (LET, z\*) take an optional `quantity_variants`, e.g.
`{"let": "LETt"}`.

Resolution is forgiving and deterministic:

1. An explicit `kernel_set` wins.
2. Otherwise the sets of the right kind matching `data_model`.
3. Otherwise the sets that the structures' tissues resolve to (by parameter matching); of
   several, the sets they resolve to with their own provenance label.
4. Otherwise the set marked `default`, or the only set of that kind.
5. Still ambiguous: an error listing the candidate sets and how to choose one.

Unset parameters therefore just work on machines with one set per kind, which covers all
bundled data.

### 5.4 Per-structure resolution

Tissue resolution is per structure: each structure's tissue resolves to one
(kernel set, column) pair, and the evaluator gathers per-voxel values from those pairs. The
first implementation may restrict a calculation to one kernel set per formalism, but the
resolution result is shaped per structure so that mixing provenances (e.g. LEM I for the
target, mMKM for an organ at risk) and later biological scenarios (alternative
parameterizations evaluated side by side) need no redesign. The GUI does not expose
mixing.

### 5.5 Options and compatibility check

- `bio_model_options(radiation_mode, machine=None, engine=None) -> list[BioModelOption]`:
  one entry per (formalism, kernel set or table) available on the machine, with identifier,
  parameters, label ("α/β kernels · LEM I"), description, tissue classes, availability and
  the reason when unavailable. Without a machine: the plain formalism list.
- **The dose engine** is the third dimension. A pencil-beam engine evaluates the machine's
  kernel data; a Monte Carlo engine does not use it and supplies only what it scores
  itself. Each engine class declares this without loading a machine
  (`DoseEngineBase.bio_capability()`, Section 11). With `engine=`, a formalism needing a
  quantity the engine neither reads from the machine's kernels nor scores is unavailable
  and the reason names the engine ("The dose engine FRED does not use the machine's α/β
  kernels."); a model with outputs needs an engine computing biological influence
  matrices; a quantity the engine scores replaces the machine's kernel sets of that kind
  (FRED: LET as LETd, so a LET model is offered on a machine without LET kernels, and one
  configured for `{"let": "LETt"}` is not).
- `BiologicalModel.check_tissues(machine, cst, engine=None) -> list[TissueIssue]`: one issue
  per affected structure; what the engine cannot compute is a `model_unavailable` issue,
  checked even without a machine. Each issue has a `severity`, `"error"` or `"warning"`.
  Errors: model unavailable for the machine, a tissue parameter class sharing no
  identifying parameter with the kernel set's (`type_mismatch`), no matching tissue class
  (with the closest class as suggestion), missing runtime parameter (e.g. α₀/β₀ for z\*
  kernels), invalid reference α_x/β_x. Warning: `provenance_differs`, a structure matching
  a class by its parameters but labelled with another provenance (e.g. MKM for mMKM z\*
  kernels, LEM I for LEM IV). **Every structure must resolve**, i.e. have no error.
- The dose engine runs the same check, with itself as `engine`, before building the
  evaluator, so the GUI check and the calculation cannot drift. With `calc_bio_dose=True`
  errors raise; with `"auto"` the engine keeps its physical-dose fallback for scripted use.
  Warnings never change the calculation: the engine logs them, reports them as engine
  warnings (`report_warning`, shown by the GUI) and evaluates the model.

## 6. GUI behaviour

- **Model dropdown**: built from `bio_model_options` for the selected machine and dose
  engine (the engine combo; without a selection the mode's default engine, as the dose
  engine factory chooses it), labelled formalism · provenance. Entries the machine or the
  engine cannot run are shown disabled with the reason. An unknown or unloadable machine
  shows the plain formalism list, still judged by the engine. The dropdown is rebuilt when
  the radiation mode, machine or engine changes.
- **Selection after a change**: a selection the new radiation mode, machine or engine can
  run is kept; otherwise the default of the mode is selected (the first available entry of
  `default_bio_models[mode]`, else `none`) and a status note names the replaced model and
  why. A mode's default followed into a new mode is replaced by the new default silently.
  The disabled "(not supported by this machine)" / "(not supported by the dose engine X)"
  entry is only used to keep the model of a plan restored from the workspace, so Apply
  keeps it.
- **Tissue editors** (plan widget table, VOI dialog): offer only tissue classes and cell-line
  presets of the selected entry's provenance (an entry the engine cannot run restricts
  nothing). On OK, `check_tissues` runs; rows with errors are
  highlighted and a dialog offers "Apply anyway" / "Keep editing" (default). Warnings never
  ask: a model label (or LEM version) differing from the matching class is tinted softly in
  a warning colour, distinct from the error tint, and an optional, non-modal "Relabel as
  <model>" action adopts the base data's provenance (`fix_provenance`).
- **Before dose calculation** (dose influence and dose recalculation): if `check_tissues`
  (with the plan's dose engine) reports errors, a dialog lists them and offers "Calculate physical dose only" or "Cancel".
  Accepting sets the plan's biological model to `none` in the workspace and then runs the
  calculation; cancelling runs nothing. Invalid α_x/β_x block the calculation. Warnings
  alone are logged and shown in the workflow status line, and the calculation runs with the
  biological model; the engine reports them again once it has run.
- **Plan widget indicator**: a marker next to "Set tissue parameters" while the structures
  have issues for the selected model and machine: ⚠ for errors, ℹ for warnings only;
  re-checked when the model, machine or structure set changes.

## 7. Bundled data

- `carbon_Generic.mat`: declares its α/β kernels as LEM IV ("α/β kernels · LEM IV"); the
  tissue classes give α_x/β_x, `Dt` and `rNucleus` are not documented (NaN, Section 4.4).
- `oxygen_Generic.mat`: declares "α/β kernels · LEM I" and its z\* kernels as mMKM with the
  variant `z1D_saturation_corrected` ("z\* kernels · mMKM", tissue class `Generic_MKM` of
  provenance mMKM). The two HSG · mMKM presets (Inaniwa 2010) have its geometry; an
  MKM-labelled tissue with the class's geometry resolves to it with a `provenance_differs`
  warning (Section 5.2).
- `RBEtable_LEMI_Scholz06_AX01_BX005.mat`: add `formalism`, the LEM version and parameters.
- No bundled machine carries fluence spectra, so `alpha_beta_tables` stays unavailable on
  bundled data; its tests use synthetic spectra.

## 8. Implementation stages

1. Metadata parsing (`kernel_sets`, extended `tissueParameters`, table meta) with legacy
   inference; provenance registry.
2. Formalism identifiers, aliases, names, descriptions; `data_model`, `kernel_set`,
   `quantity_variants`; resolution rules; per-structure resolution.
3. `bio_model_options` and `check_tissues`; the engine uses the check.
4. GUI: dropdown, filtered tissue editors, checks on accept and before dose calculation,
   indicator.
5. Bundled data metadata; documentation of the spec in the user guide.

Independent of these stages: `zstar_kernels` must reject tissues whose α₀/β₀ are unset
instead of producing NaN voxels.

## 9. Tests

- Metadata parsing per field and legacy inference; round trip through `.mat`.
- Resolution rules 1–5 with machines carrying one and several sets per kind.
- `check_tissues` for every issue kind, and parity: `calc_bio_dose=True` raises exactly when
  the check reports errors.
- Per-structure resolution with two provenances (library only).
- GUI (offscreen): dropdown per machine, editor filtering, accept dialog, physical-dose-only
  path reconfiguring the plan, cancel path running nothing.

## 10. Open questions

- Settled: the field names and nesting of `kernelSets` and the storage of several sets of
  one kind (Section 4.1, `.mat` layout). Open: whether matRad adopts the same spec.
- Whether `data_model` should also accept a provenance family (e.g. any LEM version). Not
  implemented; a bare `LEM` names LEM I, as for tissue parameters.

## 11. Implementation (stages 1–3)

Library API:

- `machine.kernel_sets -> dict[str, KernelSetInfo]` (`id`, `kind`, `fields`, `variant`,
  `provenance`, `reference`, `default`, `declared`, `tissue_classes: tuple[TissueClassInfo]`
  with `tissue`/`kernel_set`/`column`, `problems`, `complete`, `label`). Machine metadata is
  kept as `machine.kernel_set_declarations` and `machine.tissue_class_refs`; extra kernel
  fields as `kernel.additional_data`. `machine.provided_quantities()` adds
  `"<quantity>:<variant>"` (e.g. `"let:LETt"`) where several sets of a kind exist.
- `pyRadPlan.biology.tissues`: `get_provenance(model, version)`, `register_provenance`,
  `tissue_provenance(tissue)` / `TissueProperties.provenance`, `provenance_label`.
- `BiologicalModel`: `name`, `description`, `kernel_set_kind`, `required_tissue_parameters`,
  `select_kernel_set(machine, tissues)`, `resolve(machine, cst) -> BioModelResolution`,
  `check_tissues(machine, cst) -> list[TissueIssue]`; `raise_for_tissue_issues(model,
  issues)` is how dose engines act on the check.
- `bio_model_options(radiation_mode, machine=None) -> list[BioModelOption]` (`model`,
  `parameters`, `spec`, `label`, `formalism`, `description`, `provenance`, `kernel_set`,
  `tissue_classes`, `available`, `reason`, `create()`).
- `BioMachineSummary.from_machine(machine)`: cacheable, kernel-free stand-in accepted
  wherever these functions take a machine.
- Engine capability: `DoseEngineBase` class attributes `bio_kernel_quantities` (machine
  quantities read from the pencil-beam kernels, `None` for all), `bio_scored_quantities`
  (`"<quantity>"` or `"<quantity>:<variant>"`), `bio_scored_radiation_modes` (the modes
  the scored quantities exist for, `None` for all; `EngineBioCapability.scored_radiation_modes`,
  narrowed per plan by `for_radiation_mode(mode)`, which `bio_model_options`,
  `check_tissues` and `machine_view` apply) and `computes_bio_influence`;
  `DoseEngineBase.bio_capability() -> EngineBioCapability` and
  `DoseEngineBase.bio_quantities(machine=None)` (the quantities a model may use, variants
  included; `provided_quantities()` of an engine instance). `engine_provided_quantities()`
  is deprecated. `EngineBioCapability.model_reason(model)` and `.machine_view(machine)`
  (the kernel sets the engine reads, a scored quantity as set `scored_<kind>`);
  `engine_bio_capability(engine)` accepts a short name, class, instance or capability.
  `pyRadPlan.dose.engines.get_engine_class(pln_or_mode, engine=None)` resolves the engine
  as `get_engine` does, without instantiating it.

  | Engine | Machine kernels read | Scores | Biological influence |
  |---|---|---|---|
  | HongPB (particle pencil beam) | all (α/β, z\*, LET, fluence) | – | yes |
  | FRED (Monte Carlo) | none | LET (`let:LETd`) | yes, from the scored LET |
  | TOPAS (Monte Carlo) | none | LET (`let:LETd`, protons only: ProtonLET scorer) | yes, from the scored LET |
  | SVDPB (photon pencil beam) | none | – | no |

- `TissueProperties.matches(..., ignore_undeclared=True)` / `find_match(...)` /
  `lookup(...)`: parameters the tissue class leaves NaN are not compared, nor the
  provenance labels (`provenance_fields`: `model`, LEM `version`) unless named in `fields`;
  `find_match` / `lookup` prefer a candidate of the same provenance (`same_provenance`).
  Resolution, the exact lookup and the closest-class suggestion use them.
- Reference parameters are never unset on a structure's tissue. A `TissueProperties` model
  validator resolves a missing or NaN/None `alpha_x` / `beta_x`: explicit value, else the
  provenance's derivation (`derive_reference`; MKM: α_x = α₀ + β₀·z₁D,ref, mMKM with the
  saturation-corrected z\*₁D,ref, β_x = β₀, for the tissue's `reference_radiation`; kept
  derived from the current model parameters unless explicit or published,
  `derives_reference`), else `default_reference_parameters()` (`DEFAULT_ALPHA_X` = 0.1,
  `DEFAULT_BETA_X` = 0.05, the generic bundled classes; the one place to make modality-aware
  later). Base-data values are rule 3, applied by `apply_tissue_class`. `reference_sources`
  maps each reference parameter to its source ∈ `REFERENCE_SOURCES` (`explicit`,
  `published`, `estimated`, `base_data`, `default`; `FIXED_REFERENCE_SOURCES` are never
  derived again); it is a serialized metadata field (alias `referenceSources`, hidden from
  `repr`), excluded from `__eq__` (`TissueProperties.metadata_fields`) and from matching (not
  an identifying field). The `reference_source` property summarizes it (the least specific);
  a single `reference_source` / `referenceSource` given on input (files of earlier versions)
  applies to both and overrides a mapping given besides. A given source is kept for the given
  values (dumps); assignment validation makes an assigned reference parameter `explicit`, the
  other keeping its source, as `voi.alpha_x = v` does.
- Base-data classes are built by `TissueProperties.from_base_data(data, **overrides)`, which
  validates with a context flag that keeps given NaN reference values ("not declared") and
  records `base_data`. Machine `tissueParameters`, legacy kernel columns, ion machines and RBE
  table metadata use it. The flag is not stored on the instance: classes are never
  revalidated (instances pass pydantic validation unchanged), and `cell_line` stays part of
  `==` as before.
- `adopt_provenance(tissue, target) -> ProvenanceAdoption` (`tissue`, `source`, `target`,
  `carried`, `defaulted`, `cell_line`, `changed`, `describe()`), also
  `TissueProperties.adopt_provenance(target)`: shares every parameter between provenances of
  one parameter class (MKM/mMKM, LEM versions), only α_x/β_x across classes.
  `apply_tissue_class(tissue, tissue_class)`: adopts the class provenance (a reference-only
  class of undeclared provenance keeps the tissue's), copies declared identifying parameters
  and, for non-identifying reference parameters, the class's declared values
  (`declared_reference_parameters`) only over `default` / `base_data` ones. The GUI editors
  copy identifying fields themselves so far.
- `TissueIssue.severity` (`ISSUE_SEVERITIES`: `"error"`, `"warning"`; `is_error`,
  `is_warning`), `BioModelResolution.warnings`. `raise_for_tissue_issues(model, issues)`
  raises for errors only and returns the warnings, which `DoseEngineBase._resolve_bio_model`
  logs and reports (`report_warning`).
- `TissueIssue.fix_provenance`: set on a `provenance_differs` warning whose tissue parameter
  class is the matched class's provenance class (MKM vs mMKM, LEM version): the optional
  relabel, named in the message; `apply_provenance_fixes(cst, issues)` returns a relabelled
  copy (StructureSet, VOI list or mapping). `compare_tissue_parameters(model, machine, cst,
  engine=...)` / `BiologicalModel.compare_tissues` -> `list[TissueComparison]`
  (`tissue_class` = matched or closest class, `matched` = the parameters match,
  `provenance_mismatch` = the provenance differs, a warning with `matched` and a
  `type_mismatch` error without, `fix_provenance`, `parameters: ParameterComparison(name,
  declared, value, matches)` without the provenance labels, `mismatched`, `issue`). It
  shares the kernel-set selection (`_select_base_data`) and the per-structure class check
  (`_tissue_class_issue`) with `resolve_bio_model`, so it cannot disagree with
  `check_tissues`.

Deviations and details:

- Tissue classes know their kernel set and column through `KernelSetInfo.tissue_classes`
  rather than through attributes on `TissueProperties`, so a machine class copied onto a
  structure stays equal to a hand-entered one and serializes as before.
- Legacy sets take the provenance their tissue classes declare unanimously (Section 4.4).
- Besides the issue kinds of Section 5.5 there is `base_data` (inconsistent kernel set) and
  `reference_domain` (α_x/β_x valid but outside the model's domain, e.g. β_x = 0 for
  McNamara). Only `invalid_reference` (negative, infinite, NaN) always raises in dose
  engines; `reference_domain` follows `calc_bio_dose` like every other error, as before (it
  stays an error: the model cannot be evaluated outside its domain).
  Each structure gets at most one error (reference, then tissue class, then runtime
  parameters); a `provenance_differs` warning does not hide a later error. A model
  unavailable for the machine reports no per-structure class issues.
- `alpha_beta_tables` describes its RBE table as a kernel set of kind `rbe_table`; the
  table is chosen by `quantity_table_name` or `data_model`, there is no `kernel_set`
  parameter. Tables carry no formalism; a `formalism` entry in their metadata is ignored.
- A LET set stored in another field than `let` resolves, but is reported as
  `model_unavailable` since dose engines read the `let` kernel field only.
- Structures resolving to several kernel sets of one formalism raise `NotImplementedError`
  in the evaluator; the resolution result is shaped per structure already.
