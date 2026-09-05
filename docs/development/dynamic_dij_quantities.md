# Dynamic quantities on `Dij`

Status: design note only; no dynamic storage API is implemented yet.

## Motivation

`Dij` currently declares every influence matrix as a Pydantic field. This is simple and gives
good validation, but adding a matrix for a new physical or biological methodology requires edits
throughout `Dij`, dose-engine allocation, namespace conversion, beam composition, serialization,
and result assembly.

A biological-only dictionary would solve the immediate case but would establish a second storage
path before the general `Dij` design is settled. The same problem can arise for uncertainty,
imaging, or other derived influence quantities, so the eventual mechanism should be generic.

## Recommended direction

Use one canonical mapping of influence-matrix containers on `Dij`, validated against a small,
central registry. Keep biological-model registration and fluence-dependent quantity registration
separate from this low-level storage registry.

Conceptually:

```python
@dataclass(frozen=True)
class DijQuantitySpec:
    identifier: str
    matrad_name: str | None = None
    aliases: tuple[str, ...] = ()
    export_to_matrad: bool = False


register_dij_quantity(DijQuantitySpec("physical_dose", matrad_name="physicalDose"))
register_dij_quantity(DijQuantitySpec("alpha_dose", matrad_name="alphaDose"))
register_dij_quantity(DijQuantitySpec("rbe_dose"))
```

The eventual data model could expose:

```python
class Dij(...):
    matrices: dict[str, InfluenceMatrixContainer]

    def get_matrix(self, identifier: str) -> InfluenceMatrixContainer | None: ...
    def set_matrix(self, identifier: str, value: Any) -> None: ...
```

All matrix-generic operations would iterate over `matrices`: validation, backend conversion,
scenario selection, sparse-index sharing, beam composition, and multiplication by fluence.

The registry answers only whether and how a matrix may be stored. It should not implement the
meaning of a computed quantity. For example, normalization of `let_dose` or the nonlinear LQ
effect belongs to the quantity layer and its dependency graph, not to `Dij` storage metadata.
This separation also avoids a circular import from `dij` into `quantities`.

## Registration behavior

- Registration happens before constructing or validating a `Dij` that uses the identifier.
- Identifiers are non-empty canonical snake-case strings.
- Duplicate identifiers and aliases fail immediately.
- Unregistered matrix names are rejected by default, catching misspellings at the boundary.
- A registration may explicitly describe matRad import/export. No implicit export of unknown
  matrices should occur.
- Model registration may offer a convenience hook that registers both its `Dij` matrix specs and
  its higher-level quantity implementations, but these remain distinct registries internally.

## Compatibility strategy

A staged migration avoids changing every caller at once:

1. Introduce the registry and register the five existing matrix identifiers without changing
   `Dij` storage.
2. Add the canonical `matrices` mapping and move generic operations to `get_matrix()` and
   `set_matrix()`.
3. Keep `physical_dose`, `let_dose`, `alpha_dose`, and similar names as compatibility properties
   or validated constructor aliases backed by the mapping.
4. Move result construction to registered `FluenceDependentQuantity` implementations so `Dij`
   no longer needs hard-coded intensive/extensive biological rules.
5. Deprecate direct field-based matrix storage only after matRad and GUI compatibility are
   verified.

During migration there must be one source of truth. A matrix should not be independently mutable
through both a field and the mapping.

Validation must use the complete set of quantities planned for the calculation, rather than the
keys already allocated on a partially initialized `Dij`. Particle pencil-beam setup currently
validates biological outputs before allocating `let_dose`; intersecting only with the dictionary
at that point would miss a future collision.

## Why not dynamically add Pydantic fields?

Mutating `Dij.model_fields` after class creation makes validation and generated schemas depend on
import order, complicates serialization, and is difficult to make reliable for plugins. A
registry-validated mapping keeps the Pydantic schema stable while allowing the contents to be
extended.

## Open decisions

- Whether unknown registered matrices should survive matRad/HDF5 round trips as pyRadPlan-only
  metadata or be rejected by those exporters.
- Whether registration is process-global or scoped to an application/plugin registry.
- How vector- or tensor-valued quantities should describe their matrix shape.
- Whether a quantity may provide multiple stored sufficient statistics.
- How schema versioning records the registered identifier and its semantics.
- Whether native attribute compatibility is permanent or only a deprecation bridge.

Until this is implemented, biological evaluators may declare their additive influence outputs,
but dose engines accept only the `alpha_dose` and `sqrt_beta_dose` fields already present on
`Dij`. Unsupported names fail before dose calculation begins.
