# Performance branch scope

2026-11-03. Architectural batching work is separated for independent review.

`refactor/dose-calc-performance` retains the incremental optimizations, including
prepared single-grid interpolation, particle focus/depth and energy/kernel reuse,
the PyTorch resampling correction, and the earlier indexing, cutoff, geometry,
sampling and sparse-output improvements. The scenario changes merged from
`develop` remain intact. The dose engines use their scalar execution loops;
there are no batching hooks or batch configuration fields on this branch.

`refactor/dose-calc-batching` builds on that branch and contains:

- The common two-stage scheduler, delivery metadata and engine lifecycle hooks.
- The opt-in particle adapter, packet transfers and selected-packet reuse.
- `ArrayBackend1DInterpolatorBank` and its optional CuPy fusion, introduced to
  support these packet calculations. Earlier single-grid interpolators remain
  available on both branches.
- Batching-specific tests, benchmark scripts and development reports.

The bank is reusable independently, but currently its consumers are batching
code and experiments. Keeping it with that work makes the incremental branch
smaller without creating dangling imports. Cutoff quadrature and its existing
depth/sample vectorization remain part of the earlier incremental work.

No prior history is rewritten. The separation is recorded on the performance
branch, and the batching branch restores the architectural changes on top of
that shared base. This makes later review or merging possible without relying
on a branch whose changes have already been reverted on its target.

Review the isolated changes with:

```bash
git diff refactor/dose-calc-performance...refactor/dose-calc-batching
```

On the batching branch, `docs/development/production_particle_batching.md` contains
the implementation and measurements. The scalar default and opt-in status there
are unchanged. The branch split makes no new performance claim.

Validation of the lean branch passed 1,313 selected tests covering scalar dose,
biology, scenarios, engine reuse, prepared interpolation, resampling and the
complete GUI test directory. Runtime/test differences from the pre-hook
`51353010` revision are limited to removing the interpolation bank and its tests;
the independent dose and resampling improvements are unchanged.

Generated benchmark JSON results are local artifacts and are not versioned on
either branch. `benchmark/.gitignore` excludes them recursively. Raw-output paths
in historical reports refer to local measurements; the Markdown summaries and
the scripts used to reproduce them remain versioned.
