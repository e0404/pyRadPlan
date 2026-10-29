## Release

<!-- Merge request to cut a new release of pyRadPlan into `main`.
     Releases are cut from a release candidate branch `rc/X.Y.Z` branched off `develop`,
     never from `develop` directly: that keeps `develop` open for regular work while the
     candidate is stabilised, reviewed and version-bumped. -->

Version: `vX.Y.Z`

## Type of Release

<!-- Check one. Follows Semantic Versioning (https://semver.org). -->

- [ ] Major (`X.0.0`) — incompatible API changes
- [ ] Minor (`X.Y.0`) — new functionality, backwards compatible
- [ ] Patch (`X.Y.Z`) — backwards compatible bug fixes
- [ ] Pre-release (e.g. `X.Y.Z-rc1`)

## Highlights

<!-- A few bullet points on the most important changes for users in this release.
     Pull from the [Unreleased] section of CHANGELOG.md. -->

## Checklist

<!-- All items must be checked before this MR can be merged. -->

- [ ] This MR targets the `main` branch from `rc/X.Y.Z` (not from `develop`), and `rc/X.Y.Z` was branched from `develop`.
- [ ] Review performed, all resulting fixes applied or explicitly approved, and no open threads remain.
- [ ] Full test suite passes locally (`pytest test`).
- [ ] Documentation builds without errors.
- [ ] The desktop installers build: the `installer-linux` job of the pipeline of `rc/X.Y.Z` (GitLab) and the "Desktop installers" workflow (GitHub, all three platforms) are green.
- [ ] No outstanding deprecations are due to be removed in this version (or they have been removed).
- [ ] The pre-commit hook has been run and all files are formatted (`ruff`).
- [ ] Executed example notebooks committed on `rc/X.Y.Z` once the review is addressed and CI is green (examples are never executed in CI or on Read the Docs): `python docs/execute_examples.py --release` executes all runnable examples without errors into `docs/tutorials/examples/` and stages them (`git add -f`, the folder is git-ignored); commit them on `rc/X.Y.Z`. A local docs build shows their outputs and no "older than its script" warning (rerun `--release <name>` for an example changed afterwards).
- [ ] Last commit before merging, once the release code is final: version bumped in `pyproject.toml`, `CHANGELOG.md` `[Unreleased]` section renamed to `[X.Y.Z] - YYYY-MM-DD` with a fresh empty `[Unreleased]` section added on top, and `CITATION.cff` updated (version, date-released, authors if applicable).

## Post-Merge Actions

<!-- Steps to perform after the MR is merged into `main`. -->

- [ ] Tag the release commit on `main` with an annotated tag `vX.Y.Z` (the tag message becomes part of the release notes).
- [ ] Push the tag to trigger the release workflow (`.github/workflows/release.yml`).
- [ ] Verify the release was published with the correct notes from `CHANGELOG.md` and the tag annotation.
- [ ] Once the "Desktop installers" workflow of the tag has finished, verify the GitHub release carries the Windows (`.msi`), macOS (`.dmg`) and Linux (`.deb`) installers.
- [ ] Once the release is published, verify the PyPI release: version `X.Y.Z` is listed, the sdist and wheel are present, and `pip install pyRadPlan==X.Y.Z` resolves in a clean environment.
- [ ] Verify Read the Docs built the release (`stable` / `vX.Y.Z`) with the example outputs.
- [ ] Merge `main` back into `develop` to keep branches in sync, without the executed notebooks (they are tracked on `main` and the tags only):
  ```bash
  git switch develop
  git merge --no-ff --no-commit main
  git rm -r --cached --quiet --ignore-unmatch docs/tutorials/examples
  git commit
  ```
  `git ls-files docs/tutorials/examples` must print nothing on `develop` afterwards.
- [ ] Delete the `rc/X.Y.Z` branch.

## Additional Notes

<!-- Breaking changes, migration notes for users, known issues, or anything else worth flagging. -->
