## Release

<!-- Merge request to cut a new release of pyRadPlan from `develop` into `main`. -->

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

- [ ] This MR targets the `main` branch from `develop`.
- [ ] Version bumped in `pyproject.toml`.
- [ ] `CHANGELOG.md` `[Unreleased]` section renamed to `[X.Y.Z] - YYYY-MM-DD` and a fresh empty `[Unreleased]` section added on top.
- [ ] `CITATION.cff` updated (version, date-released, authors if applicable).
- [ ] The pre-commit hook has been run and all files are formatted (`ruff`).
- [ ] Full test suite passes locally (`pytest test`).
- [ ] Documentation builds without errors.
- [ ] No outstanding deprecations are due to be removed in this version (or they have been removed).

## Post-Merge Actions

<!-- Steps to perform after the MR is merged into `main`. -->

- [ ] Tag the release commit on `main` with an annotated tag `vX.Y.Z` (the tag message becomes part of the release notes).
- [ ] Push the tag to trigger the GitHub Release workflow (`.github/workflows/release.yml`).
- [ ] Verify the GitHub Release was created with the correct notes from `CHANGELOG.md` and tag annotation.
- [ ] Merge `main` back into `develop` to keep branches in sync.

## Additional Notes

<!-- Breaking changes, migration notes for users, known issues, or anything else worth flagging. -->
