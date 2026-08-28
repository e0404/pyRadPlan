## Description

<!-- Provide a clear summary of the changes and the motivation behind them.
     Link the related issue(s) below. -->

Closes #<!-- issue number -->

## Type of Change

<!-- Check all that apply. -->

- [ ] Bug fix (`bugfix/*` branch)
- [ ] Hotfix for a breaking change on `main` (`hotfix/*` branch)
- [ ] New feature (`feature/*` branch)
- [ ] Refactor (`refactor/*` branch)
- [ ] Interface to external software (`interface/*` branch)
- [ ] DevOps / CI (`devops/*` branch)
- [ ] Documentation (documentation infrastructure via `devops/*`, docs changes only via `docs/*`)
- [ ] Other (`dev/*` branch) — please describe:

## Checklist

<!-- All items must be checked before this MR can be merged. -->

- [ ] This MR targets the `develop` branch (or `main` for hotfixes and documentation only).
- [ ] The pre-commit hook has been run and all files are formatted (`ruff`).
- [ ] Unit tests pass locally (`pytest test`).
- [ ] New or changed code is covered by tests; coverage has not dropped significantly.
- [ ] `CHANGELOG.md` has been updated in the `[UNRELEASED]` section following [Keep a Changelog](https://keepachangelog.com/) conventions.
- [ ] If example scripts in `examples/` were added or changed: notebooks re-executed locally (`python docs/execute_examples.py`) and the updated `docs/tutorials/examples/*.ipynb` committed.
- [ ] If new contributor: Authors in  `CITATION.cff` and `pyproject.toml` have been updated

## Testing

<!-- Describe how the changes were tested. Include relevant pytest commands or test file names. -->

## Additional Notes

<!-- Anything else reviewers should know: breaking changes, performance implications, open questions, etc. -->
