# Rules for Contribution

Everybody is welcome to contribute, still we recommend to get in touch such that we do not double the work.

## Branch Name Rules
On the pyRadPlan repository, we will adhere to the branch rules below. If you plan to contribute, we encourage employing them as well.

There's two predefined branch names for main development
- **main:**         Protected main branch. Only latest release version + hotfixes
- **develop:**      Development branch. Used to prepare patches & minor versions and main target for merge requests.

For individual development work, we will use branch groups. The following groups are allowed
- **bugfix/*:**      Fixing a bug
- **hotfix/*:**      Fixing a breaking change on the main branch
- **feature/*:**     Individual larger novel feature contributions, e.g., feature/myCoolNewDoseCalculation
- **refactor/*:**    An overhaul / refactor of code
- **interface/*:**   Interface to other software
- **devops/*:**      Work on DevOps like the CI pipeline or formatting hooks (e.g., devops/addPython311)
- **dev/*:**         Individual development efforts that are not part of a single group above, or don't have a clear purpose or timeline regarding merging


## Code Style
We try to adhere to PEP conventions as much as possible. This is enforced with a pre-commit hook running ruff.
The hook will be automatically installed when doing an editable install of the developer dependencies: `pip install -e .[dev]`

After successfully running `pip install -e .[dev]` check in the console that `pre-commit --version` provides a correct response. Afterwards run `pre-commit install`. This reads the `.pre-commit_config.yaml` and adds a hook to your git repository, which whenever a commit is made the changed files are reformatted to ensure the PEP standards.

## Preparing a Merge Request
When you think a something is ready to integrate, prepare a merge request. Here's what you should do before submitting:
- Make sure the **pre-commit** hook has formatted your code (see above).
- Run **pytest** run unit tests before publishing the code. Run on the test folder via `pytest test`, or choose any test file following the pytest syntax. We encourage writing at least fundamental unit tests for new code. Will also install `coverage` and the pytest extension to monitor coverage. If coverage drops too much because of insufficient testing, we will (probably) not merge.

Merge requests should (almost) always point to the `develop` branch. Exceptions are hotfixes of a crucial problem.

To finalize the MR, please modify the CHANGELOG.md in the [UNRELEASED] section at the top with your changes. The CHANGELOG.md follows the [Keep A Changelog](https://keepachangelog.com/) conventions.
