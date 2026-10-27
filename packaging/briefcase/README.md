# pyRadPlan desktop bundle (Briefcase)

This directory is a [Briefcase](https://briefcase.readthedocs.io/) project that packages the
pyRadPlan GUI as an installable desktop application: a Windows MSI, a macOS DMG and a Linux
`.deb`. The bundle is CPU-only (numpy, scipy, numba; no torch, cupy or jax) and is intended
for research and education, not for clinical use. It is a convenience for people who do not
want to set up a Python environment; `pip install pyRadPlan[gui]` remains the primary way to
install pyRadPlan.

## Layout

| Path | Purpose |
|---|---|
| `pyproject.toml` | The Briefcase project (app metadata, requirements, per-platform settings). Separate from the repository's root `pyproject.toml`, which describes the library. |
| `src/pyradplan_app/` | The app package Briefcase launches. It only calls `pyRadPlan.gui.main()`; the icons live in `resources/`. |
| `tests/pyradplan_app/__main__.py` | Smoke test that runs inside the built bundle (`briefcase run --test`). |
| `scripts/sync_version.py` | Copies the version from the root `pyproject.toml` into the Briefcase project, and the root `CHANGELOG.md` next to it (git-ignored; Briefcase needs a changelog there for the Linux package). |
| `scripts/collect_licenses.py` | Writes the third-party license notices into the bundle. |
| `scripts/make_icons.py` | Regenerates the `.ico`/`.icns`/`.png` icons and the Windows installer's side panel and banner bitmaps from the pyRadPlan logo SVGs. |
| `scripts/run_briefcase.py` | Runs Briefcase with a workaround for `rcedit` failing on relative paths (see below). |
| `build/`, `dist/`, `logs/` | Briefcase output (ignored by git). |

pyRadPlan itself is installed into the bundle as a wheel built from the repository checkout
(`requires = ["../.."]`), so the bundle always contains the pyRadPlan of the current commit
together with its dependencies. The Qt requirements repeat the root project's `gui` extra (a
test keeps them equal). The requirements are installed into the bundle's
`app_packages` directory as regular wheels:

| Platform | `app_packages` inside `build/pyradplan_app/` |
|---|---|
| Windows (`app`) | `windows/app/src/app_packages` |
| macOS (`app`) | `macos/app/pyRadPlan.app/Contents/Resources/app_packages` |
| Linux (`system`, e.g. Ubuntu noble) | `ubuntu/noble/pyradplan_app-<version>/usr/lib/pyradplan_app/app_packages` |

## Building locally

Briefcase picks the Python support package matching the interpreter that runs it; CI uses
Python 3.12. Install it with `pip install briefcase` (CI pins `briefcase==0.4.5`) and run
everything from this directory:

```bash
cd packaging/briefcase
python scripts/sync_version.py            # or --check
briefcase create <platform> <format>      # scaffold + install requirements
briefcase build <platform> <format>
briefcase run <platform> <format> --test  # smoke test inside the bundle
briefcase build <platform> <format> --update   # drop the test sources again
python scripts/collect_licenses.py --check
briefcase package <platform> <format> [--adhoc-sign]
```

`python scripts/run_briefcase.py <briefcase arguments>` runs the same commands through a thin
wrapper. It only matters on Windows, where `rcedit` (the tool Briefcase uses to stamp the
version and icon into the stub executable) fails with `Fatal error: Unable to commit changes`
on some machines when it is given the stub as a relative path, which is what Briefcase does.
The wrapper passes the path absolutely and is otherwise transparent, so CI uses it everywhere.

On Windows, run Briefcase with `PYTHONUTF8=1` set: it re-prints the app's output through its
own console and aborts the smoke test with a `UnicodeEncodeError` on a cp1252 console otherwise.

`<platform> <format>` is one of:

- **Windows:** `windows app`, run on Windows. Produces `dist/pyRadPlan-<version>.msi`, a
  per-user installer (`system_installer = false`, no admin rights). Briefcase downloads the WiX
  toolset on first use. Without `--identity` the MSI is unsigned (Briefcase prints a warning).
- **macOS:** `macOS app`, run on macOS. Produces `dist/pyRadPlan-<version>.dmg` for the
  architecture of the build machine (`universal_build = false`), so Apple silicon and Intel
  Macs need one build each; CI builds both and appends `-arm64` or `-x86_64` to the file
  name. `--adhoc-sign` is required for an unsigned build and skips notarization. Several
  dependencies have no Intel Mac wheels in their latest releases, so the Intel app bundles
  older ones. The GUI dependencies follow the root project's `gui` extra on both
  architectures.
- **Linux:** `linux system`, run on the target distribution. Produces
  `dist/pyradplan-app_<version>-1~<distro>-<codename>_<arch>.deb` (or an RPM/pkg on other
  families). Briefcase must run under the distribution's `/usr/bin/python3` (a venv created
  from it is fine) and needs `python3-dev build-essential dpkg-dev` plus the
  `system_runtime_requires` Qt libraries listed in `pyproject.toml`. Docker builds for other
  distributions (`briefcase create linux system --target ubuntu:jammy`) are possible but not
  used in CI.

The packages are only built for the platform they are built on; there is no cross-compilation.

### Smoke test

`briefcase run <platform> <format> --test` starts `tests/pyradplan_app/__main__.py` inside the
bundle instead of the GUI. It imports pyRadPlan, loads the bundled TG119 phantom, runs a small
photon dose calculation and optimization and builds the main window offscreen
(`QT_QPA_PLATFORM=offscreen` is the default in the test). The result is reported through the
`>>>>>>>>>> EXIT <code> <<<<<<<<<<` line Briefcase looks for, so no test framework is bundled.
Test mode copies `tests/` into the bundle; a following `briefcase build ... --update`
refreshes the app code without the tests (and without reinstalling the requirements).

### Version

`[tool.briefcase].version` must equal the library version in the root `pyproject.toml`.
`python scripts/sync_version.py` rewrites it (and copies the changelog, see above), `--check` only verifies the version. Keep in mind that the
MSI requires a plain `MAJOR.MINOR.PATCH` version; pre-release suffixes do not package on
Windows.

### Third-party license notices

`python scripts/collect_licenses.py` enumerates the distributions in the bundle's
`app_packages` directory through `importlib.metadata` (without importing them) and writes
`THIRD_PARTY_NOTICES.txt`: name, version, declared license, homepage and the license text
shipped in each `*.dist-info`. The file is written next to the dist-info directories, so it
is part of the installer, and copied to `dist/THIRD_PARTY_NOTICES-<platform>.txt` for the CI
artifact. Run it after `briefcase build` and before `briefcase package`. With no argument it
locates the single bundle under `build/`; pass the `app_packages` path to pick one, or
`--output-dir DIR` to write somewhere else without touching the bundle (for example to
inspect a virtual environment's `site-packages`). `--check` fails when a distribution
declares a GPL-only license (GPL/AGPL without an LGPL or permissive alternative), as a
tripwire against pulling in a dependency the bundle cannot redistribute under its terms.

### Qt and the LGPL

The GUI uses PySide6, which Qt offers under the LGPL v3 (or the GPL, or a commercial
license). The bundle keeps Qt as the separate shared libraries of the `PySide6` wheel inside
`app_packages`; nothing is statically linked and the libraries can be replaced in an installed
copy, which is what the LGPL's relinking requirement asks for. The Qt license texts are part
of `THIRD_PARTY_NOTICES.txt`. The collected notices and the pyRadPlan license (Briefcase
copies `../../LICENSE` into the installer metadata) together document the terms of everything
shipped in the bundle.

## CI: building the installers

The installers are built on GitHub (all three platforms) and on the internal GitLab (the Linux
package only). Both build on `main`, on the release candidate branches `rc/*`, on `devops/*`
branches (for trying out changes to the pipelines) and for `v*` tags;
GitHub also builds pull requests into `main`, so the release PR from `rc/X.Y.Z` shows a broken
installer before the release.

### GitHub: the "Desktop installers" workflow

`.github/workflows/installers.yml` runs the steps above on `windows-latest`, `macos-latest`
(Apple silicon), `macos-15-intel` and `ubuntu-24.04` (fail-fast off, Windows with a 90 minute
timeout). The Linux runner is pinned because the `.deb` is built for the distribution it is
built on, and `ubuntu-latest` moves to Ubuntu 26 in October 2026:

- **Pushes to `main`, `rc/**` and `devops/**`, and pull requests into `main`.** A newer push to a branch or
  pull request cancels the older build; builds of `main` and of tags are never cancelled. A
  push to an `rc` branch and the pull request opened from it build different commits (the
  branch head and its merge into `main`), so both run. The installers are uploaded as the
  workflow artifacts `pyRadPlan-windows-installer`, `pyRadPlan-macos-arm64-installer`,
  `pyRadPlan-macos-x86_64-installer` and `pyRadPlan-linux-installer` (each also contains the
  `THIRD_PARTY_NOTICES-<platform>.txt`), kept for 14 days.
- **`v*` tags.** The installers of the tag are uploaded as above (kept for 90 days) and
  attached to its GitHub release (`softprops/action-gh-release` with `files:`, job permission
  `contents: write`). `release.yml` creates that release for the same tag; it does not matter
  which of the two workflows gets there first. The trigger is the tag push and not the
  `release: published` event, because `release.yml` creates the release with the workflow's
  `GITHUB_TOKEN`, and events caused by that token do not start other workflows.
- **Manually:** Actions -> "Desktop installers" -> "Run workflow". The `attach_to_release`
  input additionally uploads the installers to the GitHub release of the selected ref; it only
  has an effect when the run is started from a tag.

On failure the Briefcase logs (`packaging/briefcase/logs/`) are uploaded as
`<artifact>-logs`.

### GitLab: the `installer-linux` job

The job `installer-linux` of `.gitlab-ci.yml` (stage `package`, so it runs once the tests pass)
builds the Linux `.deb` in an `ubuntu:24.04` container with the same Briefcase steps, for
`main`, `rc/*` and `devops/*` branches and tags. The runners are Docker runners, so the Windows MSI and the
macOS DMG are only built on GitHub. The package is not kept as a job artifact (the runners
limit their size and the `.deb` is almost 300 MB): the job only shows that the installer
builds, and keeps the Briefcase logs for a week when it fails. A merge request from `rc/X.Y.Z` into `main` shows the pipeline of its source
branch, which includes this job; the job has no `merge_request_event` rule on purpose, as
that would create a second, installer-only pipeline which GitLab then reports as the merge
request's pipeline in place of the one that runs the tests.

## Code signing (not yet done)

All current builds are unsigned (Windows, Linux) or ad-hoc signed (macOS). Users see the
corresponding warnings when installing.

- **Windows:** an Authenticode certificate is needed (an OV or EV code signing certificate
  from a CA, or Azure Trusted Signing). Briefcase signs with `signtool` from the Windows SDK:
  `briefcase package windows app --identity <SHA-1 thumbprint of the certificate>`, optionally
  with `--cert-store`, `--use-local-machine-stores`, `--file-digest` and `--timestamp-url`
  (Briefcase 0.4.5 has no `pyproject.toml` key for the identity; it is a command line option). Unsigned MSIs trigger a Microsoft Defender SmartScreen warning; even signed
  ones do until the certificate has built up reputation (EV certificates are trusted at
  once).
- **macOS:** a "Developer ID Application" certificate from an Apple Developer account,
  installed in the keychain of the build machine. `briefcase package macOS app --identity
  "Developer ID Application: <name> (<team id>)"` signs the app and the DMG and then
  notarizes them with `xcrun notarytool`; the notarization credentials (an App Store Connect
  API key, or an Apple ID with an app-specific password) must be stored in the keychain first
  with `xcrun notarytool store-credentials briefcase-macOS-<team id>`. `--no-notarize` skips
  the upload. Without notarization Gatekeeper refuses the app on current macOS versions
  unless the user removes the quarantine attribute.
- **Linux:** a `.deb` needs no signature. If the package is distributed through an apt
  repository, that repository's index is signed with GPG; the package file itself is not.

In CI the certificates and credentials would have to be stored as GitHub secrets and imported
into the runner's certificate store or keychain before the `briefcase package` step.

## Follow-ups

- AppImage (`linux appimage`, configured in `pyproject.toml` but untested) and Flatpak builds
  for distribution-independent Linux packages.
- A single universal macOS build instead of one per architecture: `universal_build` would
  double the bundle size and requires universal wheels of all dependencies.
- Code signing and notarization as described above.
