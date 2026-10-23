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
| `scripts/sync_version.py` | Copies the version from the root `pyproject.toml` into the Briefcase project. |
| `scripts/collect_licenses.py` | Writes the third-party license notices into the bundle. |
| `scripts/make_icons.py` | Regenerates the `.ico`/`.icns`/`.png` icons and the Windows installer's side panel and banner bitmaps from the pyRadPlan logo SVGs. |
| `scripts/run_briefcase.py` | Runs Briefcase with a workaround for `rcedit` failing on relative paths (see below). |
| `build/`, `dist/`, `logs/` | Briefcase output (ignored by git). |

pyRadPlan itself is installed into the bundle as a wheel built from the repository checkout
(`requires = ["../.."]`), so the bundle always contains the pyRadPlan of the current commit
together with its dependencies. The requirements are installed into the bundle's
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
  architecture of the build machine (`universal_build = false`; the CI runner is Apple
  silicon). `--adhoc-sign` is required for an unsigned build and skips notarization.
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
`python scripts/sync_version.py` rewrites it, `--check` only verifies. Keep in mind that the
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

## CI: the "Desktop installers" workflow

`.github/workflows/installers.yml` runs the steps above on `windows-latest`, `macos-latest`
and `ubuntu-latest` (fail-fast off, Windows with a 90 minute timeout). It is not triggered by
pushes:

- **Manually:** Actions -> "Desktop installers" -> "Run workflow". The installers from `dist/`
  are uploaded as the workflow artifacts `pyRadPlan-windows-installer`,
  `pyRadPlan-macos-installer` and `pyRadPlan-linux-installer` (each also contains the
  `THIRD_PARTY_NOTICES-<platform>.txt`). The `attach_to_release` input additionally uploads
  them to the GitHub release of the selected ref; it only has an effect when the run is
  started from a tag.
- **On a published release:** `release.yml` creates a GitHub release for every `v*` tag. The
  `release: published` event then starts this workflow, which attaches the installers to that
  release (`softprops/action-gh-release` with `files:`, job permission `contents: write`).
  Releases created in other ways (for example by hand in the GitHub UI) trigger it as well.

On failure the Briefcase logs (`packaging/briefcase/logs/`) are uploaded as
`<artifact>-logs`.

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
- An Intel macOS build (`macos-15-intel` runner) if there is demand; `universal_build` would
  double the bundle size and requires universal wheels of all dependencies.
- Code signing and notarization as described above.
