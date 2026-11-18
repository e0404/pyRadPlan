"""Collect the license notices of every distribution installed in a Briefcase bundle.

Briefcase installs the app's requirements as regular wheels into the bundle's
``app_packages`` directory, so each dependency ships with its ``*.dist-info``
metadata and license files. This script gathers them into one
``THIRD_PARTY_NOTICES.txt`` that is written next to the dist-info folders (so it
ends up inside the installer) and copied to ``packaging/briefcase/dist/`` as
``THIRD_PARTY_NOTICES-<platform>.txt`` for the CI artifact.

Run it after ``briefcase build`` and before ``briefcase package``::

    python packaging/briefcase/scripts/collect_licenses.py            # auto-detect the bundle
    python packaging/briefcase/scripts/collect_licenses.py <app_packages dir>
    python packaging/briefcase/scripts/collect_licenses.py --check    # also fail on GPL-only

``--output-dir DIR`` writes the notices file only to DIR and leaves the
``app_packages`` directory untouched (useful for inspecting a virtual environment's
``site-packages``).

The distributions are enumerated through ``importlib.metadata`` without
importing them, so the script runs with any Python 3.11+ interpreter, not only
the one inside the bundle.
"""

import argparse
import re
import shutil
import sys
import tomllib
from importlib.metadata import Distribution, PackagePath, distributions
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRIEFCASE_DIR = HERE.parent
BUILD_DIR = BRIEFCASE_DIR / "build"
DIST_DIR = BRIEFCASE_DIR / "dist"
NOTICES_NAME = "THIRD_PARTY_NOTICES.txt"

# Distributions built from this repository; their license is the project's own.
FIRST_PARTY = {"pyradplan"}

LICENSE_FILE_RE = re.compile(r"^(LICEN[CS]E|COPYING|NOTICE|AUTHORS)", re.IGNORECASE)
# "GPL" that is not the tail of "LGPL" (AGPL counts as GPL-only here).
GPL_RE = re.compile(r"(?<!L)GPL", re.IGNORECASE)
PERMISSIVE_RE = re.compile(
    r"LGPL|MPL|MIT|BSD|Apache|PSF|Python Software Foundation", re.IGNORECASE
)

RULE = "=" * 79
THIN_RULE = "-" * 79


def normalize(name: str) -> str:
    """Normalize a distribution name PEP 503 style."""
    return re.sub(r"[-_.]+", "-", name).lower()


def detect_app_packages() -> Path:
    """Find the app_packages directory of the (single) bundle under build/.

    Every Briefcase bundle contains a rendered ``briefcase.toml`` whose
    ``[paths].app_packages_path`` is relative to that file, for all platforms
    (``build/<app>/<platform>/<format>/`` or ``build/<app>/<vendor>/<codename>/``
    for Linux system packages).
    """
    found = []
    for toml_path in sorted(BUILD_DIR.glob("*/*/*/briefcase.toml")):
        paths = tomllib.loads(toml_path.read_text("utf-8")).get("paths", {})
        app_packages = toml_path.parent / paths.get("app_packages_path", "app_packages")
        if app_packages.is_dir():
            found.append(app_packages)
    if not found:
        sys.exit(f"No built bundle with an app_packages directory found under {BUILD_DIR}")
    if len(found) > 1:
        listing = "\n  ".join(str(p) for p in found)
        sys.exit(
            f"Several bundles found, pass the app_packages directory explicitly:\n  {listing}"
        )
    return found[0]


def platform_label(app_packages: Path) -> str:
    """``windows``/``macos``/``ubuntu``... from ``build/<app>/<platform>/...``, else ``local``."""
    try:
        parts = app_packages.resolve().relative_to(BUILD_DIR.resolve()).parts
    except ValueError:
        return "local"
    return parts[1] if len(parts) > 1 else "local"


def license_of(dist: Distribution) -> str:
    """Return the declared license from ``License-Expression``, ``License`` or the classifiers."""
    meta = dist.metadata
    expression = meta.get("License-Expression")
    if expression:
        return expression.strip()
    classifiers = [
        c.split("::")[-1].strip()
        for c in meta.get_all("Classifier", [])
        if c.startswith("License ::")
    ]
    license_field = (meta.get("License") or "").strip()
    # Older packages put the full license text into the License field; prefer the
    # classifiers then and keep the field only when it is a short identifier.
    if license_field and "\n" not in license_field and len(license_field) <= 120:
        return license_field
    if classifiers:
        return " / ".join(dict.fromkeys(classifiers))
    return "unknown"


def homepage_of(dist: Distribution) -> str:
    """Return the project homepage from ``Project-URL`` or ``Home-page`` (empty if none)."""
    meta = dist.metadata
    urls = {}
    for entry in meta.get_all("Project-URL", []):
        label, _, url = entry.partition(",")
        urls[label.strip().lower()] = url.strip()
    for key in ("homepage", "home", "home-page", "source", "repository", "documentation"):
        if key in urls:
            return urls[key]
    home_page = (meta.get("Home-page") or "").strip()
    if home_page:
        return home_page
    return next(iter(urls.values()), "")


def _dist_info_files(dist: Distribution) -> list[PackagePath]:
    files = dist.files or []
    return [f for f in files if f.parts and f.parts[0].endswith((".dist-info", ".egg-info"))]


def license_files(dist: Distribution) -> list[PackagePath]:
    """License files inside the dist-info directory, declared ones first."""
    inside = _dist_info_files(dist)
    declared = {name.replace("\\", "/") for name in dist.metadata.get_all("License-File", [])}
    chosen = []
    for f in inside:
        rel = Path(*f.parts[1:]).as_posix()
        if (
            rel in declared or rel.startswith("licenses/") or LICENSE_FILE_RE.match(Path(rel).name)
        ) and not rel.endswith((".pyc", "RECORD", "METADATA", "WHEEL")):
            chosen.append(f)
    return sorted(
        set(chosen), key=lambda f: (Path(*f.parts[1:]).as_posix() not in declared, str(f))
    )


def read_license_text(dist: Distribution, file: PackagePath) -> str:
    """Read a license file from the dist-info directory, tolerating non-UTF-8 content."""
    relative = Path(*file.parts[1:]).as_posix()
    try:
        text = dist.read_text(relative)
    except UnicodeDecodeError:
        text = None
    if text is None:
        text = file.locate().read_bytes().decode("utf-8", errors="replace")
    return text.strip()


def is_gpl_only(license_text: str) -> bool:
    """Flag GPL/AGPL licenses that offer no LGPL or permissive alternative (heuristic)."""
    return bool(GPL_RE.search(license_text)) and not PERMISSIVE_RE.search(license_text)


def collect(app_packages: Path) -> tuple[str, int, list[str], list[str]]:
    """Render the notices file; return (text, count, names without text, GPL-only names)."""
    dists = sorted(
        (d for d in distributions(path=[str(app_packages)]) if d.metadata.get("Name")),
        key=lambda d: normalize(d.metadata["Name"]),
    )
    chunks = [
        "Third-party software notices",
        "",
        "This application bundles the following Python distributions. Each entry lists",
        "the distribution name, version, declared license and homepage, followed by the",
        "license text shipped in its package metadata. The complete package metadata is",
        "kept in the *.dist-info directories next to this file.",
        "",
    ]
    missing, gpl_only, count = [], [], 0
    seen = set()
    for dist in dists:
        name = dist.metadata["Name"]
        key = normalize(name)
        if key in FIRST_PARTY or key in seen:
            continue
        seen.add(key)
        count += 1
        lic = license_of(dist)
        if is_gpl_only(lic):
            gpl_only.append(f"{name} {dist.version}: {lic}")
        chunks += [RULE, f"{name} {dist.version}", f"License: {lic}"]
        homepage = homepage_of(dist)
        if homepage:
            chunks.append(f"Homepage: {homepage}")
        chunks.append(THIN_RULE)
        files = license_files(dist)
        if not files:
            missing.append(name)
            chunks.append("No license text found in the distribution.")
        for i, file in enumerate(files):
            if i:
                chunks.append(THIN_RULE)
            chunks.append(f"[{Path(*file.parts[1:]).as_posix()}]")
            chunks.append(read_license_text(dist, file))
        chunks.append("")
    return "\n".join(chunks) + "\n", count, missing, gpl_only


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "app_packages",
        nargs="?",
        type=Path,
        help="bundle app_packages directory (default: the single bundle under build/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=f"write {NOTICES_NAME} only to this directory and leave app_packages untouched",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit with status 1 if a distribution declares a GPL-only license",
    )
    args = parser.parse_args(argv)

    app_packages = (args.app_packages or detect_app_packages()).resolve()
    if not app_packages.is_dir():
        parser.error(f"not a directory: {app_packages}")

    text, count, missing, gpl_only = collect(app_packages)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        written = [args.output_dir / NOTICES_NAME]
        written[0].write_text(text, "utf-8")
    else:
        bundled = app_packages / NOTICES_NAME
        bundled.write_text(text, "utf-8")
        DIST_DIR.mkdir(parents=True, exist_ok=True)
        copy = DIST_DIR / f"THIRD_PARTY_NOTICES-{platform_label(app_packages)}.txt"
        shutil.copyfile(bundled, copy)
        written = [bundled, copy]

    print(
        f"{count} distributions, {len(missing)} without license text"
        + (f" ({', '.join(missing)})" if missing else "")
        + " -> "
        + ", ".join(str(p) for p in written)
    )
    if gpl_only:
        print("GPL-only licensed distributions:\n  " + "\n  ".join(gpl_only), file=sys.stderr)
        if args.check:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
