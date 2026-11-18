"""Build the artifacts the Pyodide bootstrap installs into ``web/public/pyrp/``.

Writes the pyRadPlan wheel (built with ``pip wheel`` from a clean copy of the sources), the
SimpleITK shim wheel (``web/pyodide/shims``), the vendored pure-Python dependency wheels
pinned in ``bootstrap.js`` (``PURE_WHEELS``) and ``manifest.json`` (file names, versions,
sizes, SHA-256). Needs ``pip`` and network access for the build backend and the wheels::

    python web/pyodide/build.py [--out web/public/pyrp] [--skip-pure]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SHIM_DIR = HERE / "shims" / "SimpleITK"
SHIM_DIST = "pyradplan_sitk_shim"
SHIM_VERSION = "0.1.0"


def sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entry(path: Path, **extra) -> dict:
    """Return the manifest entry (file name, size, hash) of an artifact."""
    return {"file": path.name, "size": path.stat().st_size, "sha256": sha256(path), **extra}


def pure_wheel_pins() -> list[str]:
    """Return the ``name==version`` pins of ``PURE_WHEELS`` in ``bootstrap.js``."""
    source = (HERE / "bootstrap.js").read_text(encoding="utf-8")
    block = re.search(r"export const PURE_WHEELS = \[(.*?)\];", source, re.S)
    if block is None:
        raise RuntimeError("PURE_WHEELS not found in bootstrap.js")
    return re.findall(r'"([^"]+==[^"]+)"', block.group(1))


def build_pyradplan(out: Path) -> Path:
    """Build the pyRadPlan wheel from a clean copy of the sources into ``out``."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        src.mkdir()
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            shutil.copy2(REPO / name, src / name)
        shutil.copytree(
            REPO / "pyRadPlan",
            src / "pyRadPlan",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        dist = Path(tmp) / "dist"
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "-q", "-w", str(dist), str(src)],
            check=True,
        )
        (wheel,) = dist.glob("pyradplan-*.whl")
        target = out / wheel.name
        shutil.copy2(wheel, target)
    return target


def _record_line(name: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{name},sha256={digest},{len(data)}"


def build_shim(out: Path) -> Path:
    """Write the SimpleITK shim as a pure-Python wheel into ``out``."""
    files = {}
    for path in sorted(SHIM_DIR.rglob("*.py")):
        files[f"SimpleITK/{path.relative_to(SHIM_DIR).as_posix()}"] = path.read_bytes()
    info = f"{SHIM_DIST}-{SHIM_VERSION}.dist-info"
    files[f"{info}/METADATA"] = (
        "Metadata-Version: 2.1\n"
        f"Name: {SHIM_DIST.replace('_', '-')}\n"
        f"Version: {SHIM_VERSION}\n"
        "Summary: numpy stand-in for the SimpleITK subset used by pyRadPlan (Pyodide only)\n"
        "Requires-Dist: numpy\nRequires-Dist: scipy\n"
    ).encode()
    files[f"{info}/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: pyradplan-build\nRoot-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
    ).encode()
    files[f"{info}/top_level.txt"] = b"SimpleITK\n"
    record = [_record_line(name, data) for name, data in files.items()]
    record.append(f"{info}/RECORD,,")
    files[f"{info}/RECORD"] = ("\n".join(record) + "\n").encode()
    target = out / f"{SHIM_DIST}-{SHIM_VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zinfo = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            zinfo.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zinfo, data)
    return target


def download_pure(out: Path, pins: list[str]) -> list[dict]:
    """Download the pinned pure-Python wheels into ``out/wheels``."""
    wheel_dir = out / "wheels"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "-q"]
        + ["-d", str(wheel_dir), *pins],
        check=True,
    )
    entries = []
    for pin in pins:
        name, version = pin.split("==")
        stem = re.sub(r"[-_.]+", "_", name).lower()
        matches = [
            p for p in wheel_dir.glob("*.whl") if p.name.lower().startswith(f"{stem}-{version}-")
        ]
        if len(matches) != 1 or not matches[0].name.endswith("-none-any.whl"):
            raise RuntimeError(f"No unique pure wheel for {pin}: {matches}")
        item = entry(matches[0], requirement=pin)
        item["file"] = f"wheels/{matches[0].name}"
        entries.append(item)
    return entries


def main(argv=None) -> None:
    """Build all artifacts and write ``manifest.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "web" / "public" / "pyrp")
    parser.add_argument("--skip-pure", action="store_true", help="keep the vendored wheels")
    args = parser.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.whl"):
        stale.unlink()

    manifest_path = out / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    wheel = build_pyradplan(out)
    version = wheel.name.split("-")[1]
    shim = build_shim(out)
    if args.skip_pure and previous.get("pure_wheels"):
        pure = previous["pure_wheels"]
    else:
        if (out / "wheels").exists():
            shutil.rmtree(out / "wheels")
        pure = download_pure(out, pure_wheel_pins())

    bootstrap = (HERE / "bootstrap.js").read_text(encoding="utf-8")
    pyodide_version = re.search(r'PYODIDE_VERSION = "([^"]+)"', bootstrap).group(1)
    manifest = {
        "pyodide_version": pyodide_version,
        "pyradplan": entry(wheel, version=version),
        "sitk_shim": entry(shim, version=SHIM_VERSION),
        "pure_wheels": pure,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(e["size"] for e in (manifest["pyradplan"], manifest["sitk_shim"], *pure))
    print(f"wrote {manifest_path} ({total / 1e6:.1f} MB of wheels)")


if __name__ == "__main__":
    main()
