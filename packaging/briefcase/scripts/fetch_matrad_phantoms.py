"""Download the phantoms of a matRad release into a pyRadPlan base data folder.

The desktop installers and the container images ship matRad's phantoms, the patient cases
of the CORT dataset and a box phantom, without keeping them in this repository. Their
builds run

    python fetch_matrad_phantoms.py <folder>

which downloads the phantoms into ``<folder>/phantoms`` and matRad's license and a note on
the origin of the data into ``<folder>``. With ``<folder>`` among the shared base data
folders (``PYRADPLAN_BASE_DATA_PATH``, see :mod:`pyRadPlan.core._base_data`) the phantoms
are listed under File → Open Example Phantom.

Every file is checked against its SHA-256 below, and files already in place with the right
hash are not downloaded again. For a newer matRad release, change MATRAD_RELEASE and the
hashes (``sha256sum`` of the release's ``matRad/phantoms`` files and ``LICENSE.md``).
"""

import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

MATRAD_RELEASE = "v3.2.3"
BASE_URL = f"https://raw.githubusercontent.com/e0404/matRad/{MATRAD_RELEASE}/"

#: File in the matRad repository: (file below the target folder, SHA-256).
FILES = {
    "matRad/phantoms/BOXPHANTOM.mat": (
        "phantoms/BOXPHANTOM.mat",
        "0da18069a74bbbf9fa9fdd8b735bcb99b539c033d43619ff3c0efd1fe5735c95",
    ),
    "matRad/phantoms/HEAD_AND_NECK.mat": (
        "phantoms/HEAD_AND_NECK.mat",
        "6e0c0b55090b3b7ae545a39360ee033b4cd65691c36609626c8f05cd90e1400e",
    ),
    "matRad/phantoms/LIVER.mat": (
        "phantoms/LIVER.mat",
        "793e51be95f36e6e9246b797e7d31d361a7a8d49086de85d2bd32bae69d69066",
    ),
    "matRad/phantoms/PROSTATE.mat": (
        "phantoms/PROSTATE.mat",
        "00713103179b0e4f00f4a96ba39035a0fb5011ee7f3243dd166ec5351aff606f",
    ),
    "LICENSE.md": (
        "matRad-LICENSE.md",
        "b181fe90b43888437bf90289a5992caf6cafd932345e17b7ba69662bb5614be6",
    ),
}

README = f"""\
Phantoms of matRad {MATRAD_RELEASE}
https://github.com/e0404/matRad/tree/{MATRAD_RELEASE}/matRad/phantoms

Distributed with pyRadPlan under matRad's license, see matRad-LICENSE.md.

HEAD_AND_NECK, LIVER and PROSTATE are the patient cases of the CORT dataset, with the CT
images downsampled and the data converted to .mat files by matRad:

  D. Craft, M. Bangert, T. Long, D. Papp, J. Unkelbach. Shared data for intensity
  modulated radiation therapy (IMRT) optimization research: the CORT dataset.
  GigaScience 3, 37 (2014). https://doi.org/10.1186/2047-217X-3-37
  Original data: https://doi.org/10.5524/100110

BOXPHANTOM is matRad's box phantom.

For research and education only, not for clinical use.
"""

ATTEMPTS = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path, sha256: str) -> None:
    partial = target.with_name(target.name + ".part")
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as f:
                digest = hashlib.sha256()
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    digest.update(chunk)
                    f.write(chunk)
            break
        except (urllib.error.URLError, OSError) as error:
            partial.unlink(missing_ok=True)
            if attempt == ATTEMPTS:
                raise RuntimeError(f"Download of {url} failed: {error}") from error
            print(f"  {error}, trying again", file=sys.stderr)
            time.sleep(5 * attempt)
    if digest.hexdigest() != sha256:
        partial.unlink()
        raise RuntimeError(f"{url} has SHA-256 {digest.hexdigest()}, expected {sha256}")
    os.replace(partial, target)


def fetch(folder: Path, base_url: str = BASE_URL) -> None:
    """Download the matRad files into ``folder`` and write the README next to them."""
    for source, (name, sha256) in FILES.items():
        target = folder / name
        if target.is_file() and _sha256(target) == sha256:
            print(f"{target} is up to date")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = base_url.rstrip("/") + "/" + source
        print(f"Downloading {url}")
        _download(url, target, sha256)
    (folder / "README.md").write_text(README, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Download the phantoms into the folder given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path, help="base data folder to download into")
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help="where the matRad repository's files are, e.g. a mirror (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    try:
        fetch(args.folder, args.base_url)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
