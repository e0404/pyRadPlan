"""Copy the pyRadPlan version and changelog from the repository root into the Briefcase project.

The version goes into the Briefcase ``pyproject.toml``. The changelog is copied next to it
as the git-ignored ``CHANGELOG.md``, because Briefcase takes the changelog of a Linux system
package (``.deb``) from the directory of its own ``pyproject.toml`` and has no setting to
point elsewhere.

Usage (from anywhere)::

    python packaging/briefcase/scripts/sync_version.py          # rewrite
    python packaging/briefcase/scripts/sync_version.py --check  # exit 1 if the version is out of sync
"""

import re
import shutil
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRIEFCASE_PYPROJECT = HERE.parent / "pyproject.toml"
ROOT_PYPROJECT = HERE.parents[2] / "pyproject.toml"
ROOT_CHANGELOG = HERE.parents[2] / "CHANGELOG.md"
BRIEFCASE_CHANGELOG = HERE.parent / "CHANGELOG.md"


def main(argv: list[str]) -> int:
    root_version = tomllib.loads(ROOT_PYPROJECT.read_text("utf-8"))["project"]["version"]
    text = BRIEFCASE_PYPROJECT.read_text("utf-8")
    current = tomllib.loads(text)["tool"]["briefcase"]["version"]
    if current == root_version:
        print(f"briefcase version already {root_version}")
    elif "--check" in argv:
        print(f"briefcase version {current} != pyRadPlan {root_version}", file=sys.stderr)
        return 1
    else:
        new_text, n = re.subn(
            r'^version = "[^"]*"', f'version = "{root_version}"', text, count=1, flags=re.M
        )
        assert n == 1
        BRIEFCASE_PYPROJECT.write_text(new_text, "utf-8")
        print(f"briefcase version {current} -> {root_version}")

    if "--check" not in argv:
        shutil.copyfile(ROOT_CHANGELOG, BRIEFCASE_CHANGELOG)
        print(f"changelog copied to {BRIEFCASE_CHANGELOG.relative_to(HERE.parents[2])}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
