"""Copy the pyRadPlan version from the root pyproject.toml into the Briefcase project.

Usage (from anywhere)::

    python packaging/briefcase/scripts/sync_version.py          # rewrite
    python packaging/briefcase/scripts/sync_version.py --check  # exit 1 if out of sync
"""

import re
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRIEFCASE_PYPROJECT = HERE.parent / "pyproject.toml"
ROOT_PYPROJECT = HERE.parents[2] / "pyproject.toml"


def main(argv: list[str]) -> int:
    root_version = tomllib.loads(ROOT_PYPROJECT.read_text("utf-8"))["project"]["version"]
    text = BRIEFCASE_PYPROJECT.read_text("utf-8")
    current = tomllib.loads(text)["tool"]["briefcase"]["version"]
    if current == root_version:
        print(f"briefcase version already {root_version}")
        return 0
    if "--check" in argv:
        print(f"briefcase version {current} != pyRadPlan {root_version}", file=sys.stderr)
        return 1
    new_text, n = re.subn(
        r'^version = "[^"]*"', f'version = "{root_version}"', text, count=1, flags=re.M
    )
    assert n == 1
    BRIEFCASE_PYPROJECT.write_text(new_text, "utf-8")
    print(f"briefcase version {current} -> {root_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
