"""Convert codespell text output (stdin) to GitLab Code Quality JSON (stdout)."""

import hashlib
import json
import re
import sys

LINE_RE = re.compile(r"(.+?):(\d+): (.+?) ==> (.+)")


def main() -> None:
    """Convert codespell text output from stdin to GitLab Code Quality JSON format on stdout."""
    items = []
    for line in sys.stdin:
        m = LINE_RE.match(line.rstrip())
        if not m:
            continue
        path, lineno, word, sugg = m.groups()
        fp = hashlib.md5(f"{path}:{lineno}:{word}".encode()).hexdigest()
        items.append(
            {
                "description": f"Possible misspelling: {word.strip()!r} -> {sugg.strip()!r}",
                "check_name": "codespell",
                "fingerprint": fp,
                "severity": "minor",
                "location": {
                    "path": path.removeprefix("./"),
                    "lines": {"begin": int(lineno)},
                },
            }
        )
    json.dump(items, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
