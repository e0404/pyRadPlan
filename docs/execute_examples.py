"""Execute the example scripts locally and store them as notebooks with outputs.

The notebooks are written to ``docs/tutorials/examples`` (tracked in git) and rendered by the
Sphinx build without any execution on the documentation build system. Run this locally when
examples change and as part of the release recipe, then commit the updated notebooks. It is
never run in CI. Examples needing external tools are skipped.

Usage::

    python docs/execute_examples.py                       # all runnable examples
    python docs/execute_examples.py pencilbeam_proton     # only selected examples
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import pathlib
import sys
import time

import jupytext
import nbformat
from nbclient import NotebookClient

DOCS_DIR = pathlib.Path(__file__).resolve().parent
EXAMPLES_DIR = DOCS_DIR.parent / "examples"
OUTPUT_DIR = DOCS_DIR / "tutorials" / "examples"

NOTEBOOK_METADATA = {  # lets myst-nb pick the python lexer for notebooks converted without execution
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
}


def convert_without_outputs(py_file, target):
    nb = jupytext.read(py_file)
    nb.metadata.update(NOTEBOOK_METADATA)
    jupytext.write(nb, target)


# Examples that cannot run unattended (glob pattern -> reason)
SKIP = {
    "gui_*": "needs an interactive Qt session",
    "mc_*": "needs an external Monte Carlo engine (FRED / TOPAS)",
    "utils_matrad": "needs MATLAB or Octave",
    "*ai_agent*": "needs AI API credentials",
}


def skip_reason(stem: str) -> str | None:
    for pattern, reason in SKIP.items():
        if fnmatch.fnmatch(stem, pattern):
            return reason
    return None


def execute(py_file: pathlib.Path, output_dir: pathlib.Path) -> None:
    # Kernels inherit the environment: no interactive GUI, static (inline) figures instead
    os.environ["PYRADPLAN_GUI_DISABLED"] = "1"
    os.environ.pop("MPLBACKEND", None)  # keep ipykernel's inline backend so figures are captured
    nb = jupytext.read(py_file)
    client = NotebookClient(
        nb,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(py_file.parent)}},
    )
    client.execute()
    nbformat.write(nb, output_dir / f"{py_file.stem}.ipynb")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("examples", nargs="*", help="example names (without .py); default: all")
    parser.add_argument("--output-dir", type=pathlib.Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--force", action="store_true", help="also run explicitly named examples listed in SKIP"
    )
    args = parser.parse_args(argv)

    if args.examples:
        py_files = [EXAMPLES_DIR / f"{name}.py" for name in args.examples]
    else:
        py_files = sorted(EXAMPLES_DIR.glob("*.py"))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for py_file in py_files:
        if not py_file.is_file():
            print(f"[missing] {py_file}")
            failed.append(py_file.stem)
            continue
        reason = skip_reason(py_file.stem)
        if reason and not (args.force and args.examples):
            print(f"[skip]    {py_file.stem}: {reason} - converted without outputs")
            convert_without_outputs(py_file, args.output_dir / f"{py_file.stem}.ipynb")
            continue
        print(f"[run]     {py_file.stem} ...", flush=True)
        start = time.perf_counter()
        try:
            execute(py_file, args.output_dir)
        except Exception as exc:  # noqa: BLE001 - report and continue with the next example
            print(f"[FAILED]  {py_file.stem}: {type(exc).__name__}: {exc}")
            failed.append(py_file.stem)
            continue
        print(f"[done]    {py_file.stem} ({time.perf_counter() - start:.0f} s)")

    if failed:
        print(f"\n{len(failed)} example(s) failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
