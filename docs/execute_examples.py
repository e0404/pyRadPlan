"""Turn the example scripts into the notebooks of the documentation, and execute them.

The scripts in ``examples/`` (jupytext percent format) are the source of the examples. The
Sphinx build (``docs/conf.py``) converts them with :func:`write_notebooks` into
``docs/tutorials/examples`` (git-ignored) and renders them without executing anything, on Read
the Docs as in CI.

Executed notebooks stay tracked from one release to the next and only change on release
commits: on the ``rc/X.Y.Z`` branch the release manager runs
``python docs/execute_examples.py --release``, which executes the examples locally and writes
them into ``docs/tutorials/examples`` with their outputs, tagged with the hash of their
script. The docs build renders such a notebook as is while its script is
unchanged, and replaces it by a conversion without outputs (with a warning) once the script
changed. Examples matching :data:`SKIP` need tools a build system does not have and are never
executed.

Without ``--release`` the examples are executed into ``docs/_build/executed_examples``
(git-ignored), e.g. to check that changed examples still run.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import time

import jupytext
import nbformat
from nbclient import NotebookClient

DOCS_DIR = pathlib.Path(__file__).resolve().parent
REPO_DIR = DOCS_DIR.parent
EXAMPLES_DIR = REPO_DIR / "examples"
NOTEBOOKS_DIR = DOCS_DIR / "tutorials" / "examples"
OUTPUT_DIR = DOCS_DIR / "_build" / "executed_examples"

NOTEBOOK_METADATA = {  # lets myst-nb pick the kernel and the python lexer
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
}
# Notebook metadata key recording the script an executed notebook was produced from
SOURCE_KEY = "pyradplan_example"

# Examples that cannot run unattended (glob pattern -> reason)
SKIP = {
    "gui_*": "needs an interactive Qt session",
    "mc_*": "needs an external Monte Carlo engine (FRED / TOPAS)",
    "utils_matrad": "needs MATLAB or Octave",
    "*ai_agent*": "needs AI API credentials",
    "*ai_model*": "downloads model repositories from the HuggingFace Hub",
}


def skip_reason(stem: str) -> str | None:
    """Why the example ``stem`` cannot run unattended, None if it can."""
    for pattern, reason in SKIP.items():
        if fnmatch.fnmatch(stem, pattern):
            return reason
    return None


def set_example_environment() -> None:
    """
    Prepare the environment kernels started from this process inherit.

    The examples must not open the interactive GUI (which would block until closed), and
    figures have to stay on ipykernel's inline backend to be captured as outputs.
    """
    os.environ["PYRADPLAN_GUI_DISABLED"] = "1"
    os.environ.pop("MPLBACKEND", None)


def source_hash(py_file: pathlib.Path) -> str:
    """SHA-256 of an example script, independent of the checkout's line endings."""
    text = py_file.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def executed_source_hash(notebook: pathlib.Path) -> str | None:
    """Script hash recorded in an executed notebook, None for notebooks without outputs."""
    try:
        metadata = json.loads(notebook.read_text(encoding="utf-8")).get("metadata", {})
    except (OSError, ValueError):
        return None
    return metadata.get(SOURCE_KEY, {}).get("sha256")


def to_notebook(py_file: pathlib.Path) -> nbformat.NotebookNode:
    """Read an example script as a notebook without outputs."""
    nb = jupytext.read(py_file)
    nb.metadata.update(NOTEBOOK_METADATA)
    for i, cell in enumerate(nb.cells):  # jupytext draws random ids, rewriting every notebook
        cell.id = f"cell-{i}"
    return nb


def write_notebooks(target_dir: pathlib.Path = NOTEBOOKS_DIR) -> tuple[list[str], list[str]]:
    """
    Provide a notebook for every example script in ``target_dir``.

    Executed notebooks (written by ``--release``) are kept while their script is unchanged.
    Every other example, including those whose executed notebook is stale, is converted into
    a notebook without outputs. Notebooks are only rewritten when their content changes,
    keeping incremental Sphinx builds incremental, and notebooks whose script was removed are
    deleted.

    Returns
    -------
    tuple[list[str], list[str]]
        Names of the notebooks written, and of the stale executed notebooks among them.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    scripts = sorted(EXAMPLES_DIR.glob("*.py"))
    written, stale = [], []
    for py_file in scripts:
        target = target_dir / f"{py_file.stem}.ipynb"
        executed_hash = executed_source_hash(target) if target.is_file() else None
        if executed_hash == source_hash(py_file):
            continue
        text = jupytext.writes(to_notebook(py_file), fmt="ipynb")
        if not target.is_file() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8")
            written.append(target.name)
            if executed_hash is not None:
                stale.append(target.name)
    stems = {py_file.stem for py_file in scripts}
    for orphan in target_dir.glob("*.ipynb"):
        if orphan.stem not in stems:
            orphan.unlink()
    return written, stale


def _path_pattern(paths: list[pathlib.Path]) -> re.Pattern:
    """Match the given paths as printed, with forward slashes, or escaped in a repr."""
    variants = set()
    for path in paths:
        native = str(path)
        variants.update({native, native.replace("\\", "/"), native.replace("\\", "\\\\")})
    alternatives = "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True))
    return re.compile(alternatives, re.IGNORECASE if os.name == "nt" else 0)


def _scrub(value, replacements):
    if isinstance(value, str):
        for pattern, placeholder in replacements:
            value = pattern.sub(placeholder, value)
        return value
    if isinstance(value, list):
        return [_scrub(v, replacements) for v in value]
    if isinstance(value, dict):
        return {k: _scrub(v, replacements) for k, v in value.items()}
    return value


def scrub_outputs(nb: nbformat.NotebookNode) -> None:
    """
    Remove local paths and user names from the outputs of an executed notebook.

    The repository path becomes ``<repo>`` and the home directory ``~``. Outputs on stderr
    (warnings, log records) are dropped: the documentation does not show them, and they are
    the main source of absolute paths.
    """
    checkouts = [REPO_DIR, REPO_DIR.resolve()]
    spec = importlib.util.find_spec("pyRadPlan")
    if spec is not None and spec.origin is not None:
        installed = pathlib.Path(spec.origin).resolve().parent.parent
        if (installed / "pyproject.toml").is_file():  # editable install of another checkout
            checkouts.append(installed)
    replacements = [
        (_path_pattern(checkouts), "<repo>"),
        (_path_pattern([pathlib.Path.home(), pathlib.Path.home().resolve()]), "~"),
    ]
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        outputs = [
            out
            for out in cell.get("outputs", [])
            if not (out.output_type == "stream" and out.get("name") == "stderr")
        ]
        cell.outputs = [nbformat.from_dict(_scrub(dict(out), replacements)) for out in outputs]


def execute(py_file: pathlib.Path) -> nbformat.NotebookNode:
    """
    Execute an example script as a notebook, ready to be rendered as is by the docs.

    The outputs are scrubbed of local paths, and the notebook records the hash of its script
    and opts out of execution by myst-nb.
    """
    set_example_environment()
    nb = to_notebook(py_file)
    client = NotebookClient(
        nb,
        timeout=None,
        kernel_name="python3",
        record_timing=False,
        resources={"metadata": {"path": str(py_file.parent)}},
    )
    client.execute()
    nb.metadata.update(NOTEBOOK_METADATA)  # nbclient adds interpreter details
    nb.metadata["mystnb"] = {"execution_mode": "off"}
    nb.metadata[SOURCE_KEY] = {
        "script": py_file.relative_to(REPO_DIR).as_posix(),
        "sha256": source_hash(py_file),
    }
    scrub_outputs(nb)
    return nb


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=REPO_DIR, capture_output=True, text=True, check=False
    )


def stage(notebooks: list[pathlib.Path]) -> None:
    """Stage executed notebooks for the release commit, on ``rc/*`` branches only."""
    branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if not branch.startswith("rc/"):
        print(
            f"\nNot staged: '{branch or 'no git checkout'}' is not a release candidate branch "
            "(rc/X.Y.Z); executed notebooks are only committed on release commits."
        )
        return
    paths = [nb.relative_to(REPO_DIR).as_posix() for nb in notebooks]
    result = _git("add", "--force", "--", *paths)
    if result.returncode:
        print(f"\ngit add failed: {result.stderr.strip()}")
        return
    print(f"\nStaged {len(paths)} executed notebook(s) on {branch}; commit them there.")


def main(argv: list[str] | None = None) -> int:
    """Execute the selected (default: all runnable) examples, non-zero exit on failures."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python docs/execute_examples.py                     check that all runnable examples run
  python docs/execute_examples.py pencilbeam_proton   check selected examples
  python docs/execute_examples.py --force mc_fred     run a skipped example, tools installed
  python docs/execute_examples.py --release           release step on rc/X.Y.Z (see below)

--release executes the examples into docs/tutorials/examples, where the docs build renders
them with their outputs, and on an rc/* branch stages them with 'git add --force' (the folder
is git-ignored) for the release commit. Examples listed in SKIP are not executed and stay
without outputs. Outputs are scrubbed of the repository and home paths. Never commit changes
to executed notebooks outside of release commits.""",
    )
    parser.add_argument("examples", nargs="*", help="example names (without .py); default: all")
    parser.add_argument(
        "--release",
        action="store_true",
        help="write the executed notebooks into docs/tutorials/examples and stage them on rc/* "
        "branches",
    )
    parser.add_argument(
        "--no-stage", action="store_true", help="with --release: do not stage the notebooks"
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="where to write the executed notebooks (default: docs/_build/executed_examples)",
    )
    parser.add_argument(
        "--force", action="store_true", help="also run explicitly named examples listed in SKIP"
    )
    args = parser.parse_args(argv)
    if args.release and args.output_dir is not None:
        parser.error("--release writes to docs/tutorials/examples, drop --output-dir")
    output_dir = NOTEBOOKS_DIR if args.release else (args.output_dir or OUTPUT_DIR)

    if args.examples:
        py_files = [EXAMPLES_DIR / f"{name}.py" for name in args.examples]
    else:
        py_files = sorted(EXAMPLES_DIR.glob("*.py"))

    output_dir.mkdir(parents=True, exist_ok=True)

    failed, executed = [], []
    total = time.perf_counter()
    for py_file in py_files:
        if not py_file.is_file():
            print(f"[missing] {py_file}")
            failed.append(py_file.stem)
            continue
        reason = skip_reason(py_file.stem)
        if reason and not (args.force and args.examples):
            print(f"[skip]    {py_file.stem}: {reason}")
            continue
        print(f"[run]     {py_file.stem} ...", flush=True)
        start = time.perf_counter()
        try:
            nb = execute(py_file)
        except Exception as exc:  # noqa: BLE001 - report and continue with the next example
            print(f"[FAILED]  {py_file.stem}: {type(exc).__name__}: {exc}")
            failed.append(py_file.stem)
            continue
        target = output_dir / f"{py_file.stem}.ipynb"
        nbformat.write(nb, target)
        executed.append(target)
        print(f"[done]    {py_file.stem} ({time.perf_counter() - start:.0f} s)")
    print(f"\nTotal: {time.perf_counter() - total:.0f} s, notebooks in {output_dir}")

    if args.release and executed and not args.no_stage:
        stage(executed)
    if failed:
        print(f"{len(failed)} example(s) failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
