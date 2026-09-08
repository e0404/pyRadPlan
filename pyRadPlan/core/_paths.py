"""Resolution of pyRadPlan's writable data directory.

pyRadPlan keeps user-writable data (downloaded model weights and, in the future,
datasets it fetches such as patients or phantoms) under a single *data root*.
The default is a ``.pyradplan`` directory in the user's home on every platform;
it can be relocated by setting the ``PYRADPLAN_DATA_DIR`` environment variable.

Conventional subdirectories under the data root:

``ai_models/``
    Local / downloaded AI model repositories (see :mod:`pyRadPlan.ai`).
``patients/``, ``phantoms/``, ``machines/``
    Reserved for datasets pyRadPlan may download in the future.
``cache/``
    Reserved for transient caches.

Consumers obtain their subdirectory via :func:`get_data_subdir`, which creates
it lazily, so merely importing pyRadPlan never touches the filesystem.

This module deliberately depends only on the standard library. Should a
platform-native layout (e.g. via :mod:`platformdirs`) ever be desired, only
:func:`get_data_dir` needs to change.
"""

import os
from pathlib import Path

#: Default data root when ``PYRADPLAN_DATA_DIR`` is unset.
DEFAULT_DATA_DIR = Path.home() / ".pyradplan"


def get_data_dir() -> Path:
    """Return the root directory for pyRadPlan's writable data.

    Reads the ``PYRADPLAN_DATA_DIR`` environment variable; a leading ``~`` is
    expanded. An unset or blank value falls back to :data:`DEFAULT_DATA_DIR`.

    Returns
    -------
    Path
        The data root. Not guaranteed to exist yet (see :func:`get_data_subdir`).
    """
    env = os.environ.get("PYRADPLAN_DATA_DIR")
    if env and env.strip():
        return Path(env).expanduser()
    return DEFAULT_DATA_DIR


def get_data_subdir(name: str, *, create: bool = True) -> Path:
    """Return a named subdirectory of the data root.

    Parameters
    ----------
    name : str
        Subdirectory name (e.g. ``"ai_models"``).
    create : bool, optional, default=True
        Create the directory (and the data root) if it does not exist.

    Returns
    -------
    Path
        ``<data_dir>/<name>``.
    """
    path = get_data_dir() / name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
