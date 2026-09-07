"""Static GUI assets (logos, icons) bundled with pyRadPlan.

Assets are plain files shipped inside the wheel (see ``package-data`` in
``pyproject.toml``).  Use :func:`asset_path` to obtain a real filesystem path
suitable for Qt loaders such as ``QSvgWidget`` / ``QIcon``::

    from pyRadPlan.gui.assets import asset_path
    QSvgWidget(str(asset_path("logos", "dkfz_logo_blue.svg")))

Why not just ``files(__name__) / ...``?
--------------------------------------
:func:`importlib.resources.files` yields a path that is only real when the
package lives on disk.  Under a zip import (zipapp, some PyInstaller / frozen
bundles) the resource has no filesystem path, so handing ``str(...)`` of it to Qt
would fail.  The blessed fix, :func:`importlib.resources.as_file`, materialises a
real path but only for the duration of its ``with`` block -- which is awkward for
Qt: ``QIcon`` reads the path *lazily*, long after such a block would have closed.

To stay zip-safe while still returning a plain, persistent path, asset extraction
is bound to a module-level :class:`~contextlib.ExitStack` that is closed at
interpreter exit.  On a normal on-disk install ``as_file`` simply returns the real
path (no extraction); under a zip import it extracts to a temp file that lives for
the whole process, so both eager (``QSvgWidget``) and lazy (``QIcon``) consumers
work.  Results are cached so each asset is materialised at most once.

Folder layout
-------------
``logos/``
    Branding artwork (pyRadPlan + DKFZ logos, light/dark variants).
``icons/``
    Toolbar / action icons (reserved for future use).
"""

from __future__ import annotations

import atexit
from contextlib import ExitStack
from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

ASSETS_DIR: Traversable = files(__name__)
LOGO_DIR: Traversable = ASSETS_DIR / "logos"

# Keeps any temp files materialised from a zip import alive until the process
# exits, so the paths handed to Qt stay valid for the application's lifetime.
_file_manager = ExitStack()
atexit.register(_file_manager.close)


@lru_cache(maxsize=None)
def asset_path(*parts: str) -> Path:
    """Return a real filesystem path to the bundled asset addressed by *parts*.

    Parameters
    ----------
    *parts:
        Path components relative to the assets directory, e.g.
        ``asset_path("logos", "dkfz_logo_blue.svg")``.

    Returns
    -------
    pathlib.Path
        A real path valid for the lifetime of the process (see module docstring),
        accepted by ``str()`` / ``os.fspath`` and therefore by Qt loaders.

    Raises
    ------
    FileNotFoundError
        If the resolved asset does not exist.
    """
    resource = ASSETS_DIR.joinpath(*parts)
    if not resource.is_file():
        raise FileNotFoundError(f"GUI asset not found: {resource}")
    return _file_manager.enter_context(as_file(resource))


__all__ = ["ASSETS_DIR", "LOGO_DIR", "asset_path"]
