"""Run Briefcase with a workaround for ``rcedit`` failing on relative paths.

On some Windows machines (seen on Windows 11 build 26200 with rcedit 1.1.1 and
2.0.0) ``rcedit`` aborts with ``Fatal error: Unable to commit changes`` when the
executable is given as a plain relative path such as ``src\pyRadPlan.exe``,
which is what Briefcase passes while "Setting stub app details". The same
command succeeds with an absolute path, so this wrapper rewrites that one
argument and otherwise runs the normal Briefcase command line::

    python scripts/run_briefcase.py build windows app

It is a no-op for every other command and platform.
"""

import sys
from pathlib import Path

from briefcase.__main__ import main
from briefcase.platforms.windows import app as windows_app

_original_build_app = windows_app.WindowsAppBuildCommand.build_app


def _build_app(self, app, **kwargs):
    subprocess_run = self.tools.subprocess.run
    bundle_path = self.bundle_path(app)

    def run_with_absolute_binary(args, *run_args, **run_kwargs):
        if len(args) > 1 and Path(args[0]).name.startswith("rcedit"):
            args = [args[0], bundle_path / args[1], *args[2:]]
        return subprocess_run(args, *run_args, **run_kwargs)

    self.tools.subprocess.run = run_with_absolute_binary
    try:
        return _original_build_app(self, app, **kwargs)
    finally:
        self.tools.subprocess.run = subprocess_run


windows_app.WindowsAppBuildCommand.build_app = _build_app

if __name__ == "__main__":
    sys.argv[0] = "briefcase"
    main()
