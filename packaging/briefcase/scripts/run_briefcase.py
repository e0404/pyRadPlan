r"""Run Briefcase with two additions to building the Windows app.

Use it in place of the ``briefcase`` command::

    python scripts/run_briefcase.py build windows app

It is a no-op for every other command and platform.

``rcedit`` workaround
    On some Windows machines (seen on Windows 11 build 26200 with rcedit 1.1.1 and
    2.0.0) ``rcedit`` aborts with ``Fatal error: Unable to commit changes`` when the
    executable is given as a plain relative path such as ``src\pyRadPlan.exe``,
    which is what Briefcase passes while "Setting stub app details". The same
    command succeeds with an absolute path, so that one argument is rewritten.

C++ runtime next to the executable
    SimpleITK links the system's ``msvcp140.dll``. On machines with an outdated Visual C++
    Redistributable, Qt then runs on a runtime older than it needs and fails to load. The
    runtime PySide6 ships is copied next to ``pyRadPlan.exe``, which Windows searches before
    the system directory, so every package in the app uses it.
"""

import shutil
import sys
from pathlib import Path


def copy_msvc_runtime(app_packages: Path, target: Path) -> list[Path]:
    """Copy the ``msvcp140*.dll`` that PySide6 ships from ``app_packages`` into ``target``."""
    dlls = sorted((app_packages / "shiboken6").glob("msvcp140*.dll"))
    if not dlls:
        raise FileNotFoundError(f"No msvcp140*.dll in {app_packages / 'shiboken6'}")
    return [Path(shutil.copy2(dll, target / dll.name)) for dll in dlls]


def _patch_windows_build(windows_app) -> None:
    original_build_app = windows_app.WindowsAppBuildCommand.build_app

    def build_app(self, app, **kwargs):
        subprocess_run = self.tools.subprocess.run
        bundle_path = self.bundle_path(app)

        def run_with_absolute_binary(args, *run_args, **run_kwargs):
            if len(args) > 1 and Path(args[0]).name.startswith("rcedit"):
                args = [args[0], bundle_path / args[1], *args[2:]]
            return subprocess_run(args, *run_args, **run_kwargs)

        self.tools.subprocess.run = run_with_absolute_binary
        try:
            result = original_build_app(self, app, **kwargs)
        finally:
            self.tools.subprocess.run = subprocess_run

        copied = copy_msvc_runtime(self.app_packages_path(app), self.binary_path(app).parent)
        self.console.info(
            f"Copied {', '.join(dll.name for dll in copied)} next to the executable",
            prefix=app.app_name,
        )
        return result

    windows_app.WindowsAppBuildCommand.build_app = build_app


if __name__ == "__main__":
    from briefcase.__main__ import main
    from briefcase.platforms.windows import app as windows_app

    _patch_windows_build(windows_app)
    sys.argv[0] = "briefcase"
    # main() returns Briefcase's exit code instead of exiting: dropping it would report
    # every failure (a failing smoke test, a package that cannot be built) as success.
    sys.exit(main())
