"""Cross-platform keyboard listener utility for early-cancel support.

This module encapsulates non-blocking keyboard input handling on POSIX and
Windows platforms, exposing a simple class-based API that integrates with
long-running processes (e.g., optimizers) via a threading.Event.

Typical usage within a solver-like component:

    stop_event = Event()
    listener = KeyboardListener(
        stop_event=stop_event,
        allow_keyboard_cancel=True,
        cancel_key="q",
        allow_esc_cancel=True,
        logger=logger,  # optional
        component_name=lambda: self.name,  # optional; used only for logging
    )

    # Before starting the long-running operation
    stop_event.clear()
    listener.initialize()

    # ... do work, periodically check stop_event.is_set() ...

    # On exit (success or error)
    listener.finalize()

"""

from __future__ import annotations

from threading import Event, Thread
from typing import Optional, Callable

import logging
import os
import sys
import time

# Try POSIX-specific imports (termios/tty). If unavailable (e.g. Windows), fall back.
try:  # POSIX path
    import termios  # type: ignore
    import tty  # type: ignore
    import select  # type: ignore

    _HAS_POSIX_TERMIOS = True
except Exception:  # pragma: no cover - platform dependent
    termios = None  # type: ignore
    tty = None  # type: ignore
    select = None  # type: ignore
    _HAS_POSIX_TERMIOS = False

# Try Windows-specific msvcrt for non-blocking keyboard input
try:  # Windows path
    import msvcrt  # type: ignore

    _HAS_MS_KBHIT = True
except Exception:  # pragma: no cover - platform dependent
    msvcrt = None  # type: ignore
    _HAS_MS_KBHIT = False

__all__ = ["KeyboardListener"]

logger = logging.getLogger(__name__)

# Seconds without input after which an escape sequence is over. Its parts can arrive apart on
# a slow connection (e.g. SSH); the delay this adds to the ESC key is not noticeable.
_ESCAPE_SEQUENCE_GAP = 0.1


def _in_terminal_foreground() -> bool:
    """Whether this process owns the terminal on stdin.

    Changing the terminal mode from a background job (``python run.py &``) makes the
    kernel stop the whole process with ``SIGTTOU``.
    """
    try:
        return os.tcgetpgrp(sys.stdin.fileno()) == os.getpgrp()
    except OSError:
        return False


class KeyboardListener:
    """Cross-platform non-blocking keyboard listener.

    Parameters
    ----------
    stop_event : threading.Event
        Event to set when cancellation is requested.
    allow_keyboard_cancel : bool, default=False
        If False, the listener won't start. If True but platform support is
        not available, will be disabled with a log message.
    cancel_key : str, default="q"
        Single-character key to trigger cancellation.
    allow_esc_cancel : bool, default=True
        Whether to also allow the ESC key to trigger cancellation.
    logger : logging.Logger, optional
        Logger for messages; defaults to this module's logger.
    component_name : str | Callable[[], str], optional
        Name of the component for logging, or callable that returns it lazily.
    """

    def __init__(
        self,
        *,
        allow_keyboard_cancel: bool = False,
        cancel_key: str = "q",
        allow_esc_cancel: bool = True,
        component_name: Optional[Callable[[], str] | str] = None,
    ) -> None:
        self.stop_event = Event()
        self.allow_keyboard_cancel = allow_keyboard_cancel
        self.cancel_key = cancel_key
        self.allow_esc_cancel = allow_esc_cancel
        self.logger = logging.getLogger(__name__)
        self._component_name = component_name

        self._thread: Optional[Thread] = None

        # Disable if platform doesn't support non-blocking keyboard input
        if self.allow_keyboard_cancel and not (_HAS_POSIX_TERMIOS or _HAS_MS_KBHIT):
            self.allow_keyboard_cancel = False
            self.logger.info(
                "Imports for keyboard cancellation not available or platform unsupported! "
                "Resume with disabled keyboard cancellation."
            )

    # --- Public API ---
    def initialize(self) -> None:
        """Start the background keyboard listener thread if enabled."""

        self.stop_event.clear()
        if not self.allow_keyboard_cancel:
            self.logger.debug("Keyboard cancellation disabled by configuration.")
            return

        # Choose platform-specific listener
        if _HAS_POSIX_TERMIOS and sys.stdin.isatty() and _in_terminal_foreground():
            target = self._kb_listener_posix
        elif _HAS_MS_KBHIT and os.name == "nt":
            target = self._kb_listener_windows
        else:
            self.logger.info(
                "Keyboard cancellation requested but unavailable (no non-blocking keyboard "
                "input on this platform, or stdin is not a terminal in the foreground). "
                "Continuing without it."
            )
            return

        self._thread = Thread(target=target, daemon=True)
        self._thread.start()
        if self.allow_esc_cancel:
            self.logger.info(f"Press '{self.cancel_key}' or ESC to stop optimization early.")
        else:
            self.logger.info(f"Press '{self.cancel_key}' to stop optimization early.")

    def finalize(self) -> None:
        """Stop the background keyboard listener thread."""
        aborted_by_user = self.stop_event.is_set()
        if self._thread and self._thread.is_alive():
            self.stop_event.set()
            self._thread.join(timeout=0.2)
        if aborted_by_user:
            logger.info("Optimization aborted by user. Returning last known iterate.")

    def cancel(self) -> None:
        """Request cancellation and log a message."""
        name = None
        if callable(self._component_name):
            try:
                name = self._component_name()
            except Exception:
                name = None
        elif isinstance(self._component_name, str):
            name = self._component_name

        if name:
            self.logger.info(f"Cancellation requested. Stopping Optimization of {name}.")
        else:
            self.logger.info("Cancellation requested.")
        self.stop_event.set()

    # --- Platform specific helpers ---
    def _kb_listener_posix(self):  # pragma: no cover - platform dependent
        fd = sys.stdin.fileno()
        old_attrs = termios.tcgetattr(fd)  # type: ignore[attr-defined]
        try:
            tty.setcbreak(fd)  # type: ignore[attr-defined]
            while not self.stop_event.is_set():
                r, _, _ = select.select([fd], [], [], 0.1)  # type: ignore[attr-defined]
                if not r:
                    continue
                # Read the descriptor itself: sys.stdin buffers the rest of an escape
                # sequence where select() no longer sees it.
                keys = os.read(fd, 64).decode(errors="ignore")
                if keys.startswith("\x1b"):
                    # Consume the rest of an escape sequence (arrow keys, ...): only an ESC
                    # that nothing follows is the ESC key.
                    while not self.stop_event.is_set():
                        r, _, _ = select.select([fd], [], [], _ESCAPE_SEQUENCE_GAP)  # type: ignore[attr-defined]
                        chunk = os.read(fd, 64) if r else b""
                        if not chunk:
                            break
                        keys += chunk.decode(errors="ignore")
                    cancel = self.allow_esc_cancel and keys == "\x1b"
                else:
                    cancel = self.cancel_key in keys
                if cancel:
                    self.cancel()
                    break
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)  # type: ignore[attr-defined]
            except Exception:
                pass

    def _kb_listener_windows(self):  # pragma: no cover - platform dependent
        while not self.stop_event.is_set():
            if msvcrt and msvcrt.kbhit():  # type: ignore[attr-defined]
                ch = msvcrt.getwch()  # type: ignore[attr-defined]
                if ch == self.cancel_key or (self.allow_esc_cancel and ch in {"\x1b", "\033"}):
                    self.cancel()
                    break
            time.sleep(0.1)
