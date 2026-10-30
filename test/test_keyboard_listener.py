"""Tests for the POSIX terminal path of pyRadPlan.util.keyboard_listener."""

import os
import select
import signal
import sys
import time

import pytest

pty = pytest.importorskip("pty", reason="needs a POSIX pseudo terminal")

_CHILD = """
import time
from pyRadPlan.util.keyboard_listener import KeyboardListener
kb = KeyboardListener(allow_keyboard_cancel=True)
kb.initialize()
print("READY", flush=True)
t0 = time.time()
while time.time() - t0 < 2 and not kb.stop_event.is_set():
    time.sleep(0.05)
print("CANCELLED" if kb.stop_event.is_set() else "RUNNING", flush=True)
kb.finalize()
"""


def _read_until(fd, token, timeout=20.0):
    out = b""
    end = time.time() + timeout
    while token not in out and time.time() < end:
        if select.select([fd], [], [], 0.05)[0]:
            try:
                out += os.read(fd, 4096)
            except OSError:
                break
    return out


def _press_in_terminal(*chunks: bytes) -> str:
    """Run a listener in a child owning a pseudo terminal and send *chunks* into it."""
    pid, fd = pty.fork()
    if pid == 0:
        os.execv(sys.executable, [sys.executable, "-c", _CHILD])
    try:
        assert b"READY" in _read_until(fd, b"READY")
        time.sleep(0.2)
        for chunk in chunks:
            os.write(fd, chunk)
            time.sleep(0.02)
        out = _read_until(fd, b"CANCELLED", timeout=5)
        return "CANCELLED" if b"CANCELLED" in out else "RUNNING"
    finally:
        os.waitpid(pid, 0)
        os.close(fd)


@pytest.mark.parametrize("keys", [b"q", b"\x1b"], ids=["cancel-key", "esc"])
def test_cancel_keys_stop(keys):
    """The cancel key and a lone ESC stop the optimization."""
    assert _press_in_terminal(keys) == "CANCELLED"


@pytest.mark.parametrize("keys", [b"\x1b[A", b"\x1bOP", b"x"], ids=["arrow", "f1", "other"])
def test_escape_sequences_and_other_keys_do_not_stop(keys):
    """An arrow key arrives as one escape sequence; it must not read as a lone ESC."""
    assert _press_in_terminal(keys) == "RUNNING"


@pytest.mark.parametrize(
    "chunks",
    [(b"\x1b", b"[A"), (b"\x1b[", b"A"), (b"\x1b", b"[", b"1;5", b"A"), (b"\x1b[", b"q")],
    ids=["after-esc", "after-bracket", "several", "final-byte-is-cancel-key"],
)
def test_escape_sequences_split_across_reads_do_not_stop(chunks):
    """A slow connection (e.g. SSH) can deliver one escape sequence in several parts."""
    assert _press_in_terminal(*chunks) == "RUNNING"


def test_background_job_is_not_stopped_by_the_terminal():
    """Switching the terminal mode from a background job would stop it with SIGTTOU."""
    pid, fd = pty.fork()
    if pid == 0:
        # An ignored SIGTTOU is inherited (e.g. from GNU timeout) and would hide the stop.
        signal.signal(signal.SIGTTOU, signal.SIG_DFL)
        os.execvp("bash", ["bash", "--norc", "--noprofile", "-i"])
    try:
        # `wait` also returns once the job is stopped, reporting it; the quotes keep the
        # terminal's echo of this line from matching the end marker.
        command = f"{sys.executable} -c '{_CHILD}' > /dev/null 2>&1 &\nwait; echo DO''NE\n"
        os.write(fd, command.encode())
        out = _read_until(fd, b"DONE", timeout=30).decode(errors="replace")
        assert "DONE" in out and "stopped" not in out.lower(), out
    finally:
        os.kill(pid, signal.SIGKILL)
        # Close the pty first: on macOS the killed shell does not finish exiting while
        # its unread terminal output is pending, so waitpid would block forever
        os.close(fd)
        os.waitpid(pid, 0)
