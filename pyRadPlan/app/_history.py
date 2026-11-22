"""Session history: numbered log events and the metrics of the last optimization."""

from __future__ import annotations

import collections
import itertools
import threading
from typing import Any, Optional

#: Log events kept for ``session.history`` (the Qt log console keeps 2000 lines).
MAX_LOG_EVENTS = 2000


class SessionHistory:
    """What a frontend needs to rebuild the log panel and the optimization status window.

    Thread-safe: the worker thread of a host records while another thread reads.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._seq = itertools.count(1)
        self.log: collections.deque[dict] = collections.deque(maxlen=MAX_LOG_EVENTS)
        self.optimization: Optional[dict[str, Any]] = None

    def number(self, event: dict) -> dict:
        """Give a log event its ``seq`` and keep it."""
        with self._lock:
            event["seq"] = next(self._seq)
            self.log.append(dict(event))
        return event

    def start_optimization(self, label: str) -> None:
        """Begin recording the metrics of a new optimization."""
        with self._lock:
            self.optimization = {"label": label, "points": [], "finished": False, "outcome": None}

    def add_point(self, data: dict[str, Any]) -> None:
        """Record one status report of the running optimization."""
        with self._lock:
            if self.optimization is not None and not self.optimization["finished"]:
                self.optimization["points"].append(dict(data))

    def finish_optimization(self, outcome: str) -> None:
        """Mark the optimization as ``"done"``, ``"stopped"`` or ``"failed"``."""
        with self._lock:
            if self.optimization is not None and not self.optimization["finished"]:
                self.optimization["finished"] = True
                self.optimization["outcome"] = outcome

    def as_dict(self, since: int = 0) -> dict[str, Any]:
        """Return the ``session.history`` result."""
        with self._lock:
            return {
                "log": [event for event in self.log if event.get("seq", 0) > since],
                "optimization": None
                if self.optimization is None
                else {**self.optimization, "points": list(self.optimization["points"])},
            }
