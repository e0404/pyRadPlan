"""Jupyter notebook compatibility utilities."""

try:
    from IPython import get_ipython

    _IPYTHON_AVAILABLE = True
except (ImportError, AttributeError):
    _IPYTHON_AVAILABLE = False


def detect_jupyter():
    """Check if running inside a Jupyter notebook."""
    if _IPYTHON_AVAILABLE:
        shell = get_ipython()
        return shell is not None and "zmq" in type(shell).__module__
    else:
        return False
