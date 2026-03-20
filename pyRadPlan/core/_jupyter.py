"""Jupyter notebook compatibility utilities."""


def detect_jupyter():
    """Check if running inside a Jupyter notebook."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and "zmq" in type(shell).__module__
    except (ImportError, AttributeError):
        return False
