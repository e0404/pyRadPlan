#!/bin/sh
# Session server of the JupyterHub image (Dockerfile.hub). JupyterHub starts this script in
# place of jupyterhub-singleuser: jupyter-server-proxy's standalone proxy logs the user in
# through the hub, reports activity to it and forwards the browser to the Xpra web client
# of the pyRadPlan GUI, which listens on localhost only. Arguments are passed on to the
# proxy (for example --debug).
#
# The proxy starts the GUI with the first request but does not restart it, so this script
# ends the session when the GUI is closed: JupyterHub then shows the server as stopped and
# offers to start a new one.
set -eu

# Starting the GUI takes longer than the proxy's default of 5 s. The proxy's own activity
# reports always carry the current time, which would keep every session from being culled;
# without them JupyterHub takes the activity from its proxy, where the Xpra web client keeps
# traffic going while its tab is open.
jupyter-standaloneproxy --timeout=120 --activity-interval=0 "$@" -- \
    env XPRA_BIND=127.0.0.1 "XPRA_PORT={port}" entrypoint.sh pyRadPlanGUI TG119 &
proxy=$!
trap 'kill "$proxy" 2>/dev/null' INT TERM

started=no
while kill -0 "$proxy" 2>/dev/null; do
    if pgrep -f "xpra start :100" >/dev/null; then
        started=yes
    elif [ "$started" = yes ]; then
        echo "The pyRadPlan GUI has exited, ending the session." >&2
        kill "$proxy"
        break
    fi
    # In the background, so that a TERM from Kubernetes is handled at once.
    sleep 5 &
    wait $! || true
done
wait "$proxy" || true
