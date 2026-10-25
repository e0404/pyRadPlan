#!/bin/sh
# Container entrypoint: runs the given command (by default the pyRadPlan GUI) on a virtual X
# server and serves its window to web browsers with Xpra. The container exits when the
# command does, so each container is one session.
#
# Environment (all optional):
#   XPRA_PORT      port of the web client, 10000 by default
#   XPRA_PASSWORD  password the web client asks for; without it anyone who can reach the
#                  port can use the session
set -eu

XPRA_PORT="${XPRA_PORT:-10000}"
# Xpra keeps its sockets here. Always a directory of our own: one inherited from the
# environment may belong to someone else, and then the chmod fails.
XDG_RUNTIME_DIR="/tmp/runtime-$(id -u)"
export XDG_RUNTIME_DIR
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

if [ -n "${XPRA_PASSWORD:-}" ]; then
    auth="env"
else
    echo "WARNING: XPRA_PASSWORD is not set, anyone who can reach port $XPRA_PORT can use this session." >&2
    auth=allow
fi

# Xpra accepts one command string. Quote each argument so paths containing spaces and
# shell metacharacters survive the conversion from Docker's argument vector.
start_child=$(python3 -c 'import shlex, sys; print(shlex.join(sys.argv[1:]))' "$@")

exec xpra start :100 \
    --daemon=no \
    --bind-tcp="0.0.0.0:$XPRA_PORT" --tcp-auth="$auth" \
    --html=on \
    --exit-with-children=yes \
    --start-child="$start_child" \
    --mdns=no --notifications=no --pulseaudio=no --speaker=off --microphone=off \
    --webcam=no --printing=no
