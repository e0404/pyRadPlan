#!/bin/sh
# Container entrypoint: runs the given command (by default the pyRadPlan GUI) on a virtual X
# server and serves its window to web browsers with Xpra. The container exits when the
# command does, so each container is one session.
#
# Environment (all optional):
#   XPRA_PORT      port of the web client, 10000 by default
#   XPRA_BIND      address the web client listens on, 0.0.0.0 by default; 127.0.0.1 when a
#                  proxy in the same container authenticates the users (hub-session.sh)
#   XPRA_PASSWORD  password the web client asks for; without it anyone who can reach the
#                  port can use the session
#   NUMBA_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS
#                  threads of the computations, by default the container's CPU limit
set -eu

# The libraries start a thread per CPU of the host, not of the container's CPU limit. With
# more threads than the limit allows, the kernel pauses the whole container, the Xpra server
# included, for the rest of each scheduling period, and the session stops responding.
cpu_limit=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo max)
quota=${cpu_limit% *}
period=${cpu_limit#* }
if [ "$quota" != max ]; then
    threads=$(( (quota + period - 1) / period ))
    cpus=$(nproc)
    [ "$threads" -le "$cpus" ] || threads=$cpus
    : "${NUMBA_NUM_THREADS:=$threads}" "${OMP_NUM_THREADS:=$threads}" "${OPENBLAS_NUM_THREADS:=$threads}"
    export NUMBA_NUM_THREADS OMP_NUM_THREADS OPENBLAS_NUM_THREADS
fi

XPRA_PORT="${XPRA_PORT:-10000}"
XPRA_BIND="${XPRA_BIND:-0.0.0.0}"
# Xpra keeps its sockets here. Always a directory of our own: one inherited from the
# environment may belong to someone else, and then the chmod fails.
XDG_RUNTIME_DIR="/tmp/runtime-$(id -u)"
export XDG_RUNTIME_DIR
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

if [ -n "${XPRA_PASSWORD:-}" ]; then
    auth="env"
else
    if [ "$XPRA_BIND" != "127.0.0.1" ]; then
        echo "WARNING: XPRA_PASSWORD is not set, anyone who can reach port $XPRA_PORT can use this session." >&2
    fi
    auth=none
fi

# Xpra accepts one command string. Quote each argument so paths containing spaces and
# shell metacharacters survive the conversion from Docker's argument vector.
start_child=$(python3 -c 'import shlex, sys; print(shlex.join(sys.argv[1:]))' "$@")

exec xpra start :100 \
    --daemon=no \
    --bind-tcp="$XPRA_BIND:$XPRA_PORT" --tcp-auth="$auth" \
    --html=on \
    --exit-with-children=yes \
    --start-child="$start_child" \
    --mdns=no --notifications=no --pulseaudio=no --speaker=off --microphone=off \
    --webcam=no --printing=no
