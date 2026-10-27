#!/bin/sh
# Smoke test of the image, shared by the GitHub and GitLab pipelines so that both check the
# same things. Run it against a started container (the default name is "web"):
#
#   docker run -d --name web -e XPRA_PASSWORD=ci-test pyradplan-web
#   sh packaging/container/smoke_test.sh [container]
#
# It waits for the web client and for the GUI's main window, which Xpra lists once it
# exists (after the splash screen), then closes the GUI and checks that the container ends
# with exit code 0, as it is one session. Plain sh: GitLab runs it in an Alpine container.
#
#   SMOKE_ATTEMPTS  checks for the main window, 60 by default
#   SMOKE_INTERVAL  seconds between the checks, 5 by default
set -eu

container="${1:-web}"
attempts="${SMOKE_ATTEMPTS:-60}"
interval="${SMOKE_INTERVAL:-5}"
runtime_dir=/tmp/runtime-10001

i=0
while :; do
    i=$((i + 1))
    title=$(docker exec -e XDG_RUNTIME_DIR="$runtime_dir" "$container" xpra info :100 2>/dev/null \
            | grep -E '^windows\.[0-9]+\.title=pyRadPlan \(NOT FOR CLINICAL USE' || true)
    if [ -n "$title" ] \
            && docker exec "$container" curl -fsS -o /dev/null http://localhost:10000/ 2>/dev/null; then
        echo "web client and main window up after $((i * interval)) s: $title"
        break
    fi
    if [ "$i" -ge "$attempts" ]; then
        echo "no main window within $((attempts * interval)) s" >&2
        exit 1
    fi
    sleep "$interval"
done

docker exec "$container" sh -c "kill \"\$(cat $runtime_dir/xpra/100/pyRadPlanGUI.pid)\""
code=$(timeout 60 docker wait "$container")
echo "container exit code: $code"
[ "$code" = "0" ]
